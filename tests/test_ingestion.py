"""Chunking and cost-accounting tests.

Chunking is the highest-leverage thing in this pipeline: a chunk that splits a
room's name away from its pet policy cannot be rescued by better retrieval or a
better model.
"""

from __future__ import annotations

import pytest

from app.gateway.pricing import UnknownModelError, cost_usd, provider_for
from app.ingestion.chunker import (
    chunk_document,
    embedding_text,
    estimate_tokens,
    split_sections,
)
from app.ingestion.classify import classify_document, classify_heuristic
from app.models.domain import DocCategory, Document, SourceKind


def _doc(text: str, uri: str = "https://casaverde.com/rooms", title: str = "Rooms") -> Document:
    return Document(
        property_id="casa-verde",
        source_kind=SourceKind.WEBSITE,
        uri=uri,
        title=title,
        text=text,
    )


class TestSectionSplitting:
    def test_tracks_the_heading_stack(self) -> None:
        sections = split_sections(
            "# Rooms\n\nIntro.\n\n## The Barn\n\nSleeps 2.\n\n## The Loft\n\nSleeps 4."
        )
        paths = [s.heading_path for s in sections]
        assert ["Rooms"] in paths
        assert ["Rooms", "The Barn"] in paths
        assert ["Rooms", "The Loft"] in paths

    def test_pops_back_out_to_a_shallower_heading(self) -> None:
        sections = split_sections(
            "# Rooms\n\n## The Barn\n\nSleeps 2.\n\n# Policies\n\nNo smoking."
        )
        assert sections[-1].heading_path == ["Policies"]

    def test_text_before_the_first_heading_is_kept(self) -> None:
        sections = split_sections("Welcome to Casa Verde.\n\n# Rooms\n\nSleeps 2.")
        assert sections[0].heading_path == []
        assert "Welcome" in sections[0].text

    def test_document_with_no_headings_is_one_section(self) -> None:
        assert len(split_sections("Just a paragraph of prose.")) == 1


class TestChunking:
    def test_unrelated_sections_are_not_merged(self) -> None:
        """The failure that matters: pet policy absorbed into cancellation."""
        doc = _doc(
            "# Pet policy\n\nDogs are welcome in The Loft.\n\n"
            "# Cancellation\n\nFree cancellation up to 48 hours before arrival."
        )
        chunks = chunk_document(doc, target_tokens=450)
        combined = [c for c in chunks if "Dogs" in c.text and "cancellation" in c.text.lower()]
        assert not combined, "pet policy and cancellation ended up in one chunk"

    def test_sibling_sections_pack_together(self) -> None:
        """Six one-line FAQ answers should not become six useless chunks."""
        doc = _doc(
            "# FAQ\n\n## Wi-Fi\n\nFree throughout.\n\n## Parking\n\nOn site.\n\n"
            "## Breakfast\n\nServed 8-10.\n\n## Towels\n\nProvided."
        )
        chunks = chunk_document(doc, target_tokens=450)
        assert len(chunks) == 1

    def test_oversized_section_is_split(self) -> None:
        doc = _doc("# Long\n\n" + ("This is a sentence about the property. " * 400))
        chunks = chunk_document(doc, target_tokens=200)
        assert len(chunks) > 1
        assert all(c.token_estimate <= 260 for c in chunks)

    def test_every_chunk_carries_provenance(self) -> None:
        doc = _doc("# Rooms\n\n## The Barn\n\nSleeps 2.")
        doc.unit = "The Barn"
        chunk = chunk_document(doc)[0]
        assert chunk.property_id == "casa-verde"
        assert chunk.uri == "https://casaverde.com/rooms"
        assert chunk.unit == "The Barn"
        assert chunk.fetched_at
        assert chunk.metadata["content_hash"]

    def test_chunk_ids_are_stable_across_runs(self) -> None:
        """Re-crawling an unchanged page must upsert onto the same points, not
        duplicate the corpus."""
        text = "# Rooms\n\nSleeps 2."
        first = chunk_document(_doc(text))
        second = chunk_document(_doc(text))
        assert [c.chunk_id for c in first] == [c.chunk_id for c in second]

    def test_empty_document_yields_nothing(self) -> None:
        assert chunk_document(_doc("   ")) == []

    def test_positions_are_sequential(self) -> None:
        doc = _doc("# A\n\nOne.\n\n# B\n\nTwo.\n\n# C\n\nThree.")
        chunks = chunk_document(doc)
        assert [c.position for c in chunks] == list(range(len(chunks)))


class TestEmbeddingText:
    def test_prepends_the_heading_trail(self) -> None:
        """Without this a chunk reading "11:00 AM" is unretrievable."""
        doc = _doc("# Check-out\n\n11:00 AM")
        text = embedding_text(chunk_document(doc)[0])
        assert "Rooms" in text
        assert "Check-out" in text
        assert "11:00 AM" in text


class TestTokenEstimate:
    def test_never_returns_zero(self) -> None:
        assert estimate_tokens("") == 1
        assert estimate_tokens("hi") == 1


class TestHeuristicClassification:
    @pytest.mark.parametrize(
        ("uri", "title", "expected"),
        [
            ("https://x.com/cancellation-policy", "Cancellation", DocCategory.POLICIES),
            ("https://x.com/rooms/the-barn", "The Barn", DocCategory.ROOMS),
            ("https://x.com/dining", "Restaurant", DocCategory.DINING),
            ("https://x.com/getting-here", "Directions", DocCategory.LOCATION),
            ("https://x.com/faq", "FAQ", DocCategory.FAQ),
            ("https://x.com/contact", "Contact us", DocCategory.CONTACT),
        ],
    )
    def test_places_common_pages(self, uri: str, title: str, expected: DocCategory) -> None:
        assert classify_heuristic(_doc("body", uri=uri, title=title)) is expected

    def test_returns_none_when_unsure_rather_than_guessing(self) -> None:
        doc = _doc("Some prose.", uri="https://x.com/x9", title="Untitled")
        assert classify_heuristic(doc) is None

    def test_body_text_does_not_drive_the_category(self) -> None:
        """Nearly every hotel page mentions rooms; matching on body would turn
        the whole site into DocCategory.ROOMS."""
        doc = _doc(
            "Our rooms are lovely and breakfast is included.",
            uri="https://x.com/about-us",
            title="About us",
        )
        assert classify_heuristic(doc) is None


class TestCostAccounting:
    def test_prices_each_provider_separately(self) -> None:
        openai = cost_usd("gpt-4o-mini", input_tokens=1_000_000, output_tokens=0)
        anthropic = cost_usd("claude-haiku-4-5", input_tokens=1_000_000, output_tokens=0)
        assert openai == pytest.approx(0.15)
        assert anthropic == pytest.approx(1.00)

    def test_cached_reads_are_cheaper_than_fresh_input(self) -> None:
        fresh = cost_usd("claude-haiku-4-5", input_tokens=1_000_000)
        cached = cost_usd("claude-haiku-4-5", cache_read_tokens=1_000_000)
        assert cached == pytest.approx(fresh * 0.1)

    def test_unknown_model_raises_rather_than_billing_zero(self) -> None:
        """A silent fallback would make spend invisible to the daily cap."""
        with pytest.raises(UnknownModelError):
            cost_usd("some-new-model", input_tokens=1000)

    def test_provider_routing(self) -> None:
        assert provider_for("gpt-4o-mini") == "openai"
        assert provider_for("claude-sonnet-5") == "anthropic"
        assert provider_for("gemini-embedding-2") == "google"


class TestClassificationCost:
    """Regression: the heuristic short-circuit was unreachable whenever a
    gateway was wired in - which is always, in production. A 300-page crawl
    made 300 billed calls."""

    async def test_settled_pages_skip_the_model(self) -> None:
        class ExplodingGateway:
            async def complete(self, *a, **kw):
                raise AssertionError("the model must not be called for a settled page")

        doc = _doc("body", uri="https://x.com/cancellation-policy", title="Cancellation")
        category, unit = await classify_document(doc, ExplodingGateway())
        assert category is DocCategory.POLICIES
        assert unit is None

    async def test_room_pages_still_ask_for_the_unit_name(self) -> None:
        class CountingGateway:
            def __init__(self) -> None:
                self.calls = 0

            async def complete(self, *a, **kw):
                self.calls += 1

                class R:
                    text = '{"category": "rooms", "unit": "The Barn"}'

                return R()

        gateway = CountingGateway()
        doc = _doc("body", uri="https://x.com/rooms/the-barn", title="The Barn")
        category, unit = await classify_document(doc, gateway)
        assert gateway.calls == 1
        assert category is DocCategory.ROOMS
        assert unit == "The Barn"
