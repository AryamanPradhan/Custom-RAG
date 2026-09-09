"""Layer 02 - reading citation markers, then taking them off.

The marker is provenance inside the pipeline and clutter on screen, so it is
read first and stripped last. Two things are worth testing.

`build_citations` has to keep working on marked-up text, because it is what
decides which sources an answer leaned on - and an answer it finds nothing in
is deflected rather than shown.

`MarkerFilter` has to agree with `strip_markers` no matter where the token
boundaries land. A model streams `[`, `1`, `].` as three tokens as happily as
one, so the filter is checked against every possible split of the same answer
rather than against the one tokenisation someone thought of.
"""

from __future__ import annotations

import random

import pytest

from app.models.domain import Chunk, DocCategory, ScoredChunk, SourceKind
from app.pipeline.citations import MarkerFilter, build_citations, strip_markers

ANSWERS = [
    "Check-in is from 2:00 PM [1].",
    "The parking is staffed 24 hours [1], [2].",
    "Early check-in is free when available [1][2].",
    "The Loft allows dogs [2]; The Barn does not [3].",
    "Breakfast is included [1, 2].",
    "Rooms are en-suite [1], [2], [3].",
    "[1] Both are complimentary.",
    "Wifi is free [1], and parking is too [2].",
    "MG Marg - 2 km away, with cafes and handicrafts [3].",
    "I don't have that in the published information, so I'd rather not guess.",
    "The rate is [not published] here [1].",
    "See the note [ below [1].",
    "Sources [1] and [2] agree [3].",
]


def _split_everywhere(text: str):
    """Every single-cut tokenisation, plus the extremes."""
    yield [text]
    yield list(text)
    for i in range(1, len(text)):
        yield [text[:i], text[i:]]


def _run(tokens: list[str]) -> str:
    filt = MarkerFilter()
    return "".join(filt.feed(t) for t in tokens) + filt.flush()


class TestStripping:
    @pytest.mark.parametrize("answer", ANSWERS)
    def test_no_marker_survives(self, answer: str) -> None:
        assert "[1]" not in strip_markers(answer)
        assert "[2]" not in strip_markers(answer)
        assert "[3]" not in strip_markers(answer)

    @pytest.mark.parametrize(
        ("answer", "expected"),
        [
            ("Check-in is from 2:00 PM [1].", "Check-in is from 2:00 PM."),
            # The separator between two markers is orphaned once they are gone.
            ("The parking is staffed 24 hours [1], [2].", "The parking is staffed 24 hours."),
            ("Rooms are en-suite [1], [2], [3].", "Rooms are en-suite."),
            ("Early check-in is free [1][2].", "Early check-in is free."),
            ("Breakfast is included [1, 2].", "Breakfast is included."),
            # A comma doing real work in the sentence stays.
            ("Wifi is free [1], and parking is too [2].", "Wifi is free, and parking is too."),
            ("The Loft allows dogs [2]; The Barn does not [3].",
             "The Loft allows dogs; The Barn does not."),
            # A marker opening the answer must not leave it indented.
            ("[1] Both are complimentary.", "Both are complimentary."),
            # Brackets that are not markers are content.
            ("The rate is [not published] here [1].", "The rate is [not published] here."),
        ],
    )
    def test_reads_as_written(self, answer: str, expected: str) -> None:
        assert strip_markers(answer) == expected

    def test_an_answer_without_markers_is_untouched(self) -> None:
        plain = "I don't have that in the published information."
        assert strip_markers(plain) == plain


class TestStreamingFilter:
    @pytest.mark.parametrize("answer", ANSWERS)
    def test_agrees_with_the_whole_text_however_it_is_split(self, answer: str) -> None:
        """The contract. A marker split across three tokens must come out the
        same as one that arrived whole."""
        expected = strip_markers(answer)
        for tokens in _split_everywhere(answer):
            assert _run(tokens) == expected, f"split {tokens!r}"

    @pytest.mark.parametrize("answer", ANSWERS)
    def test_agrees_under_random_tokenisation(self, answer: str) -> None:
        rng = random.Random(0)
        expected = strip_markers(answer)
        for _ in range(40):
            tokens, rest = [], answer
            while rest:
                take = rng.randint(1, 4)
                tokens.append(rest[:take])
                rest = rest[take:]
            assert _run(tokens) == expected, f"split {tokens!r}"

    def test_text_is_released_as_it_arrives(self) -> None:
        """The point of streaming: a filter that buffered the whole answer to
        be safe would undo the reason tokens are streamed at all."""
        filt = MarkerFilter()
        assert filt.feed("Check-in is from ") == "Check-in is from"
        assert filt.feed("2pm") == " 2pm"

    def test_only_the_separator_before_a_marker_is_held(self) -> None:
        filt = MarkerFilter()
        filt.feed("Check-in is 2pm")
        # The space could precede a marker, so it waits.
        assert filt.feed(" ") == ""
        assert filt.feed("[1]") == ""
        assert filt.feed(" and") == " and"

    def test_a_held_space_is_released_when_no_marker_follows(self) -> None:
        filt = MarkerFilter()
        filt.feed("Rooms are quiet")
        assert filt.feed(" ") == ""
        assert filt.feed("and warm.") == " and warm."

    def test_an_unclosed_bracket_does_not_stall_the_stream(self) -> None:
        """A '[' that never closes would otherwise hold the rest of the answer
        hostage."""
        filt = MarkerFilter()
        out = filt.feed("The rate is [") + filt.feed("9" * 30)
        assert "[" in out

    def test_flush_drops_a_half_written_marker(self) -> None:
        filt = MarkerFilter()
        filt.feed("Check-in is 2pm [1")
        assert "[1" not in filt.flush()


def _chunk(text: str, unit: str | None = None) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(
            chunk_id="c1",
            doc_id="d1",
            property_id="casa-verde",
            text=text,
            uri="upload://faq.pdf",
            title="FAQ",
            heading_path=["Check-in"],
            category=DocCategory.POLICIES,
            source_kind=SourceKind.UPLOAD,
            position=0,
            token_estimate=20,
            unit=unit,
            fetched_at="2026-08-12",
        ),
        score=1.0,
    )


class TestBuildCitations:
    def test_reads_the_markers_the_model_wrote(self) -> None:
        chunks = [_chunk("A."), _chunk("B."), _chunk("C.")]
        cited = build_citations(chunks, "The answer draws on [1] and [3].")
        assert [c.index for c in cited] == [1, 3]

    def test_an_uncited_answer_cites_nothing(self) -> None:
        """What makes an uncited answer a deflection rather than a reply."""
        assert build_citations([_chunk("A.")], "I don't have that.") == []

    def test_it_must_run_before_stripping(self) -> None:
        """The ordering the whole design rests on: strip first and every
        answer looks uncited, so every answer would deflect."""
        chunks = [_chunk("Check-in is from 2pm.")]
        answer = "Check-in is from 2pm [1]."

        assert build_citations(chunks, answer)
        assert build_citations(chunks, strip_markers(answer)) == []
