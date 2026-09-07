"""Property registry, origin allowlist and spend ledger.

The allowlist is the only thing identifying which client a public, anonymous
request belongs to, and the ledger is the only thing stopping an abusive
caller running up an unbounded bill. Both are worth testing properly.
"""

from __future__ import annotations

import pytest
import pytest_asyncio

from app.config import Settings
from app.gateway.budget import UsageExceeded, UsageLimiter
from app.gateway.llm_gateway import LLMGateway
from app.models.domain import Chunk, DocCategory, SourceKind
from app.retrieval.store import VectorStore
from app.storage.db import Database
from app.storage.properties import ContactRoute, PropertyRepository, normalise_origin


@pytest_asyncio.fixture
async def repo(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    await db.connect()
    yield PropertyRepository(db)
    await db.close()


class TestOriginNormalisation:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("https://casaverde.com", "https://casaverde.com"),
            ("https://CasaVerde.com/", "https://casaverde.com"),
            ("https://casaverde.com/rooms/the-barn", "https://casaverde.com"),
            ("casaverde.com", "https://casaverde.com"),
            ("http://localhost:3000", "http://localhost:3000"),
            ("https://casaverde.com:443", "https://casaverde.com"),
        ],
    )
    def test_operator_input_variants_collapse(self, raw: str, expected: str) -> None:
        """Operators paste whatever is in the address bar; browsers send a bare
        origin. Both must resolve to the same key."""
        assert normalise_origin(raw) == expected

    def test_rejects_junk(self) -> None:
        assert normalise_origin("") == ""
        assert normalise_origin("   ") == ""


class TestRegistry:
    async def test_round_trip(self, repo: PropertyRepository) -> None:
        created = await repo.create(
            "casa-verde",
            "Casa Verde",
            allowed_origins=["https://casaverde.com"],
            contact_route=ContactRoute(phone="+44 1234 567890"),
            daily_spend_cap_usd=3.0,
        )
        assert created.display_name == "Casa Verde"

        loaded = await repo.get("casa-verde")
        assert loaded is not None
        assert loaded.contact_route.phone == "+44 1234 567890"
        assert loaded.daily_spend_cap_usd == 3.0
        assert loaded.allowed_origins == ["https://casaverde.com"]

    async def test_resolve_origin_finds_the_property(self, repo: PropertyRepository) -> None:
        await repo.create(
            "casa-verde", "Casa Verde", allowed_origins=["https://casaverde.com"]
        )
        assert await repo.resolve_origin("https://casaverde.com") == "casa-verde"
        # A browser sends the bare origin even when the visitor is deep in the site.
        assert await repo.resolve_origin("https://casaverde.com/rooms") == "casa-verde"

    async def test_unregistered_origin_resolves_to_nothing(
        self, repo: PropertyRepository
    ) -> None:
        await repo.create(
            "casa-verde", "Casa Verde", allowed_origins=["https://casaverde.com"]
        )
        assert await repo.resolve_origin("https://attacker.example") is None

    async def test_origins_do_not_leak_between_properties(
        self, repo: PropertyRepository
    ) -> None:
        await repo.create("a", "A", allowed_origins=["https://a.com"])
        await repo.create("b", "B", allowed_origins=["https://b.com"])
        assert await repo.resolve_origin("https://a.com") == "a"
        assert await repo.resolve_origin("https://b.com") == "b"

    async def test_updating_origins_removes_the_old_ones(
        self, repo: PropertyRepository
    ) -> None:
        """A client changing domain must not leave the old one authorised."""
        await repo.create("a", "A", allowed_origins=["https://old.com"])
        await repo.set_origins("a", ["https://new.com"])
        assert await repo.resolve_origin("https://old.com") is None
        assert await repo.resolve_origin("https://new.com") == "a"

    async def test_multiple_origins_for_one_property(
        self, repo: PropertyRepository
    ) -> None:
        await repo.create(
            "a", "A", allowed_origins=["https://a.com", "https://www.a.com"]
        )
        assert await repo.resolve_origin("https://www.a.com") == "a"


class TestSpendLedger:
    async def test_spend_accumulates_within_the_day(self, repo: PropertyRepository) -> None:
        await repo.create("a", "A", allowed_origins=["https://a.com"])
        await repo.record_spend("a", 0.01)
        await repo.record_spend("a", 0.02)
        assert await repo.spent_today("a") == pytest.approx(0.03)

    async def test_budget_check_flips_at_the_cap(self, repo: PropertyRepository) -> None:
        prop = await repo.create(
            "a", "A", allowed_origins=["https://a.com"], daily_spend_cap_usd=0.05
        )
        within, _ = await repo.within_budget(prop)
        assert within

        await repo.record_spend("a", 0.05)
        within, spent = await repo.within_budget(prop)
        assert not within
        assert spent == pytest.approx(0.05)

    async def test_spend_is_isolated_per_property(self, repo: PropertyRepository) -> None:
        """One client's abuse must not exhaust another client's budget."""
        await repo.create("a", "A", allowed_origins=["https://a.com"])
        await repo.create("b", "B", allowed_origins=["https://b.com"])
        await repo.record_spend("a", 5.0)
        assert await repo.spent_today("b") == 0.0


class TestContactRoute:
    def test_describes_available_channels(self) -> None:
        route = ContactRoute(phone="+44 1234 567890", email="stay@casaverde.com")
        described = route.describe()
        assert "+44 1234 567890" in described
        assert "stay@casaverde.com" in described

    def test_falls_back_when_unconfigured(self) -> None:
        assert ContactRoute().describe() == "contact the property directly"

    def test_survives_a_json_round_trip(self) -> None:
        route = ContactRoute(phone="+1 555", url="https://book.example")
        restored = ContactRoute.from_json(route.to_json())
        assert restored.phone == "+1 555"
        assert restored.url == "https://book.example"


class TestAccountUsageCap:
    """The ceiling under every model call.

    The per-property cap above bills a Property. Ingestion, evals and CLI runs
    bill none, so this is the only thing between a testing session and an
    invoice - it has to hold across restarts and across callers.
    """

    @pytest_asyncio.fixture
    async def db(self, tmp_path):
        database = Database(str(tmp_path / "usage.db"))
        await database.connect()
        yield database
        await database.close()

    async def test_blocks_once_the_spend_cap_is_reached(self) -> None:
        limiter = UsageLimiter(spend_cap_usd=0.10, call_cap=0)
        await limiter.check("answer")  # nothing spent yet

        await limiter.record(0.09)
        await limiter.check("answer")  # still inside the cap

        await limiter.record(0.02)
        with pytest.raises(UsageExceeded) as exc:
            await limiter.check("answer")
        assert exc.value.limit == "spend"

    async def test_blocks_a_loop_of_cheap_calls(self) -> None:
        """A dollar cap alone cannot stop a fast loop: cost is only known after
        the call, so the call ceiling is what catches it."""
        limiter = UsageLimiter(spend_cap_usd=100.0, call_cap=3)
        for _ in range(3):
            await limiter.check("rewrite")
            await limiter.record(0.0001)

        with pytest.raises(UsageExceeded) as exc:
            await limiter.check("rewrite")
        assert exc.value.limit == "call"

    async def test_zero_means_unlimited(self) -> None:
        limiter = UsageLimiter(spend_cap_usd=0, call_cap=0)
        await limiter.record(1000.0, calls=10_000)
        await limiter.check("eval")

    async def test_counters_survive_a_restart(self, db) -> None:
        """Restarting the process must not hand it a fresh budget - otherwise
        the cap is one Ctrl-C away from meaningless."""
        first = UsageLimiter(spend_cap_usd=0.10, call_cap=0, db=db)
        await first.check("answer")
        await first.record(0.15)

        second = UsageLimiter(spend_cap_usd=0.10, call_cap=0, db=db)
        with pytest.raises(UsageExceeded):
            await second.check("answer")

    async def test_accounting_failure_never_blocks_the_caller(self, tmp_path) -> None:
        """A broken ledger must not take the pipeline down with it."""
        closed = Database(str(tmp_path / "closed.db"))
        await closed.connect()
        await closed.close()

        limiter = UsageLimiter(spend_cap_usd=1.0, call_cap=0, db=closed)
        await limiter.record(0.01)          # write fails, in-memory total stands
        await limiter.check("answer")
        assert limiter.snapshot()["spent_usd"] == pytest.approx(0.01)


class TestGatewayPreflight:
    """A misconfigured reranker degrades quietly - reranking catches provider
    failures by design - so the configuration is checked once at boot instead."""

    def test_refuses_a_reranker_it_has_no_key_for(self) -> None:
        settings = Settings(
            openai_api_key="sk-test",
            cohere_api_key=None,
            rerank_model="rerank-v3.5",
        )
        with pytest.raises(RuntimeError, match="COHERE_API_KEY"):
            LLMGateway(settings).preflight()

    def test_a_chat_reranker_needs_no_cohere_key(self) -> None:
        settings = Settings(
            openai_api_key="sk-test", cohere_api_key=None, rerank_model="gpt-4o-mini"
        )
        LLMGateway(settings).preflight()

    def test_reports_every_missing_key_at_once(self) -> None:
        """One boot, one list - not a key at a time across three restarts."""
        settings = Settings(
            openai_api_key=None, cohere_api_key=None, rerank_model="rerank-v3.5"
        )
        with pytest.raises(RuntimeError) as exc:
            LLMGateway(settings).preflight()
        assert "OPENAI_API_KEY" in str(exc.value)
        assert "COHERE_API_KEY" in str(exc.value)


class TestRunSpendCap:
    """The per-process ceiling. Scoped to one run and never read back from the
    ledger, so a single experiment is bounded regardless of the day's total."""

    async def test_stops_a_run_at_its_own_ceiling(self) -> None:
        limiter = UsageLimiter(spend_cap_usd=0, call_cap=0, run_cap_usd=0.50)
        await limiter.record(0.49)
        await limiter.check("answer")

        await limiter.record(0.02)
        with pytest.raises(UsageExceeded) as exc:
            await limiter.check("answer")
        assert exc.value.limit == "run spend"

    async def test_a_fresh_run_starts_at_zero(self, tmp_path) -> None:
        """The daily ledger is shared; the run counter is not. A new process
        must get its own allowance, or the cap would be indistinguishable from
        the daily one."""
        database = Database(str(tmp_path / "run.db"))
        await database.connect()
        try:
            first = UsageLimiter(
                spend_cap_usd=0, call_cap=0, run_cap_usd=0.50, db=database
            )
            await first.record(0.60)
            with pytest.raises(UsageExceeded):
                await first.check("answer")

            second = UsageLimiter(
                spend_cap_usd=0, call_cap=0, run_cap_usd=0.50, db=database
            )
            await second.check("answer")
            assert second.snapshot()["run_spent_usd"] == 0.0
            # ...while the day's total carried over from the first run.
            assert second.snapshot()["spent_usd"] == pytest.approx(0.60)
        finally:
            await database.close()

    async def test_the_tightest_ceiling_wins(self) -> None:
        limiter = UsageLimiter(spend_cap_usd=10.0, call_cap=0, run_cap_usd=0.05)
        await limiter.record(0.06)
        with pytest.raises(UsageExceeded) as exc:
            await limiter.check("rerank")
        assert exc.value.limit == "run spend"


class TestEmbeddedVectorStore:
    """QDRANT_URL doubles as the backend switch, so the parse has to be exact:
    reading a directory path as a URL silently produces a client that connects
    to nothing and fails on first use."""

    def test_http_urls_mean_a_server(self) -> None:
        # Key set explicitly rather than inherited from whatever .env holds:
        # this assertion is about the URL parse, nothing else.
        store = VectorStore(
            Settings(qdrant_url="https://cluster.example:6333", qdrant_api_key="k")
        )
        assert not store.embedded

    def test_anything_else_means_embedded(self, tmp_path) -> None:
        assert VectorStore(Settings(qdrant_url=":memory:")).embedded
        assert VectorStore(Settings(qdrant_url=str(tmp_path / "q"))).embedded

    async def test_hybrid_search_works_without_a_server(self, tmp_path) -> None:
        """The whole point: the real query path - two named vectors fused with
        RRF, filtered to one tenant - runs in-process."""
        store = VectorStore(Settings(qdrant_url=str(tmp_path / "qdrant")))
        await store.ensure_collection(4)

        chunk = Chunk(
            chunk_id="c1",
            doc_id="d1",
            property_id="demo",
            text="Dogs are welcome in The Barn.",
            uri="upload://house-rules.pdf",
            title="House rules",
            heading_path=["Pets"],
            category=DocCategory.POLICIES,
            source_kind=SourceKind.UPLOAD,
            position=0,
            token_estimate=10,
            fetched_at="2026-09-05",
        )
        await store.upsert([chunk], [[1.0, 0.0, 0.0, 0.0]], [([1, 2], [0.7, 0.3])])

        hits = await store.hybrid_search(
            "demo",
            dense_vector=[1.0, 0.0, 0.0, 0.0],
            sparse_vector=([1, 2], [0.7, 0.3]),
            limit=5,
        )
        assert [h.chunk.text for h in hits] == ["Dogs are welcome in The Barn."]

        # The tenant filter is not a server-side nicety - it must hold here too.
        assert await store.count("someone-else") == 0
        await store.close()

    def test_ingestion_does_not_need_a_reranker_key(self) -> None:
        """`guide upload` never reranks. Blocking an index build on a key it
        will not use would stop an operator loading a corpus before the answer
        path is configured at all."""
        settings = Settings(
            openai_api_key="sk-test", cohere_api_key=None, rerank_model="rerank-v3.5"
        )
        LLMGateway(settings).preflight(needs_rerank=False)
