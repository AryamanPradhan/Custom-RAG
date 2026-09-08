"""Layer 04 - planning, hybrid retrieval, reranking and the store.

Retrieval is where a Deflection is usually decided, and almost always for a
mechanical reason rather than a model one: a query the planner never wrote, a
chunk that fusion ranked below the cut, a threshold read in the wrong units.
So these tests pin the mechanics - what is searched for, how results are fused,
what survives the relevance floor - with fake providers, and leave whether a
given answer was any good to the eval harness.

The embedded Qdrant tests are the same query path the server runs, in-process:
one collection, two named vectors, RRF fusion, tenant filter.
"""

from __future__ import annotations

import json

import pytest

from app.config import Settings
from app.models.domain import Chunk, DocCategory, ScoredChunk, SourceKind
from app.retrieval.embeddings import GeminiEmbedder
from app.retrieval.rerank import rerank
from app.retrieval.retriever import QueryPlan, Retriever, plan_query
from app.retrieval.store import VectorStore, point_id


class FakeResult:
    def __init__(self, text: str) -> None:
        self.text = text
        self.model = "fake"
        self.refused = False
        self.stop_reason = "stop"
        self.input_tokens = self.output_tokens = 0
        self.cache_read_tokens = self.cache_write_tokens = 0
        self.request_id = None

    @property
    def cost_usd(self) -> float:
        return 0.0

    @property
    def truncated(self) -> bool:
        return False


class FakeGateway:
    """A gateway that answers without a network.

    `relevance` switches it to a dedicated reranker - the production default -
    and `rubric_scores` to the listwise chat-model path.
    """

    def __init__(
        self,
        *,
        completion: str = "{}",
        fail: Exception | None = None,
        relevance: list[float] | None = None,
        rubric_scores: list[int] | None = None,
    ) -> None:
        self._completion = completion
        self._fail = fail
        self._relevance = relevance
        self._rubric = rubric_scores
        self.completions: list[dict] = []
        self.reranks: list[dict] = []

    @property
    def uses_dedicated_reranker(self) -> bool:
        return self._relevance is not None

    async def complete(self, task, **kwargs) -> FakeResult:
        self.completions.append({"task": str(task), **kwargs})
        if self._fail:
            raise self._fail
        if self._rubric is not None:
            return FakeResult(
                json.dumps(
                    {"scores": [{"index": i, "score": s} for i, s in enumerate(self._rubric)]}
                )
            )
        return FakeResult(self._completion)

    async def rerank(self, *, question, documents, top_n, property_id=None):
        self.reranks.append({"question": question, "documents": documents, "top_n": top_n})
        if self._fail:
            raise self._fail
        pairs = [(i, self._relevance[i]) for i in range(min(len(documents), len(self._relevance)))]
        pairs.sort(key=lambda pair: pair[1], reverse=True)
        return pairs[:top_n]


def _chunk(text: str, chunk_id: str = "c1", **overrides) -> Chunk:
    fields = {
        "chunk_id": chunk_id,
        "doc_id": "d1",
        "property_id": "casa-verde",
        "text": text,
        "uri": "upload://house-rules.pdf",
        "title": "House rules",
        "heading_path": ["Pets"],
        "category": DocCategory.POLICIES,
        "source_kind": SourceKind.UPLOAD,
        "position": 0,
        "token_estimate": len(text) // 4,
        "fetched_at": "2026-08-12",
    }
    return Chunk(**{**fields, **overrides})


def _scored(text: str, chunk_id: str = "c1", score: float = 0.5, **overrides) -> ScoredChunk:
    return ScoredChunk(chunk=_chunk(text, chunk_id, **overrides), score=score)


class TestQueryPlan:
    def test_the_rerank_query_is_the_rewrite_not_the_typo(self) -> None:
        """Reranking scores every candidate against one string, so noise in it
        moves every score at once. "Is tehr free wifi" plans identically to the
        clean question and retrieves the same chunks - then scores them a
        quarter as high and falls through the floor."""
        plan = QueryPlan(queries=["free wifi availability", "internet access"])
        assert plan.rerank_query() == "free wifi availability internet access"

    def test_a_planner_outage_degrades_rather_than_deflects(self) -> None:
        fallback = QueryPlan.fallback("can I bring my dog")
        assert fallback.queries == ["can I bring my dog"]
        assert not fallback.unit_dependent


class TestPlanning:
    async def test_a_conversational_question_becomes_standalone_queries(self) -> None:
        gateway = FakeGateway(
            completion=json.dumps(
                {"queries": ["pet policy for the Loft", "dogs allowed"], "unit_dependent": True}
            )
        )
        plan = await plan_query("what about the Loft?", [], gateway, property_id="casa-verde")

        assert plan.queries == ["pet policy for the Loft", "dogs allowed"]
        assert plan.unit_dependent

    async def test_only_the_recent_turns_are_shown_to_the_planner(self) -> None:
        """History is client-supplied and capped at 20 turns. The planner needs
        the last exchange to resolve "what about it", not the whole session."""
        gateway = FakeGateway(completion=json.dumps({"queries": ["q"], "unit_dependent": False}))
        history = [{"role": "user", "content": f"turn {i}"} for i in range(10)]

        await plan_query("and parking?", history, gateway)

        sent = gateway.completions[0]["messages"][0]["content"]
        assert "turn 9" in sent
        assert "turn 3" not in sent

    async def test_more_than_three_queries_are_trimmed(self) -> None:
        """Each query is a hybrid search and an embedding call, so the ceiling
        is a cost control as much as a quality one."""
        gateway = FakeGateway(
            completion=json.dumps(
                {"queries": ["a", "b", "c", "d", "e"], "unit_dependent": False}
            )
        )
        plan = await plan_query("q", [], gateway)
        assert plan.queries == ["a", "b", "c"]

    async def test_blank_queries_are_dropped(self) -> None:
        gateway = FakeGateway(
            completion=json.dumps({"queries": ["", "   ", "pet policy"], "unit_dependent": False})
        )
        plan = await plan_query("q", [], gateway)
        assert plan.queries == ["pet policy"]

    @pytest.mark.parametrize(
        ("label", "gateway"),
        [
            ("provider down", FakeGateway(fail=RuntimeError("503"))),
            ("unparseable output", FakeGateway(completion="not json at all")),
            ("empty plan", FakeGateway(completion=json.dumps({"queries": []}))),
        ],
    )
    async def test_the_raw_question_is_searched_when_planning_fails(
        self, label: str, gateway: FakeGateway
    ) -> None:
        """Retrieval on the raw question is worse than on a rewrite, but it is
        not nothing - a planner outage should degrade the Guide, not take it
        down."""
        plan = await plan_query("can I bring my dog", [], gateway)
        assert plan.queries == ["can I bring my dog"]


class FakeStore:
    """Returns a canned result list per query and records what was asked."""

    def __init__(self, results: dict[str, list[ScoredChunk]], fail_on: str | None = None):
        self._results = results
        self._fail_on = fail_on
        self.searched: list[str] = []

    async def hybrid_search(self, property_id, *, dense_vector, sparse_vector, limit):
        query = dense_vector[0]  # the fake embedder passes the query through
        self.searched.append(query)
        if query == self._fail_on:
            raise RuntimeError("qdrant unreachable")
        return list(self._results.get(query, []))


class FakeDense:
    async def embed_query(self, text: str):
        return [text]


class FakeSparse:
    async def encode_query(self, text: str):
        return ([0], [1.0])


class TestRetriever:
    async def test_agreement_between_queries_outranks_a_single_strong_hit(self) -> None:
        """Reciprocal rank fusion, and the reason for it: the same chunk found
        by two phrasings is better evidence than one chunk found once. Score is
        replaced by the RRF contribution, because a cosine similarity and a
        BM25 score are not on the same scale."""
        both = _scored("Dogs are welcome.", "both")
        store = FakeStore(
            {
                "pet policy": [both, _scored("Breakfast is at 8.", "only-a")],
                "dogs allowed": [_scored("Parking is free.", "only-b"), both],
            }
        )
        retriever = Retriever(store, FakeDense(), FakeSparse(), top_k=10)

        fused = await retriever.search("casa-verde", ["pet policy", "dogs allowed"])

        assert fused[0].chunk.chunk_id == "both"
        assert fused[0].score == pytest.approx(1 / 60 + 1 / 61)

    async def test_one_failed_query_does_not_sink_the_rest(self) -> None:
        store = FakeStore(
            {"good": [_scored("Dogs are welcome.", "a")]},
            fail_on="broken",
        )
        retriever = Retriever(store, FakeDense(), FakeSparse(), top_k=10)

        fused = await retriever.search("casa-verde", ["broken", "good"])
        assert [c.chunk.chunk_id for c in fused] == ["a"]

    async def test_the_candidate_set_is_capped(self) -> None:
        store = FakeStore(
            {"q": [_scored(f"chunk {i}", f"c{i}") for i in range(10)]},
        )
        retriever = Retriever(store, FakeDense(), FakeSparse(), top_k=3)
        assert len(await retriever.search("casa-verde", ["q"])) == 3

    async def test_no_queries_means_no_search(self) -> None:
        store = FakeStore({})
        retriever = Retriever(store, FakeDense(), FakeSparse())
        assert await retriever.search("casa-verde", []) == []
        assert store.searched == []

    async def test_the_visitors_own_wording_is_searched_too(self) -> None:
        """A rewrite is a paraphrase, and a paraphrase can lose the one word
        the Source actually uses. The raw question rides along on every
        question rather than being held back for a retry."""
        store = FakeStore({})
        retriever = Retriever(store, FakeDense(), FakeSparse())

        await retriever.retrieve(
            "casa-verde", QueryPlan(queries=["pet policy dogs"]), "can I bring my dog"
        )
        assert store.searched == ["pet policy dogs", "can I bring my dog"]

    async def test_a_rewrite_identical_to_the_question_is_not_searched_twice(self) -> None:
        store = FakeStore({})
        retriever = Retriever(store, FakeDense(), FakeSparse())

        await retriever.retrieve("casa-verde", QueryPlan(queries=["parking"]), "parking")
        assert store.searched == ["parking"]


class TestReranking:
    async def test_the_relevance_floor_decides_what_reaches_the_answer(self) -> None:
        candidates = [
            _scored("Dogs are welcome in The Barn.", "relevant"),
            _scored("Breakfast is served until 10.", "off-topic"),
        ]
        gateway = FakeGateway(relevance=[0.8, 0.05])

        kept = await rerank("pet policy", candidates, gateway, top_n=8, min_relevance=0.2)

        assert [c.chunk.chunk_id for c in kept] == ["relevant"]
        assert kept[0].rerank_score == pytest.approx(0.8)

    async def test_nothing_relevant_returns_nothing(self) -> None:
        """Not an error: retrieval found pages, none of them answer the
        question. This is what becomes a Deflection upstream."""
        gateway = FakeGateway(relevance=[0.05, 0.01])
        kept = await rerank(
            "do you have a helipad?",
            [_scored("Breakfast is at 8.", "a"), _scored("Parking is free.", "b")],
            gateway,
            min_relevance=0.2,
        )
        assert kept == []

    async def test_a_candidate_left_out_of_the_window_scores_zero(self) -> None:
        """Cohere returns only the window asked for. A candidate outside it is
        a judgement that it is irrelevant, not missing data."""
        candidates = [_scored(f"chunk {i}", f"c{i}") for i in range(4)]
        gateway = FakeGateway(relevance=[0.9, 0.8])  # only two come back

        kept = await rerank("q", candidates, gateway, top_n=4, min_relevance=0.2)

        assert [c.chunk.chunk_id for c in kept] == ["c0", "c1"]
        assert candidates[3].rerank_score == 0

    async def test_the_kept_set_is_capped_and_ordered(self) -> None:
        candidates = [_scored(f"chunk {i}", f"c{i}") for i in range(4)]
        gateway = FakeGateway(relevance=[0.4, 0.9, 0.5, 0.7])

        kept = await rerank("q", candidates, gateway, top_n=2, min_relevance=0.2)
        assert [c.chunk.chunk_id for c in kept] == ["c1", "c3"]

    async def test_the_two_rerankers_are_never_read_in_each_others_units(self) -> None:
        """0-1 relevance against a 0-10 rubric. Reading one as the other
        deflects everything or nothing, so each threshold applies only to the
        path it belongs to."""
        low_relevance = FakeGateway(relevance=[0.3])
        kept = await rerank(
            "q", [_scored("Dogs are welcome.", "a")], low_relevance, min_score=4, min_relevance=0.2
        )
        assert [c.chunk.chunk_id for c in kept] == ["a"]

        low_rubric = FakeGateway(rubric_scores=[3])
        kept = await rerank(
            "q", [_scored("Dogs are welcome.", "a")], low_rubric, min_score=4, min_relevance=0.2
        )
        assert kept == []

    async def test_the_listwise_path_reads_the_rubric(self) -> None:
        candidates = [_scored("Dogs are welcome.", "a"), _scored("Breakfast at 8.", "b")]
        gateway = FakeGateway(rubric_scores=[9, 2])

        kept = await rerank("pet policy", candidates, gateway, min_score=4)

        assert [c.chunk.chunk_id for c in kept] == ["a"]
        assert gateway.completions[0]["task"] == "rerank"

    async def test_a_reranker_outage_keeps_the_guide_up(self) -> None:
        """Retrieval order is worse than reranked order, but the groundedness
        check still stands behind it - so a bad context set produces a
        Deflection, not a wrong answer."""
        candidates = [_scored(f"chunk {i}", f"c{i}") for i in range(5)]
        gateway = FakeGateway(relevance=[0.9], fail=RuntimeError("cohere 500"))

        kept = await rerank("q", candidates, gateway, top_n=3)
        assert [c.chunk.chunk_id for c in kept] == ["c0", "c1", "c2"]

    async def test_no_candidates_costs_nothing(self) -> None:
        gateway = FakeGateway(relevance=[1.0])
        assert await rerank("q", [], gateway) == []
        assert gateway.reranks == []

    async def test_the_whole_chunk_is_scored_not_its_opening(self) -> None:
        """The regression that made every answer in the second half of a page
        unreachable: the passage was cut to 500 characters, so the chunk that
        first says "helipad" at character 1117 was scored on text that never
        mentions one, and dropped below the floor after retrieval had ranked it
        first of forty."""
        buried = "The estate has extensive grounds. " * 33 + "There is a helipad by the lake."
        gateway = FakeGateway(relevance=[0.9])

        await rerank("do you have a helipad?", [_scored(buried, "a")], gateway)

        sent = gateway.reranks[0]["documents"][0]
        assert "helipad by the lake" in sent

    async def test_the_heading_trail_leads_the_passage(self) -> None:
        """"Rooms > The Barn > Pets" carries as much relevance signal as the
        prose under it."""
        gateway = FakeGateway(relevance=[0.9])
        await rerank(
            "pet policy",
            [_scored("Dogs are welcome.", "a", heading_path=["The Barn", "Pets"])],
            gateway,
        )
        assert gateway.reranks[0]["documents"][0].startswith("House rules > The Barn > Pets\n")

    async def test_more_is_asked_for_than_is_kept(self) -> None:
        """The threshold decides what survives, not the reranker's cutoff - a
        tie at the boundary should be ours to break."""
        gateway = FakeGateway(relevance=[0.9] * 10)
        await rerank("q", [_scored(f"c{i}", f"c{i}") for i in range(10)], gateway, top_n=3)
        assert gateway.reranks[0]["top_n"] == 6


class TestVectorStore:
    """The embedded backend, which runs the real query path with no server."""

    @pytest.fixture
    async def store(self, tmp_path):
        store = VectorStore(Settings(qdrant_url=str(tmp_path / "qdrant")))
        await store.ensure_collection(4)
        yield store
        await store.close()

    async def _index(self, store: VectorStore, chunk: Chunk) -> None:
        await store.upsert([chunk], [[1.0, 0.0, 0.0, 0.0]], [([1, 2], [0.7, 0.3])])

    def test_point_ids_are_deterministic(self) -> None:
        """Qdrant wants a UUID and our chunk ids are hex strings. A uuid5 keeps
        a re-upload an update rather than a duplicate."""
        assert point_id("abc123") == point_id("abc123")
        assert point_id("abc123") != point_id("abc124")

    async def test_reindexing_the_same_chunk_replaces_it(self, store: VectorStore) -> None:
        await self._index(store, _chunk("Dogs are welcome.", "c1"))
        await self._index(store, _chunk("Dogs are welcome, on a lead.", "c1"))

        assert await store.count("casa-verde") == 1
        hits = await store.hybrid_search(
            "casa-verde", dense_vector=[1.0, 0.0, 0.0, 0.0], sparse_vector=([1, 2], [0.7, 0.3])
        )
        assert hits[0].chunk.text == "Dogs are welcome, on a lead."

    async def test_a_citation_survives_the_round_trip(self, store: VectorStore) -> None:
        """Heading path, unit and ingest date are what a Citation is made of,
        so they have to come back off a point, not just go onto one."""
        await self._index(
            store,
            _chunk(
                "Dogs are welcome in The Barn.",
                "c1",
                heading_path=["Pets", "Dogs"],
                unit="The Barn",
                metadata={"content_hash": "deadbeef"},
            ),
        )

        hit = (
            await store.hybrid_search(
                "casa-verde",
                dense_vector=[1.0, 0.0, 0.0, 0.0],
                sparse_vector=([1, 2], [0.7, 0.3]),
            )
        )[0]

        assert hit.chunk.heading_path == ["Pets", "Dogs"]
        assert hit.chunk.unit == "The Barn"
        assert hit.chunk.fetched_at == "2026-08-12"
        assert hit.chunk.category is DocCategory.POLICIES
        assert hit.chunk.metadata["content_hash"] == "deadbeef"

    async def test_search_cannot_reach_another_property(self, store: VectorStore) -> None:
        """Tenant isolation is enforced in the store rather than by the caller,
        so there is no code path that can forget it."""
        await self._index(store, _chunk("Dogs are welcome.", "c1", property_id="casa-verde"))
        await self._index(store, _chunk("No pets at all.", "c2", property_id="glass-hotel"))

        hits = await store.hybrid_search(
            "casa-verde", dense_vector=[1.0, 0.0, 0.0, 0.0], sparse_vector=([1, 2], [0.7, 0.3])
        )
        assert [h.chunk.chunk_id for h in hits] == ["c1"]

    async def test_replacing_a_document_removes_its_old_chunks(self, store: VectorStore) -> None:
        """A re-upload that produces fewer chunks than last time must not leave
        the surplus behind, still citable and now wrong."""
        await self._index(store, _chunk("Old policy.", "c1", doc_id="doc-a"))
        await self._index(store, _chunk("Old policy, page 2.", "c2", doc_id="doc-a"))
        await self._index(store, _chunk("Rates.", "c3", doc_id="doc-b"))

        await store.delete_document("casa-verde", "doc-a")

        assert await store.count("casa-verde") == 1

    async def test_erasure_drops_one_tenant_and_only_that_tenant(
        self, store: VectorStore
    ) -> None:
        await self._index(store, _chunk("Dogs are welcome.", "c1", property_id="casa-verde"))
        await self._index(store, _chunk("No pets.", "c2", property_id="glass-hotel"))

        await store.delete_property("casa-verde")

        assert await store.count("casa-verde") == 0
        assert await store.count("glass-hotel") == 1

    async def test_doc_hashes_report_what_is_indexed(self, store: VectorStore) -> None:
        """What lets an upload skip a file that has not changed."""
        await self._index(
            store, _chunk("Dogs.", "c1", doc_id="doc-a", metadata={"content_hash": "aaa"})
        )
        await self._index(
            store, _chunk("Rates.", "c2", doc_id="doc-b", metadata={"content_hash": "bbb"})
        )

        assert await store.doc_hashes("casa-verde") == {"doc-a": "aaa", "doc-b": "bbb"}
        assert await store.doc_hashes("glass-hotel") == {}

    async def test_upserting_nothing_is_not_an_error(self, store: VectorStore) -> None:
        """An upload of a file that chunked to nothing must not 500."""
        assert await store.upsert([], [], []) == 0


class TestDenseEmbedder:
    def test_a_missing_key_is_refused_at_construction(self) -> None:
        """Without it the dense half of retrieval cannot run, and lexical-only
        search would quietly become the product."""
        with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
            GeminiEmbedder(Settings(google_api_key=None))

    def _embedder(self, vectors_per_call, *, dim: int = 4) -> GeminiEmbedder:
        from types import SimpleNamespace

        embedder = GeminiEmbedder(
            Settings(google_api_key="g-test", dense_model="gemini-embedding-2", dense_dim=dim)
        )
        calls: list[int] = []

        async def embed_content(*, model, contents, config):
            calls.append(len(contents))
            count = vectors_per_call(len(contents))
            return SimpleNamespace(
                embeddings=[SimpleNamespace(values=[1.0] * dim) for _ in range(count)]
            )

        embedder._client = SimpleNamespace(  # type: ignore[assignment]
            aio=SimpleNamespace(models=SimpleNamespace(embed_content=embed_content))
        )
        embedder.calls = calls  # type: ignore[attr-defined]
        return embedder

    async def test_documents_are_embedded_in_batches(self) -> None:
        """The endpoint caps both request count and total tokens per call, and
        a chunk is ~450 tokens."""
        embedder = self._embedder(lambda n: n)
        vectors = await embedder.embed_documents([f"chunk {i}" for i in range(70)])

        assert len(vectors) == 70
        assert embedder.calls == [32, 32, 6]  # type: ignore[attr-defined]

    async def test_a_short_batch_refuses_to_misalign_chunks_and_vectors(self) -> None:
        """google-genai coerces a bare list[str] into one Content and returns a
        single vector for the whole batch. Accepting that would attach every
        chunk's text to its neighbour's embedding - silent corruption of the
        index, visible only as bad retrieval months later."""
        embedder = self._embedder(lambda n: 1)
        with pytest.raises(RuntimeError, match="refusing to misalign"):
            await embedder.embed_documents(["a", "b", "c"])
