"""Layer 04 - query understanding and retrieval.

Two things stand between a Visitor's question and a good set of Chunks:

  1. The question is conversational and often elliptical. "What about the
     Loft?" carries all of its meaning in the previous turn. Embedding it
     verbatim retrieves nothing useful.
  2. One embedding of one phrasing is a narrow net. "Can I bring my dog"
     and "pet policy" land in different neighbourhoods.

So the planner rewrites the question into several standalone search queries
using the history, and every query is run through hybrid search before the
results are fused.

What the planner does not do is narrow the search. It once guessed a category
per question and the search was filtered to it, which is a guess made before
anything has been retrieved - and a wrong guess hides the one Source that
answers the question. See ADR 0005.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

from app.gateway.llm_gateway import LLMGateway, Task
from app.logging_setup import get_logger
from app.models.domain import ScoredChunk
from app.observability.metrics import METRICS
from app.observability.tracing import span
from app.retrieval.embeddings import GeminiEmbedder, SparseEncoder
from app.retrieval.store import VectorStore

log = get_logger(__name__)

PLANNER_SYSTEM = """\
You turn a website visitor's question about a hotel or homestay into search \
queries for that property's own content.

Rules:
- Resolve pronouns and ellipsis using the conversation. "What about the Loft?" \
after a question about pets becomes "pet policy for the Loft".
- Produce 1-3 standalone queries. Use more only when the question genuinely \
has separate parts ("do you have parking and is breakfast included").
- Write queries in the vocabulary a hotel website uses, not the visitor's. \
"can I bring my dog" -> "pet policy dogs allowed".
- Set unit_dependent to true when the answer could differ between separate \
rooms, cottages or apartments at the property (pets, capacity, price, view, \
bathroom) and the visitor has not said which one.

Return JSON only."""

_PLAN_SCHEMA = {
    "name": "query_plan",
    "schema": {
        "type": "object",
        "properties": {
            "queries": {
                "type": "array",
                "items": {"type": "string"},
                "description": "1-3 standalone search queries",
            },
            "unit_dependent": {"type": "boolean"},
        },
        "required": ["queries", "unit_dependent"],
        "additionalProperties": False,
    },
}


@dataclass(slots=True)
class QueryPlan:
    queries: list[str]
    unit_dependent: bool = False

    def rerank_query(self) -> str:
        """The information need as one string, for a stage that takes one query.

        The rewrites rather than the Visitor's raw wording, because the two
        stages want different things. Retrieval wants both: hybrid search
        fuses several queries, and the original phrasing sometimes carries a
        word the rewrite paraphrased away. Reranking takes a single query and
        scores every candidate against it, so noise in that one string moves
        every score at once.

        A typo is the clearest case. "Is tehr free wifi" and "Is there free
        wifi" plan identically - both rewrite to "free wifi availability" - and
        retrieve the same Chunks in the same order, but scored against the
        raw text the whole set drops roughly fourfold, from 0.2578 to 0.0626 at
        the top, and falls through the relevance floor. The Corpus had the
        answer, the planner had already repaired the question, and the Guide
        deflected anyway.
        """
        return " ".join(self.queries)

    @classmethod
    def fallback(cls, question: str) -> QueryPlan:
        """Used when the planner is unavailable. Retrieval on the raw question
        is worse, but it is not nothing - and a planner outage should degrade
        the Guide, not take it down."""
        return cls(queries=[question], unit_dependent=False)


async def plan_query(
    question: str,
    history: list[dict],
    gateway: LLMGateway,
    *,
    property_id: str | None = None,
) -> QueryPlan:
    recent = history[-6:]
    transcript = "\n".join(
        f"{t.get('role', '?')}: {t.get('content', '')}" for t in recent
    )
    with span("plan_query") as s:
        try:
            result = await gateway.complete(
                Task.REWRITE,
                system=PLANNER_SYSTEM,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"CONVERSATION SO FAR:\n{transcript or '(none)'}\n\n"
                            f"LATEST QUESTION: {question}"
                        ),
                    }
                ],
                max_tokens=500,
                json_schema=_PLAN_SCHEMA,
                property_id=property_id,
            )
            data = json.loads(result.text)
            queries = [q for q in data.get("queries", []) if q.strip()][:3]
            plan = QueryPlan(
                queries=queries or [question],
                unit_dependent=bool(data.get("unit_dependent", False)),
            )
        except Exception as exc:  # noqa: BLE001
            METRICS.incr("retrieval.planner_failed")
            log.warning("retrieval.planner_failed", error=f"{type(exc).__name__}: {exc}")
            plan = QueryPlan.fallback(question)

        s.attributes["queries"] = plan.queries
        s.attributes["unit_dependent"] = plan.unit_dependent
        s.summary = (
            f"Rewrote the question into {len(plan.queries)} search "
            f"{'query' if len(plan.queries) == 1 else 'queries'}."
        )
    return plan


class Retriever:
    def __init__(
        self,
        store: VectorStore,
        dense: GeminiEmbedder,
        sparse: SparseEncoder,
        *,
        top_k: int = 40,
    ) -> None:
        self._store = store
        self._dense = dense
        self._sparse = sparse
        self._top_k = top_k

    async def search(
        self,
        property_id: str,
        queries: list[str],
    ) -> list[ScoredChunk]:
        """Run every query through hybrid search and fuse the results."""
        if not queries:
            return []

        results = await asyncio.gather(
            *(self._one(property_id, q) for q in queries),
            return_exceptions=True,
        )

        fused: dict[str, ScoredChunk] = {}
        for outcome in results:
            if isinstance(outcome, BaseException):
                log.warning("retrieval.query_failed", error=str(outcome))
                continue
            for rank, scored in enumerate(outcome):
                key = scored.chunk.chunk_id
                # Reciprocal rank fusion across queries. The same chunk found
                # by two different phrasings is stronger evidence than one
                # chunk found once with a high score.
                contribution = 1.0 / (60 + rank)
                if key in fused:
                    fused[key].score += contribution
                else:
                    scored.score = contribution
                    fused[key] = scored

        ordered = sorted(fused.values(), key=lambda s: s.score, reverse=True)
        return ordered[: self._top_k]

    async def _one(self, property_id: str, query: str) -> list[ScoredChunk]:
        dense_vector, sparse_vector = await asyncio.gather(
            self._dense.embed_query(query),
            self._sparse.encode_query(query),
        )
        return await self._store.hybrid_search(
            property_id,
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            limit=self._top_k,
        )

    async def retrieve(
        self,
        property_id: str,
        plan: QueryPlan,
        raw_question: str,
    ) -> list[ScoredChunk]:
        """One pass over the whole Corpus.

        The Visitor's own wording rides along with the rewritten queries rather
        than being held back for a retry. A rewrite is a paraphrase, and a
        paraphrase can lose the one word the Source actually uses - so the
        original phrasing is worth a query on every question, not only on the
        ones that have already come back empty.
        """
        queries = list(dict.fromkeys([*plan.queries, raw_question]))
        with span("retrieve") as s:
            chunks = await self.search(property_id, queries)
            s.attributes["hits"] = len(chunks)
            s.attributes["queries"] = len(queries)
            s.summary = f"Retrieved {len(chunks)} candidates from Qdrant."
        return chunks
