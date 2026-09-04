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
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field

from app.gateway.llm_gateway import LLMGateway, Task
from app.logging_setup import get_logger
from app.models.domain import DocCategory, ScoredChunk
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
            "categories": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [str(c) for c in DocCategory],
                },
                "description": "content categories to prefer, or empty for all",
            },
            "unit_dependent": {"type": "boolean"},
        },
        "required": ["queries", "categories", "unit_dependent"],
        "additionalProperties": False,
    },
}


@dataclass(slots=True)
class QueryPlan:
    queries: list[str]
    categories: list[DocCategory] = field(default_factory=list)
    unit_dependent: bool = False

    @classmethod
    def fallback(cls, question: str) -> QueryPlan:
        """Used when the planner is unavailable. Retrieval on the raw question
        is worse, but it is not nothing - and a planner outage should degrade
        the Guide, not take it down."""
        return cls(queries=[question], categories=[], unit_dependent=False)


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
                categories=[
                    DocCategory(c) for c in data.get("categories", []) if c in set(DocCategory)
                ],
                unit_dependent=bool(data.get("unit_dependent", False)),
            )
        except Exception as exc:  # noqa: BLE001
            METRICS.incr("retrieval.planner_failed")
            log.warning("retrieval.planner_failed", error=f"{type(exc).__name__}: {exc}")
            plan = QueryPlan.fallback(question)

        s.attributes["queries"] = plan.queries
        s.attributes["unit_dependent"] = plan.unit_dependent
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
        *,
        categories: list[DocCategory] | None = None,
    ) -> list[ScoredChunk]:
        """Run every query through hybrid search and fuse the results."""
        if not queries:
            return []

        results = await asyncio.gather(
            *(self._one(property_id, q, categories) for q in queries),
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

    async def _one(
        self,
        property_id: str,
        query: str,
        categories: list[DocCategory] | None,
    ) -> list[ScoredChunk]:
        dense_vector, sparse_vector = await asyncio.gather(
            self._dense.embed_query(query),
            self._sparse.encode_query(query),
        )
        return await self._store.hybrid_search(
            property_id,
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            limit=self._top_k,
            categories=categories,
        )

    async def retrieve_with_retry(
        self,
        property_id: str,
        plan: QueryPlan,
        raw_question: str,
    ) -> tuple[list[ScoredChunk], bool]:
        """The one conditional retry the pipeline allows.

        The first pass is narrow: rewritten queries, filtered to the categories
        the planner picked. When that comes back empty the usual cause is a bad
        category guess, so the retry drops the filter and adds the Visitor's
        own wording, which sometimes matches text the rewrite paraphrased away.

        Returns (chunks, retried).
        """
        with span("retrieve") as s:
            chunks = await self.search(property_id, plan.queries, categories=plan.categories)
            s.attributes["hits"] = len(chunks)
            s.attributes["filtered"] = bool(plan.categories)

        if chunks:
            return chunks, False

        METRICS.incr("retrieval.retry")
        with span("retrieve_retry") as s:
            widened = list(dict.fromkeys([*plan.queries, raw_question]))
            chunks = await self.search(property_id, widened, categories=None)
            s.attributes["hits"] = len(chunks)
        return chunks, True
