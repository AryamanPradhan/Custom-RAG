"""Layer 04 - reranking.

Hybrid search optimises for recall: it returns 40 candidates so the right one
is somewhere in the pile. Handing 40 chunks to the answer model is a bad idea
on two counts - it costs a lot of input tokens, and small models get worse, not
better, as irrelevant context grows around the relevant part.

So a second model reads the candidates against the question and scores them.
Listwise rather than pointwise: judging a chunk against its competitors is both
cheaper (one call, not forty) and more accurate than scoring each in isolation.

The score threshold does double duty - it trims the context, and when nothing
clears it that *is* the signal to deflect rather than answer.
"""

from __future__ import annotations

import json

from app.gateway.llm_gateway import LLMGateway, Task
from app.logging_setup import get_logger
from app.models.domain import ScoredChunk
from app.observability.metrics import METRICS
from app.observability.tracing import span

log = get_logger(__name__)

RERANK_SYSTEM = """\
You score how well each numbered passage answers a visitor's question about a \
hotel or homestay.

Score 0-10:
  9-10  directly and completely answers the question
  6-8   contains part of the answer, or answers it for one unit/room only
  3-5   same topic, does not actually answer it
  0-2   unrelated

Judge only whether the passage answers THIS question. Do not reward passages \
for being well written, promotional, or about the property in general.

Score every passage you are given, once each. Return JSON only."""

_SCHEMA = {
    "name": "rerank_scores",
    "schema": {
        "type": "object",
        "properties": {
            "scores": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "score": {"type": "integer"},
                    },
                    "required": ["index", "score"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["scores"],
        "additionalProperties": False,
    },
}

# Candidates are truncated before they reach the reranker: it only needs enough
# to judge relevance, and the full text is restored for whatever survives.
_PREVIEW_CHARS = 500


async def rerank(
    question: str,
    candidates: list[ScoredChunk],
    gateway: LLMGateway,
    *,
    top_n: int = 8,
    min_score: int = 4,
    property_id: str | None = None,
) -> list[ScoredChunk]:
    """Score candidates and return the best ones above the threshold.

    On reranker failure the retrieval order is kept and the top_n taken. That
    degrades answer quality but keeps the Guide up - and the groundedness check
    still stands behind it, so a bad context set produces a Deflection rather
    than a wrong answer.
    """
    if not candidates:
        return []

    listing = "\n\n".join(
        f"[{i}] {' > '.join(p for p in [c.chunk.title, *c.chunk.heading_path] if p)}\n"
        f"{c.chunk.text[:_PREVIEW_CHARS]}"
        for i, c in enumerate(candidates)
    )

    with span("rerank", candidates=len(candidates)) as s:
        try:
            result = await gateway.complete(
                Task.RERANK,
                system=RERANK_SYSTEM,
                messages=[
                    {
                        "role": "user",
                        "content": f"QUESTION: {question}\n\nPASSAGES:\n{listing}",
                    }
                ],
                max_tokens=1200,
                json_schema=_SCHEMA,
                property_id=property_id,
            )
            scores = {
                int(item["index"]): int(item["score"])
                for item in json.loads(result.text).get("scores", [])
                if 0 <= int(item["index"]) < len(candidates)
            }
        except Exception as exc:  # noqa: BLE001
            METRICS.incr("rerank.failed")
            log.warning("rerank.failed", error=f"{type(exc).__name__}: {exc}")
            s.attributes["fallback"] = True
            return candidates[:top_n]

        for i, candidate in enumerate(candidates):
            candidate.rerank_score = float(scores.get(i, 0))

        kept = [c for c in candidates if (c.rerank_score or 0) >= min_score]
        kept.sort(key=lambda c: c.rerank_score or 0, reverse=True)
        kept = kept[:top_n]

        s.attributes["kept"] = len(kept)
        s.attributes["best_score"] = kept[0].rerank_score if kept else 0

    METRICS.observe("rerank.kept", len(kept))
    if not kept:
        # Not an error: retrieval found pages, none of them answer the
        # question. This is what turns into a Deflection upstream.
        METRICS.incr("rerank.nothing_relevant")
        log.info("rerank.nothing_relevant", question=question[:120])
    return kept
