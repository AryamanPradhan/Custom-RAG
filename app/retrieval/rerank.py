"""Layer 04 - reranking.

Hybrid search optimises for recall: it returns 40 candidates so the right one
is somewhere in the pile. Handing 40 chunks to the answer model is a bad idea
on two counts - it costs a lot of input tokens, and small models get worse, not
better, as irrelevant context grows around the relevant part.

So a second model reads the candidates against the question and scores them.
Two implementations, chosen by RERANK_MODEL:

  dedicated  Cohere rerank-v3.5. A relevance model: no prompt, one call, ~100ms,
             billed per search unit rather than per token. The default.
  listwise   A chat model scoring all candidates in one call against a 0-10
             rubric. The fallback when no reranker is configured, and the only
             one that can explain itself.

They return different units - 0-1 relevance against a 0-10 rubric - so each
carries its own threshold and neither is read in the other's units.

The threshold does double duty either way: it trims the context, and when
nothing clears it that *is* the signal to deflect rather than answer.
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

def _passage(candidate: ScoredChunk) -> str:
    """What a reranker actually reads: the heading trail, then the whole Chunk.

    The trail leads, because "Rooms > The Barn > Pets" carries as much relevance
    signal as the prose under it.

    The body was once cut to its first 500 characters, on the theory that
    relevance is decided in the opening lines. It is not. The Chunk answering
    "do you have a helipad?" first says so at character 1117 of 1387: dense and
    sparse search, which index the whole text, ranked it first of forty, and
    the reranker then scored a version of it that never mentions a helipad and
    dropped it below the floor. Every answer living in the second half of a
    page was reachable by retrieval and unreachable past reranking.

    Nothing needs truncating here. `chunk_target_tokens` is 450, so a Chunk is
    already the size this stage wants, and Cohere bills one search unit per 100
    documents of up to 4096 tokens - an order of magnitude above what a Chunk
    can be.
    """
    trail = " > ".join(
        p for p in [candidate.chunk.title, *candidate.chunk.heading_path] if p
    )
    return f"{trail}\n{candidate.chunk.text}"


async def _score_with_reranker(
    question: str,
    candidates: list[ScoredChunk],
    gateway: LLMGateway,
    *,
    top_n: int,
    property_id: str | None,
) -> dict[int, float]:
    """Cohere. Relevance 0-1, best first, already trimmed to what we asked for."""
    scored = await gateway.rerank(
        question=question,
        documents=[_passage(c) for c in candidates],
        # Ask for more than we keep. The threshold decides what survives, not
        # the reranker's cutoff, and a tie at the boundary should be ours to
        # break.
        top_n=min(len(candidates), top_n * 2),
        property_id=property_id,
    )
    return {i: score for i, score in scored if 0 <= i < len(candidates)}


async def _score_with_chat_model(
    question: str,
    candidates: list[ScoredChunk],
    gateway: LLMGateway,
    *,
    property_id: str | None,
) -> dict[int, float]:
    """Listwise LLM. Scores on the 0-10 rubric in RERANK_SYSTEM."""
    listing = "\n\n".join(f"[{i}] {_passage(c)}" for i, c in enumerate(candidates))
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
    return {
        int(item["index"]): float(item["score"])
        for item in json.loads(result.text).get("scores", [])
        if 0 <= int(item["index"]) < len(candidates)
    }


async def rerank(
    question: str,
    candidates: list[ScoredChunk],
    gateway: LLMGateway,
    *,
    top_n: int = 8,
    min_score: int = 4,
    min_relevance: float = 0.2,
    property_id: str | None = None,
) -> list[ScoredChunk]:
    """Score candidates and return the best ones above the threshold.

    `min_score` applies to the listwise rubric and `min_relevance` to a
    dedicated reranker's 0-1 output. Which one is read depends on RERANK_MODEL;
    the other is ignored rather than converted, because the scales do not map
    onto each other and a bad conversion deflects everything or nothing.

    On reranker failure the retrieval order is kept and the top_n taken. That
    degrades answer quality but keeps the Guide up - and the groundedness check
    still stands behind it, so a bad context set produces a Deflection rather
    than a wrong answer.
    """
    if not candidates:
        return []

    dedicated = gateway.uses_dedicated_reranker
    threshold = min_relevance if dedicated else float(min_score)

    with span("rerank", candidates=len(candidates), dedicated=dedicated) as s:
        try:
            if dedicated:
                scores = await _score_with_reranker(
                    question, candidates, gateway, top_n=top_n, property_id=property_id
                )
            else:
                scores = await _score_with_chat_model(
                    question, candidates, gateway, property_id=property_id
                )
        except Exception as exc:  # noqa: BLE001
            METRICS.incr("rerank.failed")
            log.warning(
                "Reranker call failed; falling back to vector order.",
                error=f"{type(exc).__name__}: {exc}",
            )
            s.attributes["fallback"] = True
            s.summary = f"Reranker unavailable; kept the top {top_n} by vector score."
            return candidates[:top_n]

        # A candidate the reranker left out scores 0: for Cohere that means it
        # fell outside the requested window, which is a judgement that it is
        # irrelevant rather than missing data.
        for i, candidate in enumerate(candidates):
            candidate.rerank_score = float(scores.get(i, 0))

        kept = [c for c in candidates if (c.rerank_score or 0) >= threshold]
        kept.sort(key=lambda c: c.rerank_score or 0, reverse=True)
        kept = kept[:top_n]

        s.attributes["kept"] = len(kept)
        s.attributes["best_score"] = kept[0].rerank_score if kept else 0
        s.summary = (
            f"Reranked {len(candidates)} candidates down to {len(kept)} "
            f"document{'' if len(kept) == 1 else 's'}."
            if kept
            else f"No candidate cleared the relevance threshold of {threshold}."
        )

    METRICS.observe("rerank.kept", len(kept))
    if not kept:
        # Not an error: retrieval found pages, none of them answer the
        # question. This is what turns into a Deflection upstream.
        METRICS.incr("rerank.nothing_relevant")
        log.info(
            "Nothing in the corpus is relevant enough to answer this.",
            question=question[:120],
        )
    return kept
