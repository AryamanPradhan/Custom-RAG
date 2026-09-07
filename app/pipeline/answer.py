"""The answer pipeline.

A fixed sequence, not an agent loop. With no tools to call and no actions to
take, there is nothing for a planner to orchestrate - so the pipeline is:

    screen -> plan -> retrieve -> rerank -> answer -> verify -> cite

Every stage can fail into a Deflection, which is a designed outcome rather than
an error. Flat latency, one traceable path, and an eval harness that can score
each stage independently.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from app.gateway.llm_gateway import LLMGateway, Task
from app.guardrails.input_guard import (
    GuardOutcome,
    check_history,
    check_message,
    sanitise_context,
)
from app.guardrails.output_guard import scrub_answer, verify_grounding
from app.logging_setup import get_logger
from app.models.domain import Citation, ScoredChunk
from app.observability.metrics import METRICS
from app.observability.tracing import current_trace_id, span
from app.pipeline.intent import Intent, classify_intent
from app.pipeline.prompts import (
    build_answer_system,
    build_deflection,
    build_smalltalk,
    build_user_turn,
)
from app.retrieval.rerank import rerank
from app.retrieval.retriever import Retriever, plan_query
from app.storage.properties import Property

log = get_logger(__name__)

MAX_HISTORY_FOR_MODEL = 6


@dataclass(slots=True)
class AnswerResult:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    deflected: bool = False
    grounded: bool = True
    reason: str = ""
    trace_id: str = ""
    stages: dict[str, Any] = field(default_factory=dict)


class AnswerPipeline:
    def __init__(
        self,
        *,
        gateway: LLMGateway,
        retriever: Retriever,
        rerank_top_n: int = 8,
        min_rerank_score: int = 4,
        min_rerank_relevance: float = 0.2,
    ) -> None:
        self._gateway = gateway
        self._retriever = retriever
        self._top_n = rerank_top_n
        # Both thresholds are carried, and the reranking stage reads whichever
        # matches the configured reranker's units.
        self._min_score = min_rerank_score
        self._min_relevance = min_rerank_relevance

    # -- shared stages ---------------------------------------------------

    async def _prepare(
        self, prop: Property, question: str, history: list[dict]
    ) -> tuple[str, list[ScoredChunk], str, list[dict], AnswerResult | None]:
        """Screen, plan, retrieve, rerank.

        Returns (clean_question, chunks, context, clean_history, early_result).
        A non-None early_result means the turn is already decided - blocked or
        deflected - and the answer model should not be called at all. The
        history comes back redacted, so callers must use the returned copy
        rather than the one they passed in.
        """
        contact = prop.contact_route.describe()

        message_verdict = check_message(question)
        if message_verdict.outcome is GuardOutcome.REJECT:
            METRICS.incr("pipeline.blocked", stage="message")
            log.warning("Question rejected by the input guard.", reason=message_verdict.reason)
            return (
                question,
                [],
                "",
                [],
                AnswerResult(
                    answer=build_deflection(prop.display_name, contact),
                    deflected=True,
                    reason=message_verdict.reason,
                    trace_id=current_trace_id(),
                ),
            )

        history_verdict, history = check_history(history)
        if history_verdict.outcome is GuardOutcome.REJECT:
            METRICS.incr("pipeline.blocked", stage="history")
            log.warning("Conversation history rejected by the input guard.",
                        reason=history_verdict.reason)
            return (
                question,
                [],
                "",
                [],
                AnswerResult(
                    answer=build_deflection(prop.display_name, contact),
                    deflected=True,
                    reason=history_verdict.reason,
                    trace_id=current_trace_id(),
                ),
            )

        clean = message_verdict.message or question

        # Before the planner, because a greeting should cost nothing: no
        # rewrite call, no embedding, no vector search, no rerank.
        with span("intent", question=clean[:120]) as s:
            intent = classify_intent(clean)
            s.attributes["intent"] = intent.value
            s.summary = f"Read as {intent.value}."
        if intent is not Intent.INFORMATIONAL:
            METRICS.incr("pipeline.smalltalk", intent=intent.value)
            return (
                clean,
                [],
                "",
                history,
                AnswerResult(
                    # Not a Deflection: nothing was asked and nothing failed.
                    # It cites nothing because it claims nothing.
                    answer=build_smalltalk(intent.value, prop.display_name),
                    deflected=False,
                    reason="",
                    trace_id=current_trace_id(),
                    stages={"intent": intent.value},
                ),
            )

        plan = await plan_query(
            clean, history, self._gateway, property_id=prop.property_id
        )
        chunks = await self._retriever.retrieve(prop.property_id, plan, clean)

        if chunks:
            # The planner's rewrites, not the raw question - see
            # QueryPlan.rerank_query. `clean` stays the question the answer
            # model is asked, which is the Visitor's own words.
            chunks = await self._rerank(
                plan.rerank_query() or clean, chunks, prop.property_id
            )

        # Nothing survives reranking only when the Corpus genuinely has no
        # answer: the search that produced these candidates saw every Chunk the
        # Property has. There is no second pass to fall back to, which is the
        # point - a Deflection here is a real gap rather than a bad guess.
        if not chunks:
            METRICS.incr("pipeline.deflected", stage="retrieval")
            return (
                clean,
                [],
                "",
                history,
                AnswerResult(
                    answer=build_deflection(prop.display_name, contact),
                    deflected=True,
                    reason="nothing in the corpus answers this",
                    trace_id=current_trace_id(),
                ),
            )

        context = sanitise_context(chunks)
        return clean, chunks, context, history, None

    async def _rerank(
        self, question: str, chunks: list[ScoredChunk], property_id: str
    ) -> list[ScoredChunk]:
        return await rerank(
            question,
            chunks,
            self._gateway,
            top_n=self._top_n,
            min_score=self._min_score,
            min_relevance=self._min_relevance,
            property_id=property_id,
        )

    def _messages(self, question: str, context: str, history: list[dict]) -> list[dict]:
        turns = [
            {"role": t["role"], "content": t["content"]}
            for t in history[-MAX_HISTORY_FOR_MODEL:]
            if t.get("role") in ("user", "assistant") and t.get("content")
        ]
        return [*turns, {"role": "user", "content": build_user_turn(question, context)}]

    # -- non-streaming (used by evals and the JSON endpoint) --------------

    async def answer(
        self, prop: Property, question: str, history: list[dict] | None = None
    ) -> AnswerResult:
        history = history or []
        clean, chunks, context, history, early = await self._prepare(
            prop, question, history
        )
        if early is not None:
            return early

        contact = prop.contact_route.describe()
        system = build_answer_system(prop.display_name, contact)

        with span("generate") as s:
            result = await self._gateway.complete(
                Task.ANSWER,
                system=system,
                messages=self._messages(clean, context, history),
                max_tokens=1200,
                property_id=prop.property_id,
            )
            s.attributes["output_tokens"] = result.output_tokens
            s.summary = (
                f"Wrote an answer from {len(chunks)} "
                f"source{'' if len(chunks) == 1 else 's'} "
                f"({result.output_tokens} tokens)."
                if not result.refused
                else "The answer model declined to respond."
            )

        if result.refused or not result.text.strip():
            METRICS.incr("pipeline.deflected", stage="generation")
            return AnswerResult(
                answer=build_deflection(prop.display_name, contact),
                deflected=True,
                reason="answer model declined or returned nothing",
                trace_id=current_trace_id(),
            )

        # An answer that cites nothing claims nothing: the system prompt is
        # explicit that a claim about the property must carry its [n]. So an
        # uncited answer is a non-answer however warmly it is phrased, and it
        # becomes a Deflection without a verifier call - which is both cheaper
        # and safer than showing unverified prose, since the one thing that
        # could be hiding in it is a claim the model chose not to cite.
        if not build_citations(chunks, result.text):
            METRICS.incr("pipeline.deflected", stage="uncited")
            return AnswerResult(
                answer=build_deflection(prop.display_name, contact),
                deflected=True,
                reason="the answer cited nothing",
                trace_id=current_trace_id(),
            )

        verdict = await verify_grounding(
            question=clean,
            answer=result.text,
            context=context,
            gateway=self._gateway,
            property_id=prop.property_id,
        )
        if not verdict.grounded:
            METRICS.incr("pipeline.deflected", stage="grounding")
            return AnswerResult(
                answer=build_deflection(prop.display_name, contact),
                deflected=True,
                grounded=False,
                reason=verdict.detail or "answer was not supported by the sources",
                trace_id=current_trace_id(),
                stages={"unsupported_claims": verdict.unsupported_claims},
            )

        text = scrub_answer(result.text, trusted_text=context + " " + contact)
        return AnswerResult(
            answer=text,
            citations=build_citations(chunks, text),
            trace_id=current_trace_id(),
            stages={"chunks": len(chunks)},
        )

    # -- streaming (the widget) ------------------------------------------

    async def stream(
        self, prop: Property, question: str, history: list[dict] | None = None
    ) -> AsyncIterator[dict]:
        """Yield widget events.

        Tokens stream immediately; citations are withheld until the
        groundedness check clears. If it fails, a `retract` event tells the
        widget to discard what it has shown and replace it with the Deflection.
        That is the price of streaming a verified answer - the alternative is
        several seconds of spinner on every turn.
        """
        history = history or []
        contact = prop.contact_route.describe()

        clean, chunks, context, history, early = await self._prepare(
            prop, question, history
        )
        if early is not None:
            # The widget renders this as an ordinary reply; the flag is what
            # distinguishes "we could not answer" from "nothing was asked",
            # so it has to carry the result's own value rather than True.
            yield {"type": "deflect", "answer": early.answer, "reason": early.reason}
            yield {
                "type": "done",
                "trace_id": current_trace_id(),
                "deflected": early.deflected,
            }
            return

        system = build_answer_system(prop.display_name, contact)
        buffer: list[str] = []

        try:
            async for token in self._gateway.stream_answer(
                system=system,
                messages=self._messages(clean, context, history),
                max_tokens=1200,
                property_id=prop.property_id,
            ):
                buffer.append(token)
                yield {"type": "token", "text": token}
        except Exception as exc:  # noqa: BLE001
            METRICS.incr("pipeline.stream_failed")
            log.error("Answer generation failed mid-stream; retracting.",
                      error=f"{type(exc).__name__}: {exc}")
            yield {
                "type": "retract",
                "answer": build_deflection(prop.display_name, contact),
                "reason": "generation failed",
            }
            yield {"type": "done", "trace_id": current_trace_id(), "deflected": True}
            return

        text = "".join(buffer).strip()
        if not text:
            yield {
                "type": "retract",
                "answer": build_deflection(prop.display_name, contact),
                "reason": "empty answer",
            }
            yield {"type": "done", "trace_id": current_trace_id(), "deflected": True}
            return

        # Same rule as the non-streaming path: uncited is a non-answer.
        if not build_citations(chunks, text):
            METRICS.incr("pipeline.deflected", stage="uncited")
            yield {
                "type": "retract",
                "answer": build_deflection(prop.display_name, contact),
                "reason": "the answer cited nothing",
            }
            yield {"type": "done", "trace_id": current_trace_id(), "deflected": True}
            return

        verdict = await verify_grounding(
            question=clean,
            answer=text,
            context=context,
            gateway=self._gateway,
            property_id=prop.property_id,
        )
        if not verdict.grounded:
            METRICS.incr("pipeline.deflected", stage="grounding")
            yield {
                "type": "retract",
                "answer": build_deflection(prop.display_name, contact),
                "reason": verdict.detail or "answer was not supported by the sources",
            }
            yield {"type": "done", "trace_id": current_trace_id(), "deflected": True}
            return

        scrubbed = scrub_answer(text, trusted_text=context + " " + contact)
        if scrubbed != text:
            yield {"type": "replace", "answer": scrubbed}

        yield {
            "type": "citations",
            "citations": [
                {
                    "index": c.index,
                    "uri": c.uri,
                    "label": c.label,
                    "snippet": c.snippet,
                    "published_on": c.published_on,
                    "unit": c.unit,
                }
                for c in build_citations(chunks, scrubbed)
            ],
        }
        yield {"type": "done", "trace_id": current_trace_id(), "deflected": False}


def build_citations(chunks: list[ScoredChunk], answer: str) -> list[Citation]:
    """Return only the sources the answer actually cited.

    Listing all eight retrieved chunks would imply the answer leaned on all of
    them. Citations are a checkable claim about provenance, so they track the
    [n] markers the model actually wrote.
    """
    cited: list[Citation] = []
    for i, scored in enumerate(chunks, start=1):
        if f"[{i}]" not in answer:
            continue
        chunk = scored.chunk
        snippet = chunk.text.strip().replace("\n", " ")
        cited.append(
            Citation(
                index=i,
                uri=chunk.uri,
                label=scored.citation_label,
                snippet=snippet[:220] + ("..." if len(snippet) > 220 else ""),
                published_on=chunk.fetched_at,
                unit=chunk.unit,
            )
        )
    return cited
