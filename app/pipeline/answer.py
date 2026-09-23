"""The answer pipeline.

A fixed sequence, not an agent loop. With no tools to call and no actions to
take, there is nothing for a planner to orchestrate - so the pipeline is:

    screen -> plan -> retrieve -> rerank -> answer -> verify -> cite

Every stage can fail into a Deflection, which is a designed outcome rather than
an error. Flat latency, one traceable path, and an eval harness that can score
each stage independently.

The sequence is written once and rendered twice. What a turn decided is an
`Outcome` - transport-neutral, markers intact, no trace id - and the two public
entry points are adapters over it: `answer` renders one as an `AnswerResult`
for the JSON endpoint and the evals, `stream` renders one as widget events.
Only generation itself is implemented per transport, because a single call and
a token stream are genuinely different calls. Everything that decides whether
an answer may be shown at all lives in `_finish`, so a stage added to the
sequence reaches both surfaces or neither.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from app.gateway.llm_gateway import LLMGateway, Task
from app.guardrails.input_guard import (
    GuardOutcome,
    check_history,
    check_message,
    sanitise_context,
)
from app.guardrails.output_guard import scrub_answer, verify_grounding
from app.guardrails.patterns import redact_pii
from app.logging_setup import get_logger
from app.models.domain import Citation, ScoredChunk, TurnRecord
from app.observability.metrics import METRICS
from app.observability.tracing import current_trace, current_trace_id, span
from app.pipeline.citations import MarkerFilter, build_citations, strip_markers
from app.pipeline.intent import Intent, classify_intent
from app.pipeline.prompts import (
    build_answer_system,
    build_deflection,
    build_smalltalk,
    build_smalltalk_system,
    build_user_turn,
)
from app.retrieval.rerank import rerank
from app.retrieval.retriever import Retriever, plan_query
from app.storage.properties import Property

log = get_logger(__name__)

MAX_HISTORY_FOR_MODEL = 6


# Called with a finished turn, after the Visitor already has their answer.
OnTurn = Callable[[TurnRecord], Awaitable[None]]


@dataclass(slots=True)
class Outcome:
    """What the Guide decided this turn, before anything has rendered it.

    The answer keeps its [n] markers. Taking them off is a rendering choice -
    the Visitor does not want them, `build_citations` and the chat log do - so
    it belongs in the adapters and not here.
    """

    answer: str
    citations: list[Citation] = field(default_factory=list)
    deflected: bool = False
    grounded: bool = True
    reason: str = ""
    stages: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class _TurnDraft:
    """What the chat log needs that an Outcome does not carry.

    The screened question above all: the record must hold the text the model
    saw, not the raw one, or a redacted card number lands in SQLite anyway.
    """

    question: str
    intent: str = "informational"
    blocked: bool = False
    # Set once the turn is decided. `stream` is a generator and cannot return
    # a value, so this is how the decision reaches the wrapper that files it -
    # and None is meaningful: it is a Visitor who closed the tab before the
    # pipeline had decided anything.
    outcome: Outcome | None = None


@dataclass(slots=True)
class AnswerResult:
    """An Outcome as the JSON endpoint and the eval harness read it."""

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
        on_turn: OnTurn | None = None,
    ) -> None:
        self._gateway = gateway
        self._retriever = retriever
        # Absent by default, so an eval sweep does not file 22 visitor
        # conversations that no visitor had.
        self._on_turn = on_turn
        self._top_n = rerank_top_n
        # Both thresholds are carried, and the reranking stage reads whichever
        # matches the configured reranker's units.
        self._min_score = min_rerank_score
        self._min_relevance = min_rerank_relevance

    # -- deciding ---------------------------------------------------------

    def _decline(
        self,
        prop: Property,
        *,
        reason: str,
        stage: str,
        metric: str | None = "pipeline.deflected",
        grounded: bool = True,
        stages: dict[str, Any] | None = None,
    ) -> Outcome:
        """The Guide declining, in one place.

        Every Deflection is the same four things - the text, the flag, the
        reason and the counter - and they used to be written out at each site,
        which is how a stage came to be counted on one transport and not on the
        other. Passing through here is what keeps the metric and the reason
        describing the same event.
        """
        if metric:
            METRICS.incr(metric, stage=stage)
        return Outcome(
            answer=build_deflection(prop.display_name, prop.contact_route.describe()),
            deflected=True,
            grounded=grounded,
            reason=reason,
            stages=stages or {},
        )

    async def _prepare(
        self, prop: Property, question: str, history: list[dict], draft: _TurnDraft
    ) -> tuple[str, list[ScoredChunk], str, list[dict], Outcome | None]:
        """Screen, plan, retrieve, rerank.

        Returns (clean_question, chunks, context, clean_history, early).
        A non-None early means the turn is already decided - blocked,
        deflected, or answered without a model - and the answer model should
        not be called at all. The history comes back redacted, so callers must
        use the returned copy rather than the one they passed in.
        """
        message_verdict = check_message(question)
        if message_verdict.outcome is GuardOutcome.REJECT:
            # The guard rejected before it ever looked for PII, so this is the
            # one path where the log has to do its own redaction.
            draft.question = redact_pii(question)
            draft.blocked = True
            log.warning("Question rejected by the input guard.", reason=message_verdict.reason)
            return (
                question,
                [],
                "",
                [],
                self._decline(
                    prop,
                    reason=message_verdict.reason,
                    stage="message",
                    metric="pipeline.blocked",
                ),
            )

        history_verdict, history = check_history(history)
        if history_verdict.outcome is GuardOutcome.REJECT:
            draft.question = message_verdict.message or redact_pii(question)
            draft.blocked = True
            log.warning("Conversation history rejected by the input guard.",
                        reason=history_verdict.reason)
            return (
                question,
                [],
                "",
                [],
                self._decline(
                    prop,
                    reason=history_verdict.reason,
                    stage="history",
                    metric="pipeline.blocked",
                ),
            )

        clean = message_verdict.message or question
        draft.question = clean

        # Before the planner, because a greeting should cost nothing: no
        # rewrite call, no embedding, no vector search, no rerank.
        with span("intent", question=clean[:120]) as s:
            intent = classify_intent(clean)
            s.attributes["intent"] = intent.value
            s.summary = f"Read as {intent.value}."
        draft.intent = intent.value
        if intent is not Intent.INFORMATIONAL:
            METRICS.incr("pipeline.smalltalk", intent=intent.value)
            return (
                clean,
                [],
                "",
                history,
                # Not a Deflection: nothing was asked and nothing failed. It
                # cites nothing because it claims nothing.
                Outcome(
                    answer=await self._smalltalk(prop, intent, clean),
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
            return (
                clean,
                [],
                "",
                history,
                self._decline(
                    prop,
                    reason="nothing in the corpus answers this",
                    stage="retrieval",
                ),
            )

        context = sanitise_context(chunks)
        return clean, chunks, context, history, None

    async def _finish(
        self,
        prop: Property,
        *,
        question: str,
        text: str,
        chunks: list[ScoredChunk],
        context: str,
        refused: bool = False,
    ) -> Outcome:
        """Everything between a generated answer and a servable one.

        Four policies decide whether a Visitor may see what the model wrote: it
        declined, it cited nothing, it was not grounded, and whatever scrubbing
        left behind. Each used to be written once per transport, which is how
        the streaming path came to count one stage differently from the JSON
        path and to carry groundedness by a different route. They run here so a
        rule added to the sequence arrives on both surfaces at once.
        """
        if refused or not text.strip():
            return self._decline(
                prop,
                reason="answer model declined or returned nothing",
                stage="generation",
            )

        # An answer that cites nothing claims nothing: the system prompt is
        # explicit that a claim about the property must carry its [n]. So an
        # uncited answer is a non-answer however warmly it is phrased, and it
        # becomes a Deflection without a verifier call - which is both cheaper
        # and safer than showing unverified prose, since the one thing that
        # could be hiding in it is a claim the model chose not to cite.
        if not build_citations(chunks, text):
            return self._decline(
                prop, reason="the answer cited nothing", stage="uncited"
            )

        verdict = await verify_grounding(
            question=question,
            answer=text,
            context=context,
            gateway=self._gateway,
            property_id=prop.property_id,
        )
        if not verdict.grounded:
            return self._decline(
                prop,
                reason=verdict.detail or "answer was not supported by the sources",
                stage="grounding",
                grounded=False,
                stages={"unsupported_claims": verdict.unsupported_claims},
            )

        contact = prop.contact_route.describe()
        scrubbed = scrub_answer(text, trusted_text=context + " " + contact)
        return Outcome(
            answer=scrubbed,
            citations=build_citations(chunks, scrubbed),
            stages={"chunks": len(chunks)},
        )

    async def _smalltalk(self, prop: Property, intent: Intent, message: str) -> str:
        """The reply to a turn that asked nothing about the property.

        Generated, so that "hi" and "thanks, that's exactly what I needed" do
        not get the same sentence back. It is one call to the cheap model with
        no retrieval, no rerank and no verifier, because there is nothing to
        retrieve for a greeting and nothing to check it against.

        That last part is why the prompt is written as a list of things not to
        say: this is the one reply in the pipeline that reaches a Visitor
        unverified, so it must not make a claim about the Property at all.
        Scrubbing catches the contact details it was told not to invent - the
        one kind of hallucination here that a Visitor would act on.

        The fixed template is the fallback. A provider outage should cost a
        Visitor some warmth, not their reply.
        """
        fallback = build_smalltalk(intent.value, prop.display_name)
        # CAPABILITY describes how the Guide behaves rather than greeting
        # anyone. That is a fact about this system, not something to improvise.
        if intent is Intent.CAPABILITY:
            return fallback

        try:
            with span("smalltalk", intent=intent.value) as s:
                result = await self._gateway.complete(
                    Task.ANSWER,
                    system=build_smalltalk_system(prop.display_name),
                    messages=[{"role": "user", "content": message}],
                    max_tokens=120,
                    property_id=prop.property_id,
                )
                text = scrub_answer(
                    result.text.strip(), trusted_text=prop.contact_route.describe()
                )
                s.summary = (
                    f"Wrote a {intent.value} reply."
                    if text and not result.refused
                    else f"Fell back to the fixed {intent.value} reply."
                )
        except Exception as exc:  # noqa: BLE001 - provider-specific failures
            METRICS.incr("pipeline.smalltalk_failed", intent=intent.value)
            log.warning(
                "Smalltalk generation failed; using the fixed reply.",
                error=f"{type(exc).__name__}: {exc}",
            )
            return fallback

        if result.refused or not text:
            METRICS.incr("pipeline.smalltalk_failed", intent=intent.value)
            return fallback
        return text

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

    # -- filing -----------------------------------------------------------

    async def _file(
        self,
        prop: Property,
        draft: _TurnDraft,
        *,
        mode: str,
        answer: str,
        deflected: bool,
        grounded: bool,
        reason: str,
        citations: list[dict],
        latency_ms: float,
    ) -> None:
        """Hand the finished turn to the chat log.

        Never raises. The visitor has their answer by the time this runs, and
        a full disk is not a reason to turn a served answer into a 500.
        """
        if self._on_turn is None:
            return
        trace = current_trace()
        try:
            await self._on_turn(
                TurnRecord(
                    property_id=prop.property_id,
                    session_id=trace.session_id if trace else "",
                    trace_id=current_trace_id(),
                    question=draft.question,
                    answer=answer,
                    mode=mode,
                    intent=draft.intent,
                    deflected=deflected,
                    grounded=grounded,
                    blocked=draft.blocked,
                    reason=reason,
                    citations=citations,
                    latency_ms=latency_ms,
                )
            )
        except Exception as exc:  # noqa: BLE001
            METRICS.incr("chat_log.write_failed")
            log.error(
                "The turn could not be written to the chat log.",
                error=f"{type(exc).__name__}: {exc}",
            )

    async def _record(
        self,
        prop: Property,
        draft: _TurnDraft,
        outcome: Outcome,
        *,
        mode: str,
        started: float,
    ) -> None:
        """File a decided turn, markers and all.

        The log takes the Outcome rather than what was rendered: an operator
        checking whether an answer was legitimate needs to see which claim came
        from which source, which is exactly what the screen no longer shows.
        """
        await self._file(
            prop,
            draft,
            mode=mode,
            answer=outcome.answer,
            deflected=outcome.deflected,
            grounded=outcome.grounded,
            reason=outcome.reason,
            citations=[asdict(c) for c in outcome.citations],
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    # -- non-streaming (used by evals and the JSON endpoint) --------------

    async def answer(
        self, prop: Property, question: str, history: list[dict] | None = None
    ) -> AnswerResult:
        started = time.perf_counter()
        draft = _TurnDraft(question=question)
        outcome = await self._decide(prop, question, history or [], draft)
        await self._record(prop, draft, outcome, mode="json", started=started)
        return AnswerResult(
            answer=strip_markers(outcome.answer),
            citations=outcome.citations,
            deflected=outcome.deflected,
            grounded=outcome.grounded,
            reason=outcome.reason,
            trace_id=current_trace_id(),
            stages=outcome.stages,
        )

    async def _decide(
        self, prop: Property, question: str, history: list[dict], draft: _TurnDraft
    ) -> Outcome:
        """One model call, then the shared finish."""
        clean, chunks, context, history, early = await self._prepare(
            prop, question, history, draft
        )
        if early is not None:
            draft.outcome = early
            return early

        system = build_answer_system(prop.display_name, prop.contact_route.describe())

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

        outcome = await self._finish(
            prop,
            question=clean,
            text=result.text,
            chunks=chunks,
            context=context,
            refused=result.refused,
        )
        draft.outcome = outcome
        return outcome

    # -- streaming (the widget) ------------------------------------------

    async def stream(
        self, prop: Property, question: str, history: list[dict] | None = None
    ) -> AsyncIterator[dict]:
        """Stream the answer to the widget, then file the turn.

        What is filed is the decision, not the tokens: a `retract` means what
        streamed was replaced, so recording the tokens would file an answer no
        visitor was left holding. Filing happens in a `finally`, so a turn a
        visitor abandoned mid-stream is still recorded - and that is the one
        case with no decision to file, where what reached the screen is all
        there is.
        """
        started = time.perf_counter()
        draft = _TurnDraft(question=question)
        shown: list[str] = []

        try:
            async for event in self._stream(prop, question, history or [], draft):
                if event.get("type") == "token":
                    shown.append(event["text"])
                yield event
        finally:
            if draft.outcome is not None:
                await self._record(
                    prop, draft, draft.outcome, mode="stream", started=started
                )
            else:
                await self._file(
                    prop,
                    draft,
                    mode="stream",
                    answer="".join(shown).strip(),
                    deflected=False,
                    grounded=True,
                    reason="",
                    citations=[],
                    latency_ms=(time.perf_counter() - started) * 1000,
                )

    async def _stream(
        self, prop: Property, question: str, history: list[dict], draft: _TurnDraft
    ) -> AsyncIterator[dict]:
        """Render a turn as widget events.

        Tokens stream immediately; citations are withheld until the
        groundedness check clears. If it fails, a `retract` event tells the
        widget to discard what it has shown and replace it with the Deflection.
        That is the price of streaming a verified answer - the alternative is
        several seconds of spinner on every turn.
        """
        clean, chunks, context, history, early = await self._prepare(
            prop, question, history, draft
        )
        if early is not None:
            # Nothing has been shown yet, so this is an ordinary reply rather
            # than a retraction. The flag is what distinguishes "we could not
            # answer" from "nothing was asked", so it has to carry the
            # outcome's own value rather than True.
            yield {
                "type": "deflect",
                "answer": strip_markers(early.answer),
                "reason": early.reason,
            }
            yield self._done(early.deflected)
            return

        system = build_answer_system(prop.display_name, prop.contact_route.describe())
        # A marker can arrive split across tokens, so the filter holds back
        # anything that might still become one. `raw` stays unfiltered.
        raw: list[str] = []
        markers = MarkerFilter()

        try:
            async for token in self._gateway.stream_answer(
                system=system,
                messages=self._messages(clean, context, history),
                max_tokens=1200,
                property_id=prop.property_id,
            ):
                raw.append(token)
                visible = markers.feed(token)
                if visible:
                    yield {"type": "token", "text": visible}
            tail = markers.flush()
            if tail:
                yield {"type": "token", "text": tail}
        except Exception as exc:  # noqa: BLE001
            # Counted as a stream failure and not as a Deflection: the Corpus
            # was not the problem, and an operator reading the deflected turns
            # weekly should not find infrastructure in that list.
            METRICS.incr("pipeline.stream_failed")
            log.error("Answer generation failed mid-stream; retracting.",
                      error=f"{type(exc).__name__}: {exc}")
            failed = self._decline(
                prop, reason="generation failed", stage="generation", metric=None
            )
            draft.outcome = failed
            yield self._retract(failed)
            yield self._done(True)
            return

        streamed = "".join(raw).strip()
        outcome = await self._finish(
            prop, question=clean, text=streamed, chunks=chunks, context=context
        )
        draft.outcome = outcome

        if outcome.deflected:
            yield self._retract(outcome)
            yield self._done(True)
            return

        # Scrubbing may have changed what is already on screen.
        if outcome.answer != streamed:
            yield {"type": "replace", "answer": strip_markers(outcome.answer)}

        yield {"type": "citations", "citations": [asdict(c) for c in outcome.citations]}
        yield self._done(False)

    def _retract(self, outcome: Outcome) -> dict:
        """Take back what streamed.

        The Visitor has already read tokens, so a Deflection at this point has
        to say that they no longer stand.
        """
        return {
            "type": "retract",
            "answer": strip_markers(outcome.answer),
            "reason": outcome.reason,
        }

    def _done(self, deflected: bool) -> dict:
        return {"type": "done", "trace_id": current_trace_id(), "deflected": deflected}
