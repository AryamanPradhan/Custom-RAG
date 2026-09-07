"""Layer 06 - LLM Gateway.

Every model call in the system goes through this class. Nothing else
constructs a provider client. That single chokepoint is what makes per-task
model routing, cross-vendor cost accounting, audit logging and the
per-property spend cap possible at all.

Routing is by *task*, not by caller: a caller asks for a rerank, the gateway
decides who serves it. Moving the answer path from GPT-4o mini to Claude, or
the reranker to a different model, is one line here rather than a grep across
the repo.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Awaitable, Callable
from enum import StrEnum

from app.config import Settings, get_settings
from app.gateway.budget import get_usage_limiter
from app.gateway.pricing import Provider, provider_for
from app.gateway.providers import LLMResult, ProviderRegistry
from app.logging_setup import get_logger
from app.observability.metrics import METRICS
from app.observability.tracing import current_trace_id

log = get_logger(__name__)

SpendRecorder = Callable[[str, float], Awaitable[None]]


class Task(StrEnum):
    """What the call is for. Determines which model serves it."""

    ANSWER = "answer"          # Visitor-facing prose. High volume, cheap model.
    REWRITE = "rewrite"        # Conversational question -> search queries.
    RERANK = "rerank"          # Listwise reranking of retrieved candidates.
    VERIFY = "verify"          # Groundedness check. Blocking - never skipped.
    EVAL = "eval"              # Offline scoring.


class LLMGateway:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        on_spend: SpendRecorder | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._registry = ProviderRegistry(
            openai_api_key=self.settings.openai_api_key,
            anthropic_api_key=self.settings.anthropic_api_key,
            cohere_api_key=self.settings.cohere_api_key,
            max_retries=self.settings.provider_max_retries,
        )
        # Wired to the SQLite ledger by the API layer. Left unset in tests and
        # in the ingestion CLI, where there is no Property to bill.
        self._on_spend = on_spend
        # The account-wide cap. Unlike the per-property one it applies to calls
        # that bill nobody - classification during ingestion, eval runs - which
        # is exactly where an unattended loop spends real money.
        self._usage = get_usage_limiter(self.settings)

    # -- routing ---------------------------------------------------------

    def model_for(self, task: Task) -> str:
        match task:
            case Task.ANSWER:
                return self.settings.answer_model
            case Task.REWRITE:
                return self.settings.rewrite_model
            case Task.RERANK:
                return self.settings.rerank_model
            case Task.VERIFY:
                return self.settings.verifier_model
            case Task.EVAL:
                return self.settings.eval_model

    def temperature_for(self, task: Task) -> float:
        """Sampling temperature by task.

        Only the answer is prose. Everything else - the rewrite, the
        classification, the groundedness verdict, the eval score - is a
        judgement that should not change between two identical calls, and at
        the provider default of 1.0 it does. The verifier is the one that
        matters: it blocks answers, so sampling it means blocking a different
        set of correct answers on every run.
        """
        if task is Task.ANSWER:
            return self.settings.answer_temperature
        return 0.0

    @property
    def uses_dedicated_reranker(self) -> bool:
        """Whether RERANK_MODEL is a relevance model rather than a chat model.

        The reranking stage asks this because the two paths differ in more than
        vendor: one takes a prompt and returns a 0-10 rubric score, the other
        takes documents and returns 0-1 relevance.
        """
        return provider_for(self.settings.rerank_model) is Provider.COHERE

    def preflight(self, *, needs_rerank: bool = True) -> None:
        """Refuse to start on a configuration that would degrade silently.

        Checked once at boot rather than per request. A missing Cohere key does
        not raise a 500 - reranking catches provider failures and falls back to
        raw retrieval order - so without this the Guide would come up, answer
        every question from unranked context, and say so only in a warning log.
        A stage that quietly stops working is worse than one that never starts.
        """
        missing: list[str] = []
        if not self.settings.openai_api_key:
            missing.append("OPENAI_API_KEY (every chat task)")
        # Ingestion never reranks, so a missing reranker key must not stop an
        # operator indexing documents before the answer path is configured.
        if needs_rerank and self.uses_dedicated_reranker and not self.settings.cohere_api_key:
            missing.append(
                f"COHERE_API_KEY (RERANK_MODEL={self.settings.rerank_model}; "
                f"point it at a chat model to use the listwise reranker instead)"
            )
        if missing:
            raise RuntimeError("Unusable configuration - missing: " + "; ".join(missing))

    # -- calls -----------------------------------------------------------

    async def complete(
        self,
        task: Task,
        *,
        messages: list[dict],
        system: str | None = None,
        max_tokens: int = 1024,
        json_schema: dict | None = None,
        property_id: str | None = None,
        model: str | None = None,
    ) -> LLMResult:
        model_id = model or self.model_for(task)
        temperature = self.temperature_for(task)
        await self._usage.check(str(task))
        provider = self._registry.for_model(model_id)

        started = time.perf_counter()
        try:
            result = await provider.complete(
                model=model_id,
                messages=messages,
                system=system,
                max_tokens=max_tokens,
                json_schema=json_schema,
                temperature=temperature,
            )
        except Exception as exc:
            METRICS.incr("llm.error", task=str(task), model=model_id)
            log.error(
                f"{model_id} failed on the {task} call.",
                task=str(task),
                model=model_id,
                error=f"{type(exc).__name__}: {exc}",
                trace_id=current_trace_id(),
            )
            raise

        await self._audit(task, result, started, property_id)
        return result

    async def rerank(
        self,
        *,
        question: str,
        documents: list[str],
        top_n: int,
        property_id: str | None = None,
    ) -> list[tuple[int, float]]:
        """Score passages with a dedicated reranker.

        Returns (index into `documents`, relevance 0-1), best first. Same
        chokepoint as every other model call: capped, audited and billed here,
        because a reranker that skipped the ledger would be spend the daily cap
        cannot see.
        """
        model_id = self.settings.rerank_model
        await self._usage.check(str(Task.RERANK))
        reranker = self._registry.reranker_for(model_id)

        started = time.perf_counter()
        try:
            outcome = await reranker.rerank(
                model=model_id, query=question, documents=documents, top_n=top_n
            )
        except Exception as exc:
            METRICS.incr("llm.error", task=str(Task.RERANK), model=model_id)
            log.error(
                f"{model_id} failed on the rerank call.",
                task=str(Task.RERANK),
                model=model_id,
                error=f"{type(exc).__name__}: {exc}",
                trace_id=current_trace_id(),
            )
            raise

        await self._audit(Task.RERANK, outcome.usage, started, property_id)
        return outcome.scores

    async def stream_answer(
        self,
        *,
        messages: list[dict],
        system: str | None = None,
        max_tokens: int = 2000,
        property_id: str | None = None,
    ) -> AsyncIterator[str]:
        """Token stream for the widget.

        Billing happens in a finally block. A Visitor closing the tab mid-answer
        makes the consumer stop iterating, which closes this generator - and
        those tokens were still spent. Auditing after the loop would let anyone
        stream answers for free by disconnecting, straight past the daily cap.
        """
        model_id = self.model_for(Task.ANSWER)
        await self._usage.check(str(Task.ANSWER))
        provider = self._registry.for_model(model_id)
        started = time.perf_counter()

        record = LLMResult(text="", model=model_id)
        chars = 0
        try:
            async for text in provider.stream(
                model=model_id,
                messages=messages,
                system=system,
                max_tokens=max_tokens,
                out=record,
                temperature=self.temperature_for(Task.ANSWER),
            ):
                chars += len(text)
                yield text
        finally:
            estimated = False
            if not record.input_tokens and not record.output_tokens:
                # Providers report usage in a final chunk an abandoned stream
                # never reaches. Estimate rather than bill zero.
                prompt_chars = sum(len(m.get("content", "")) for m in messages) + len(
                    system or ""
                )
                record.input_tokens = prompt_chars // 4
                record.output_tokens = chars // 4
                estimated = True
            try:
                await self._audit(
                    Task.ANSWER, record, started, property_id, estimated=estimated
                )
            except Exception as exc:  # noqa: BLE001 - must not mask the real error
                log.error(
                    "Could not record the cost of a streamed answer.",
                    error=f"{type(exc).__name__}: {exc}",
                )

    # -- budget ----------------------------------------------------------

    async def _audit(
        self,
        task: Task,
        result: LLMResult,
        started: float,
        property_id: str | None,
        *,
        estimated: bool = False,
    ) -> None:
        elapsed_ms = (time.perf_counter() - started) * 1000
        cost = result.cost_usd

        METRICS.observe("llm.latency_ms", elapsed_ms, task=str(task), model=result.model)
        METRICS.add("llm.cost_usd", cost, task=str(task), model=result.model)
        METRICS.add("llm.input_tokens", result.input_tokens, model=result.model)
        METRICS.add("llm.output_tokens", result.output_tokens, model=result.model)
        if result.cache_read_tokens:
            METRICS.add("llm.cache_read_tokens", result.cache_read_tokens, model=result.model)

        entry = {
            "task": str(task),
            "model": result.model,
            "trace_id": current_trace_id(),
            "property_id": property_id,
            "latency_ms": round(elapsed_ms, 1),
            "cost_usd": round(cost, 6),
            "stop_reason": result.stop_reason,
            "request_id": result.request_id,
            "estimated": estimated,
        }
        # One line per billable call: the money and the latency, in words,
        # because this is the line an operator reads when a bill surprises them.
        priced = f"{result.model} · {task} · {elapsed_ms:.0f}ms · ${cost:.4f}"
        if result.refused:
            METRICS.incr("llm.refusal", task=str(task))
            log.warning(f"{priced} · the model refused.", **entry)
        elif result.truncated:
            METRICS.incr("llm.truncated", task=str(task))
            log.warning(f"{priced} · output truncated at the token limit.", **entry)
        else:
            log.info(priced, **entry)

        # Every billable call lands here exactly once, including an abandoned
        # stream, so this is the only place the account counters need touching.
        await self._usage.record(cost)

        if self._on_spend is not None and property_id and cost > 0:
            await self._on_spend(property_id, cost)


_gateway: LLMGateway | None = None


def get_gateway() -> LLMGateway:
    global _gateway
    if _gateway is None:
        _gateway = LLMGateway()
    return _gateway


def set_gateway(gateway: LLMGateway) -> None:
    """Used by the app lifespan to install a gateway wired to the spend ledger."""
    global _gateway
    _gateway = gateway
