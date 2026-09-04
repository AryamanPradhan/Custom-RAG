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
        )
        # Wired to the SQLite ledger by the API layer. Left unset in tests and
        # in the ingestion CLI, where there is no Property to bill.
        self._on_spend = on_spend

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
        provider = self._registry.for_model(model_id)

        started = time.perf_counter()
        try:
            result = await provider.complete(
                model=model_id,
                messages=messages,
                system=system,
                max_tokens=max_tokens,
                json_schema=json_schema,
            )
        except Exception as exc:
            METRICS.incr("llm.error", task=str(task), model=model_id)
            log.error(
                "llm.call_failed",
                task=str(task),
                model=model_id,
                error=f"{type(exc).__name__}: {exc}",
                trace_id=current_trace_id(),
            )
            raise

        await self._audit(task, result, started, property_id)
        return result

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
                log.error("llm.audit_failed", error=f"{type(exc).__name__}: {exc}")

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
        if result.refused:
            METRICS.incr("llm.refusal", task=str(task))
            log.warning("llm.refused", **entry)
        elif result.truncated:
            METRICS.incr("llm.truncated", task=str(task))
            log.warning("llm.truncated", **entry)
        else:
            log.info("llm.call", **entry)

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
