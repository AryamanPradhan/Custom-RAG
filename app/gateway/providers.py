"""Layer 06 - Provider adapters.

OpenAI serves every chat task, Cohere reranks and Google embeds. The Anthropic
adapter is kept
but unused by default: each SDK has a different request shape, a different
usage object and a different idea of where the system prompt goes, and this
module flattens all of that into one result type so the gateway above it - and
the spend cap that depends on it - never has to care who served a call. Keeping
the second adapter is what makes "route rerank back to Claude" an env change.

The neutral message shape is:

    system:   str | None
    messages: [{"role": "user" | "assistant", "content": str}, ...]

Anthropic takes system as a top-level parameter; OpenAI wants it prepended as a
message. The adapters handle that difference, callers do not.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.gateway.pricing import Provider, cost_usd, provider_for
from app.logging_setup import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class LLMResult:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    # Cohere's billing unit. Zero for everything that bills tokens.
    search_units: int = 0
    stop_reason: str | None = None
    refused: bool = False
    request_id: str | None = None
    raw: Any = field(default=None, repr=False)

    @property
    def cost_usd(self) -> float:
        return cost_usd(
            self.model,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cache_read_tokens=self.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens,
            search_units=self.search_units,
        )

    @property
    def truncated(self) -> bool:
        """Output hit the token ceiling mid-sentence."""
        return self.stop_reason in ("max_tokens", "length")


class ChatProvider(Protocol):
    async def complete(
        self,
        *,
        model: str,
        messages: list[dict],
        system: str | None = None,
        max_tokens: int = 1024,
        json_schema: dict | None = None,
        temperature: float | None = None,
    ) -> LLMResult: ...

    def stream(
        self,
        *,
        model: str,
        messages: list[dict],
        system: str | None = None,
        max_tokens: int = 1024,
        out: LLMResult,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        """Yield text deltas, filling `out` with real usage as it arrives.

        Usage is written into a caller-owned record rather than returned,
        because the caller must still be able to bill a stream that was
        abandoned half way through.
        """
        ...


# ---------------------------------------------------------------------------
# OpenAI - the answer path
# ---------------------------------------------------------------------------


class OpenAIProvider:
    def __init__(self, api_key: str | None, max_retries: int = 1) -> None:
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(
            api_key=api_key, max_retries=max_retries, timeout=90.0
        )

    @staticmethod
    def _payload(system: str | None, messages: list[dict]) -> list[dict]:
        return ([{"role": "system", "content": system}] if system else []) + messages

    @staticmethod
    def _usage(result: LLMResult, usage) -> None:
        if usage is None:
            return
        result.input_tokens = getattr(usage, "prompt_tokens", 0) or 0
        result.output_tokens = getattr(usage, "completion_tokens", 0) or 0
        # Prompt caching is automatic on OpenAI; cached tokens are reported as a
        # subset of prompt_tokens, so split them out to price them correctly.
        details = getattr(usage, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", 0) or 0
        if cached:
            result.cache_read_tokens = cached
            result.input_tokens = max(0, result.input_tokens - cached)

    async def complete(
        self,
        *,
        model: str,
        messages: list[dict],
        system: str | None = None,
        max_tokens: int = 1024,
        json_schema: dict | None = None,
        temperature: float | None = None,
    ) -> LLMResult:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": self._payload(system, messages),
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if json_schema:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": json_schema.get("name", "response"),
                    "strict": True,
                    "schema": json_schema["schema"],
                },
            }

        response = await self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        result = LLMResult(
            text=(choice.message.content or "").strip(),
            model=model,
            stop_reason=choice.finish_reason,
            request_id=getattr(response, "id", None),
            raw=response,
        )
        # A content filter stop is OpenAI's refusal signal.
        result.refused = choice.finish_reason == "content_filter" or bool(
            getattr(choice.message, "refusal", None)
        )
        self._usage(result, getattr(response, "usage", None))
        return result

    async def stream(
        self,
        *,
        model: str,
        messages: list[dict],
        system: str | None = None,
        max_tokens: int = 1024,
        out: LLMResult,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        # The overloads key off literal `stream` and a typed message union;
        # this adapter deliberately speaks the neutral dict shape defined at
        # the top of the module.
        stream = await self._client.chat.completions.create(  # type: ignore[call-overload]
            model=model,
            messages=self._payload(system, messages),
            max_tokens=max_tokens,
            # Streamed and non-streamed answers must read the same; a widget
            # answer and an eval answer are the same product.
            **({} if temperature is None else {"temperature": temperature}),
            stream=True,
            # Without this no chunk carries usage at all, and the answer would
            # cost nothing as far as the spend cap is concerned.
            stream_options={"include_usage": True},
        )
        async for chunk in stream:
            # The usage-bearing chunk arrives last and has an empty choices
            # list, so it has to be read before the choices guard below.
            if getattr(chunk, "usage", None):
                self._usage(out, chunk.usage)
            if chunk.choices:
                choice = chunk.choices[0]
                if choice.finish_reason:
                    out.stop_reason = choice.finish_reason
                delta = choice.delta
                if delta and delta.content:
                    yield delta.content


# ---------------------------------------------------------------------------
# Anthropic - not routed to by default; reachable by pointing a task's model
# env var at a claude-* id, which also needs the optional SDK installed:
#     pip install -e ".[anthropic]"
# ---------------------------------------------------------------------------


class AnthropicProvider:
    def __init__(self, api_key: str | None, max_retries: int = 1) -> None:
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:  # optional dependency
            raise RuntimeError(
                "A claude-* model is routed to but the anthropic SDK is not "
                "installed. Either point the task's *_MODEL env var at an "
                'OpenAI model, or install the extra: pip install -e ".[anthropic]"'
            ) from exc

        kwargs: dict[str, Any] = {"max_retries": max_retries, "timeout": 90.0}
        if api_key:
            kwargs["api_key"] = api_key
        # With no explicit key the SDK resolves ANTHROPIC_API_KEY /
        # ANTHROPIC_AUTH_TOKEN / an "ant auth login" profile itself.
        self._client = AsyncAnthropic(**kwargs)

    @staticmethod
    def _usage(result: LLMResult, usage) -> None:
        if usage is None:
            return
        result.input_tokens = getattr(usage, "input_tokens", 0) or 0
        result.output_tokens = getattr(usage, "output_tokens", 0) or 0
        result.cache_read_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0
        result.cache_write_tokens = getattr(usage, "cache_creation_input_tokens", 0) or 0

    async def complete(
        self,
        *,
        model: str,
        messages: list[dict],
        system: str | None = None,
        max_tokens: int = 1024,
        json_schema: dict | None = None,
        temperature: float | None = None,
    ) -> LLMResult:
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if system:
            # Cache the system prompt: it is byte-identical across every call
            # for a given task, and it carries the bulk of the instructions.
            kwargs["system"] = [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        if json_schema:
            kwargs["output_config"] = {
                "format": {"type": "json_schema", "schema": json_schema["schema"]}
            }

        response = await self._client.messages.create(**kwargs)
        result = LLMResult(
            text="".join(b.text for b in response.content if b.type == "text").strip(),
            model=model,
            stop_reason=response.stop_reason,
            refused=response.stop_reason == "refusal",
            request_id=getattr(response, "_request_id", None),
            raw=response,
        )
        if result.refused:
            details = getattr(response, "stop_details", None)
            log.warning(
                f"{model} refused to answer.",
                model=model,
                category=getattr(details, "category", None),
            )
        self._usage(result, getattr(response, "usage", None))
        return result

    async def stream(
        self,
        *,
        model: str,
        messages: list[dict],
        system: str | None = None,
        max_tokens: int = 1024,
        out: LLMResult,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if system:
            kwargs["system"] = system
        async with self._client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text
            final = await stream.get_final_message()
            out.stop_reason = final.stop_reason
            self._usage(out, getattr(final, "usage", None))


# ---------------------------------------------------------------------------
# Cohere - reranking
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RerankOutcome:
    """Scores plus the usage record that pays for them.

    Kept together because the gateway has to audit the call whatever the
    caller does with the scores - including when it decides to discard them.
    """

    scores: list[tuple[int, float]]  # (index into the documents passed in, 0-1)
    usage: LLMResult


class CohereReranker:
    """A relevance model, not a chat model.

    Nothing is prompted here: the query and the passages go over as data and
    come back scored, which is why it costs a fraction of a listwise LLM pass
    and returns in about a tenth of the time.
    """

    def __init__(self, api_key: str | None, max_retries: int = 1) -> None:
        if not api_key:
            raise RuntimeError(
                "COHERE_API_KEY is unset but RERANK_MODEL is a Cohere reranker. "
                "Set the key, or point RERANK_MODEL at a chat model to fall "
                "back to the listwise LLM reranker."
            )
        try:
            from cohere import AsyncClientV2
        except ImportError as exc:
            raise RuntimeError(
                'The cohere SDK is not installed: pip install -e "."'
            ) from exc

        self._client = AsyncClientV2(api_key=api_key, max_retries=max_retries)

    async def rerank(
        self, *, model: str, query: str, documents: list[str], top_n: int
    ) -> RerankOutcome:
        response = await self._client.rerank(
            model=model, query=query, documents=documents, top_n=top_n
        )
        # Bill what Cohere says it billed rather than assuming one unit: a
        # request over 100 documents, or over long ones, is charged as several.
        units = 1
        meta = getattr(response, "meta", None)
        billed = getattr(meta, "billed_units", None)
        if billed is not None and getattr(billed, "search_units", None):
            units = int(billed.search_units)

        return RerankOutcome(
            scores=[(r.index, float(r.relevance_score)) for r in response.results],
            usage=LLMResult(
                text="",
                model=model,
                search_units=units,
                request_id=getattr(response, "id", None),
                raw=response,
            ),
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class ProviderRegistry:
    """Lazily constructs one client per vendor.

    Lazy matters twice over: a deployment that never reaches the eval path
    should not need that vendor's key present at import time just to boot, and
    with everything routed to OpenAI the Anthropic client is never built at all
    - so its SDK never has to be installed.
    """

    def __init__(
        self,
        *,
        openai_api_key: str | None,
        anthropic_api_key: str | None = None,
        cohere_api_key: str | None = None,
        max_retries: int = 1,
    ) -> None:
        self._max_retries = max_retries
        self._keys = {
            Provider.OPENAI: openai_api_key,
            Provider.ANTHROPIC: anthropic_api_key,
            Provider.COHERE: cohere_api_key,
        }
        self._cache: dict[Provider, ChatProvider] = {}
        self._reranker: CohereReranker | None = None

    def for_model(self, model_id: str) -> ChatProvider:
        provider = provider_for(model_id)
        if provider not in self._cache:
            if provider is Provider.OPENAI:
                self._cache[provider] = OpenAIProvider(
                    self._keys[Provider.OPENAI], self._max_retries
                )
            elif provider is Provider.ANTHROPIC:
                self._cache[provider] = AnthropicProvider(
                    self._keys[Provider.ANTHROPIC], self._max_retries
                )
            else:
                raise ValueError(
                    f"{model_id!r} belongs to provider {provider} which has no chat "
                    f"adapter. Cohere reranks, Google embeds; neither answers."
                )
        return self._cache[provider]

    def reranker_for(self, model_id: str) -> CohereReranker:
        provider = provider_for(model_id)
        if provider is not Provider.COHERE:
            raise ValueError(
                f"{model_id!r} is not a dedicated reranker. Chat models are "
                f"reranked through the listwise path instead."
            )
        if self._reranker is None:
            self._reranker = CohereReranker(
                self._keys[Provider.COHERE], self._max_retries
            )
        return self._reranker
