"""Layer 06 - Provider adapters.

Three vendors sit behind the gateway: OpenAI answers, Anthropic verifies and
reranks, Google embeds. Each SDK has a different request shape, a different
usage object and a different idea of where the system prompt goes. This module
flattens all of that into one result type so the gateway above it - and the
spend cap that depends on it - never has to care who served a call.

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
    ) -> LLMResult: ...

    def stream(
        self,
        *,
        model: str,
        messages: list[dict],
        system: str | None = None,
        max_tokens: int = 1024,
        out: LLMResult,
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
    def __init__(self, api_key: str | None) -> None:
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key, max_retries=3, timeout=90.0)

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
    ) -> LLMResult:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": self._payload(system, messages),
            "max_tokens": max_tokens,
        }
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
    ) -> AsyncIterator[str]:
        # The overloads key off literal `stream` and a typed message union;
        # this adapter deliberately speaks the neutral dict shape defined at
        # the top of the module.
        stream = await self._client.chat.completions.create(  # type: ignore[call-overload]
            model=model,
            messages=self._payload(system, messages),
            max_tokens=max_tokens,
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
# Anthropic - verification, rewriting, reranking, eval
# ---------------------------------------------------------------------------


class AnthropicProvider:
    def __init__(self, api_key: str | None) -> None:
        from anthropic import AsyncAnthropic

        kwargs: dict[str, Any] = {"max_retries": 3, "timeout": 90.0}
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
    ) -> LLMResult:
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
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
                "anthropic.refused",
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
    ) -> AsyncIterator[str]:
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        async with self._client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text
            final = await stream.get_final_message()
            out.stop_reason = final.stop_reason
            self._usage(out, getattr(final, "usage", None))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class ProviderRegistry:
    """Lazily constructs one client per vendor.

    Lazy matters: a deployment that never reaches the eval path should not need
    an Anthropic key present at import time just to boot.
    """

    def __init__(
        self,
        *,
        openai_api_key: str | None,
        anthropic_api_key: str | None,
    ) -> None:
        self._keys = {
            Provider.OPENAI: openai_api_key,
            Provider.ANTHROPIC: anthropic_api_key,
        }
        self._cache: dict[Provider, ChatProvider] = {}

    def for_model(self, model_id: str) -> ChatProvider:
        provider = provider_for(model_id)
        if provider not in self._cache:
            if provider is Provider.OPENAI:
                self._cache[provider] = OpenAIProvider(self._keys[Provider.OPENAI])
            elif provider is Provider.ANTHROPIC:
                self._cache[provider] = AnthropicProvider(self._keys[Provider.ANTHROPIC])
            else:
                raise ValueError(
                    f"{model_id!r} belongs to provider {provider} which has no chat "
                    f"adapter. Google is used for embeddings only."
                )
        return self._cache[provider]
