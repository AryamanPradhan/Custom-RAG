"""Model catalogue + cost accounting (Layer 06 / Layer 08).

Prices are USD per 1M tokens. Chat runs on OpenAI and embeddings on Google, so
cost still has to be comparable across vendors - this table is what makes a
per-property spend cap meaningful when a single answer touches both. The
Anthropic rows stay priced so that routing one task back to Claude is an env
change, not a code change.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Provider(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    COHERE = "cohere"


@dataclass(frozen=True, slots=True)
class ModelSpec:
    model_id: str
    provider: Provider
    input_per_mtok: float
    output_per_mtok: float
    # Discounts on cached prompt tokens differ per vendor: Anthropic reads
    # cache at 0.1x and charges 1.25x to write it, OpenAI reads at 0.5x and
    # never bills the write. Pricing them with one shared constant would
    # under-report OpenAI spend five-fold against the daily cap.
    cache_read_multiplier: float = 0.1
    cache_write_multiplier: float = 1.25
    # Cohere bills rerank per search unit (one query against up to 100
    # documents), not per token. Priced as its own field rather than folded
    # into an invented token rate, because the two do not convert: a rerank of
    # 40 passages costs the same as a rerank of 4.
    per_search_unit_usd: float = 0.0
    # Anthropic-only knobs; ignored for other providers.
    supports_effort: bool = False
    supports_adaptive_thinking: bool = False


CATALOGUE: dict[str, ModelSpec] = {
    # -- OpenAI: answers, rewriting, reranking, verification, eval --------
    # OpenAI reads cached prompt tokens at half price and never charges to
    # populate the cache.
    "gpt-4o-mini": ModelSpec(
        "gpt-4o-mini", Provider.OPENAI, 0.15, 0.60,
        cache_read_multiplier=0.5, cache_write_multiplier=0.0,
    ),
    "gpt-4o": ModelSpec(
        "gpt-4o", Provider.OPENAI, 2.50, 10.00,
        cache_read_multiplier=0.5, cache_write_multiplier=0.0,
    ),
    # -- Anthropic: unused by default, kept priced so a task can be routed
    #    back to Claude with an env change ------------------------------
    "claude-haiku-4-5": ModelSpec("claude-haiku-4-5", Provider.ANTHROPIC, 1.00, 5.00),
    "claude-sonnet-5": ModelSpec(
        "claude-sonnet-5", Provider.ANTHROPIC, 2.00, 10.00,
        supports_effort=True, supports_adaptive_thinking=True,
    ),
    "claude-opus-5": ModelSpec(
        "claude-opus-5", Provider.ANTHROPIC, 5.00, 25.00,
        supports_effort=True, supports_adaptive_thinking=True,
    ),
    # -- Cohere: reranking -------------------------------------------------
    # $2.00 per 1,000 search units. Verify against Cohere's current pricing
    # before trusting the daily cap - this table is what the cap counts.
    "rerank-v3.5": ModelSpec(
        "rerank-v3.5", Provider.COHERE, 0.0, 0.0, per_search_unit_usd=0.002
    ),
    # -- Google: embeddings ----------------------------------------------
    "gemini-embedding-2": ModelSpec("gemini-embedding-2", Provider.GOOGLE, 0.20, 0.0),
    "gemini-embedding-001": ModelSpec(
        "gemini-embedding-001", Provider.GOOGLE, 0.15, 0.0
    ),
}


class UnknownModelError(KeyError):
    """Raised when a model id is not in the catalogue.

    Deliberately loud: a silent fallback would under-report cost and quietly
    defeat the per-property spend cap.
    """


def spec_for(model_id: str) -> ModelSpec:
    try:
        return CATALOGUE[model_id]
    except KeyError:
        raise UnknownModelError(
            f"{model_id!r} is not in the pricing catalogue. Add it to "
            f"app/gateway/pricing.py before routing traffic to it, otherwise "
            f"its spend is invisible to the daily cap."
        ) from None


def provider_for(model_id: str) -> Provider:
    return spec_for(model_id).provider


def cost_usd(
    model_id: str,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    search_units: int = 0,
) -> float:
    """Cost of one request in USD, normalised across providers."""
    spec = spec_for(model_id)
    per_token = (
        input_tokens * spec.input_per_mtok
        + cache_read_tokens * spec.input_per_mtok * spec.cache_read_multiplier
        + cache_write_tokens * spec.input_per_mtok * spec.cache_write_multiplier
        + output_tokens * spec.output_per_mtok
    ) / 1_000_000
    return per_token + search_units * spec.per_search_unit_usd
