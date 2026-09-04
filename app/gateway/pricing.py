"""Model catalogue + cost accounting (Layer 06 / Layer 08).

Prices are USD per 1M tokens. Three vendors sit behind one gateway, so cost
has to be comparable across them - this table is what makes a per-property
spend cap meaningful when a single answer touches OpenAI, Anthropic and Google.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Provider(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


@dataclass(frozen=True, slots=True)
class ModelSpec:
    model_id: str
    provider: Provider
    input_per_mtok: float
    output_per_mtok: float
    # Anthropic-only knobs; ignored for other providers.
    supports_effort: bool = False
    supports_adaptive_thinking: bool = False


CATALOGUE: dict[str, ModelSpec] = {
    # -- OpenAI: answer path ---------------------------------------------
    "gpt-4o-mini": ModelSpec("gpt-4o-mini", Provider.OPENAI, 0.15, 0.60),
    "gpt-4o": ModelSpec("gpt-4o", Provider.OPENAI, 2.50, 10.00),
    # -- Anthropic: verification, rewriting, reranking, eval --------------
    "claude-haiku-4-5": ModelSpec("claude-haiku-4-5", Provider.ANTHROPIC, 1.00, 5.00),
    "claude-sonnet-5": ModelSpec(
        "claude-sonnet-5", Provider.ANTHROPIC, 2.00, 10.00,
        supports_effort=True, supports_adaptive_thinking=True,
    ),
    "claude-opus-5": ModelSpec(
        "claude-opus-5", Provider.ANTHROPIC, 5.00, 25.00,
        supports_effort=True, supports_adaptive_thinking=True,
    ),
    # -- Google: embeddings ----------------------------------------------
    "gemini-embedding-2": ModelSpec("gemini-embedding-2", Provider.GOOGLE, 0.20, 0.0),
    "gemini-embedding-001": ModelSpec(
        "gemini-embedding-001", Provider.GOOGLE, 0.15, 0.0
    ),
}

# Anthropic prompt caching multipliers.
CACHE_READ_MULTIPLIER = 0.1
CACHE_WRITE_MULTIPLIER = 1.25


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
) -> float:
    """Cost of one request in USD, normalised across providers."""
    spec = spec_for(model_id)
    return (
        input_tokens * spec.input_per_mtok
        + cache_read_tokens * spec.input_per_mtok * CACHE_READ_MULTIPLIER
        + cache_write_tokens * spec.input_per_mtok * CACHE_WRITE_MULTIPLIER
        + output_tokens * spec.output_per_mtok
    ) / 1_000_000
