"""Layer 02 / 09 - request admission.

Every chat request is anonymous and arrives from a public page, so admission is
the only thing standing between a client's budget and the open internet. Three
checks, cheapest first:

    origin allowlist  ->  per-IP rate limit  ->  property cap  ->  account cap

The Origin header both authenticates and identifies: the widget never sends a
property id it chose itself, because a caller who can pick the property id can
bill any client they like.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass

from fastapi import Header, HTTPException, Request, status

from app.config import Settings, get_settings
from app.gateway.budget import UsageExceeded, get_usage_limiter
from app.gateway.llm_gateway import LLMGateway
from app.models.domain import TurnRecord
from app.observability.metrics import METRICS
from app.storage.chat_log import ChatLog
from app.storage.db import get_db
from app.storage.properties import Property, PropertyRepository


@dataclass(slots=True)
class _Bucket:
    tokens: float
    updated_at: float


class RateLimiter:
    """Per-IP token bucket.

    In-process, so with several containers the effective limit is the
    configured rate times the container count. That is a deliberate trade: the
    alternative is Redis, and the design settled on no shared session store.
    The daily spend cap is the real backstop - this only smooths bursts.
    """

    def __init__(self, per_minute: int, burst: int) -> None:
        self._rate = per_minute / 60.0
        self._burst = float(burst)
        self._buckets: dict[str, _Bucket] = {}

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        bucket = self._buckets.get(key)
        if bucket is None:
            self._buckets[key] = _Bucket(tokens=self._burst - 1, updated_at=now)
            return True

        elapsed = now - bucket.updated_at
        bucket.tokens = min(self._burst, bucket.tokens + elapsed * self._rate)
        bucket.updated_at = now

        if bucket.tokens < 1:
            return False
        bucket.tokens -= 1
        return True

    def prune(self, max_entries: int = 10_000) -> None:
        """Bound memory against an IP-rotating flood.

        Age alone is not enough: during exactly the flood this defends against,
        every bucket is fresh, so an age-only sweep frees nothing and the map
        grows without limit. Stale entries go first; if that is not enough, the
        oldest are evicted until the map is back under the cap.
        """
        if len(self._buckets) <= max_entries:
            return

        cutoff = time.monotonic() - 300
        for key in [k for k, b in self._buckets.items() if b.updated_at < cutoff]:
            del self._buckets[key]

        excess = len(self._buckets) - max_entries
        if excess > 0:
            oldest = sorted(self._buckets.items(), key=lambda kv: kv[1].updated_at)
            for key, _ in oldest[:excess]:
                del self._buckets[key]


_limiter: RateLimiter | None = None


def get_limiter(settings: Settings | None = None) -> RateLimiter:
    global _limiter
    if _limiter is None:
        settings = settings or get_settings()
        _limiter = RateLimiter(settings.rate_limit_per_minute, settings.rate_limit_burst)
    return _limiter


def get_repo() -> PropertyRepository:
    return PropertyRepository(get_db())


def client_ip(request: Request, settings: Settings | None = None) -> str:
    """The key the rate limiter buckets on.

    X-Forwarded-For is written by the client and appended to by each proxy, so
    the left-hand entries are whatever the caller chose to send. Reading the
    first one let a caller vary the header and get a fresh bucket per request,
    which is the whole limit bypassed. Only the hops a proxy actually appended
    can be trusted, so TRUSTED_PROXY_HOPS says how many there are and this
    counts in from the right; with none configured the socket peer is the only
    honest answer.
    """
    settings = settings or get_settings()
    hops = settings.trusted_proxy_hops
    if hops > 0:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            chain = [part.strip() for part in forwarded.split(",") if part.strip()]
            # The last entry is the peer the nearest proxy saw; one hop further
            # in for each additional proxy. A chain shorter than the configured
            # hop count means the header did not come from where it should
            # have, so it is not used at all.
            if len(chain) >= hops:
                return chain[-hops]
    return request.client.host if request.client else "unknown"


async def resolve_property(
    request: Request,
    origin: str | None = Header(default=None),
) -> Property:
    """Identify the Property from the Origin header and admit the request."""
    settings = get_settings()
    repo = get_repo()

    # Browsers omit Origin on same-origin GETs but always send it on the
    # cross-origin POST the widget makes. Referer is a fallback for embedded
    # contexts that strip it.
    candidate = origin or request.headers.get("referer") or ""
    property_id = await repo.resolve_origin(candidate)

    if property_id is None:
        METRICS.incr("admission.rejected", reason="origin")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This origin is not registered for any property.",
        )

    prop = await repo.get(property_id)
    if prop is None or not prop.active:
        METRICS.incr("admission.rejected", reason="inactive")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Property is not active."
        )

    limiter = get_limiter(settings)
    limiter.prune()
    if not limiter.allow(f"{property_id}:{client_ip(request, settings)}"):
        METRICS.incr("admission.rejected", reason="rate_limit")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many questions in a short time. Please wait a moment.",
            headers={"Retry-After": "10"},
        )

    within, spent = await repo.within_budget(prop)
    if not within:
        METRICS.incr("admission.rejected", reason="spend_cap")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="This assistant has reached its daily limit. Please contact the "
            "property directly.",
            headers={"Retry-After": "3600", "X-Spend-Today": f"{spent:.4f}"},
        )

    # The account cap would stop this turn inside the gateway anyway; refusing
    # at admission turns a mid-stream exception into a clean 429.
    try:
        await get_usage_limiter(settings).check("chat")
    except UsageExceeded as exc:
        METRICS.incr("admission.rejected", reason="account_cap")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="This assistant has reached its daily limit. Please contact the "
            "property directly.",
            headers={"Retry-After": "3600"},
        ) from exc

    return prop


PLACEHOLDER_ADMIN_KEY = "change-me-before-deploying"


async def require_admin(
    x_admin_key: str | None = Header(default=None),
) -> None:
    """Guards ingestion, corpus deletion and reporting.

    Operator-only, never reachable from the widget. The placeholder key is
    rejected outright: a deployment that forgot to set ADMIN_API_KEY would
    otherwise expose upload and delete behind a value published in
    .env.example.
    """
    settings = get_settings()
    if settings.admin_api_key == PLACEHOLDER_ADMIN_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ADMIN_API_KEY is still the placeholder. Set it before use.",
        )
    # Constant-time: a plain != leaks the key one byte at a time under timing.
    if not x_admin_key or not secrets.compare_digest(
        x_admin_key, settings.admin_api_key
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid admin key."
        )


def admin_key_ok(x_admin_key: str | None) -> bool:
    """Whether a key is the admin key, as a question rather than a gate.

    `require_admin` refuses a request; this reports. It is for surfaces that
    stay open to everyone but show an operator more - the step trace on the
    chat stream - where absence is the ordinary case and not an error.
    """
    settings = get_settings()
    if not x_admin_key or settings.admin_api_key == PLACEHOLDER_ADMIN_KEY:
        return False
    return secrets.compare_digest(x_admin_key, settings.admin_api_key)


def make_spend_recorder(repo: PropertyRepository):
    """Bridges the gateway's cost accounting to the ledger."""

    async def record(property_id: str, amount_usd: float) -> None:
        await repo.record_spend(property_id, amount_usd)

    return record


def make_turn_recorder(chat_log: ChatLog):
    """Bridges the pipeline's finished turns to the chat log."""

    async def record(turn: TurnRecord) -> None:
        await chat_log.record(turn)

    return record


def build_gateway(settings: Settings, repo: PropertyRepository) -> LLMGateway:
    return LLMGateway(settings, on_spend=make_spend_recorder(repo))
