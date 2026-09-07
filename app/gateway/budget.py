"""Layer 06 - the account-wide usage cap.

The per-property daily cap in `app/storage/properties.py` protects one client's
budget. It cannot protect the *account*, because the calls most likely to run
away bill no property at all: an ingest of a 300-file folder, an eval sweep, a
loop in a CLI session. Those pass the per-property cap untouched.

So this is a second ceiling underneath every billable call - chat, ingestion
classification, reranking, verification, eval and embeddings alike. It is a
stop rather than a warning, because the failure it exists to prevent is finding
out from the invoice.

Three ceilings, because they fail differently:

  run spend    What this process may spend before it stops. Not persisted and
               not shared - a fresh `guide eval` or a restarted server starts
               at zero. This is the one that bounds a single experiment.
  daily spend  What the account may spend per UTC day, across every process.
               Persisted, so restarting does not hand it a fresh budget.
  daily calls  Catches a fast loop of cheap calls before its cost has even been
               reported back, which a dollar cap alone cannot do.

Any of them at 0 means unlimited. Whichever binds first wins.
"""

from __future__ import annotations

from datetime import UTC, datetime

from app.logging_setup import get_logger
from app.observability.metrics import METRICS
from app.storage.db import Database

log = get_logger(__name__)


class UsageExceeded(RuntimeError):
    """Raised before a call is made, never after it is paid for."""

    def __init__(self, limit: str, used: float, cap: float) -> None:
        self.limit = limit
        self.used = used
        self.cap = cap
        super().__init__(
            f"Account {limit} cap reached for today: {used:g} of {cap:g}. "
            f"Raise ACCOUNT_DAILY_SPEND_CAP_USD / ACCOUNT_DAILY_CALL_CAP in "
            f".env, or wait for the UTC day to roll over."
        )


class UsageLimiter:
    def __init__(
        self,
        *,
        spend_cap_usd: float,
        call_cap: int,
        run_cap_usd: float = 0.0,
        db: Database | None = None,
    ) -> None:
        self.spend_cap_usd = spend_cap_usd
        self.call_cap = call_cap
        self.run_cap_usd = run_cap_usd
        self._db = db
        self._day = _utc_day()
        self._calls = 0
        self._spent = 0.0
        self._loaded = False
        # Scoped to this process, so it survives a UTC day rolling over
        # mid-run and is never read back from the ledger.
        self._run_spent = 0.0
        self._run_calls = 0

    def attach(self, db: Database) -> None:
        """Wired once the database exists, so counters survive a restart."""
        self._db = db
        self._loaded = False

    # -- accounting ------------------------------------------------------

    def _roll(self) -> None:
        today = _utc_day()
        if today != self._day:
            self._day, self._calls, self._spent, self._loaded = today, 0, 0.0, False

    async def _load(self) -> None:
        """Read today's totals back once per day per process."""
        self._loaded = True
        if self._db is None:
            return
        try:
            cursor = await self._db.conn.execute(
                "SELECT calls, spent_usd FROM usage_ledger WHERE day = ?", (self._day,)
            )
            row = await cursor.fetchone()
        except Exception as exc:  # noqa: BLE001 - accounting must not break calls
            log.warning(
                "Could not read the day's usage ledger; counting from zero.",
                error=f"{type(exc).__name__}: {exc}",
            )
            return
        if row:
            self._calls = max(self._calls, int(row["calls"]))
            self._spent = max(self._spent, float(row["spent_usd"]))

    async def check(self, what: str) -> None:
        """Raise if this call would run past either ceiling.

        Checked before the call, so the caps can overshoot by at most the one
        call in flight - the same trade the per-property cap makes.
        """
        self._roll()
        if not self._loaded:
            await self._load()

        if self.run_cap_usd and self._run_spent >= self.run_cap_usd:
            METRICS.incr("usage.blocked", limit="run_spend", what=what)
            log.error(
                f"Run spend cap reached (${self._run_spent:.4f} of "
                f"${self.run_cap_usd:.2f}); refusing the {what} call.",
                limit="run_spend",
                what=what,
                used=round(self._run_spent, 4),
            )
            raise UsageExceeded("run spend", round(self._run_spent, 4), self.run_cap_usd)

        if self.call_cap and self._calls >= self.call_cap:
            METRICS.incr("usage.blocked", limit="calls", what=what)
            log.error(
                f"Daily call cap reached ({self._calls} of {self.call_cap}); "
                f"refusing the {what} call.",
                limit="calls",
                what=what,
                used=self._calls,
            )
            raise UsageExceeded("call", self._calls, self.call_cap)

        if self.spend_cap_usd and self._spent >= self.spend_cap_usd:
            METRICS.incr("usage.blocked", limit="spend", what=what)
            log.error(
                f"Daily spend cap reached (${self._spent:.4f} of "
                f"${self.spend_cap_usd:.2f}); refusing the {what} call.",
                limit="spend",
                what=what,
                used=round(self._spent, 4),
            )
            raise UsageExceeded("spend", round(self._spent, 4), self.spend_cap_usd)

    async def record(self, cost_usd: float, *, calls: int = 1) -> None:
        self._roll()
        self._calls += calls
        self._spent += cost_usd
        self._run_calls += calls
        self._run_spent += cost_usd
        METRICS.add("usage.spent_usd", cost_usd)

        if self._db is None:
            return
        try:
            await self._db.conn.execute(
                """INSERT INTO usage_ledger (day, calls, spent_usd)
                   VALUES (?, ?, ?)
                   ON CONFLICT(day) DO UPDATE SET
                       calls = calls + excluded.calls,
                       spent_usd = spent_usd + excluded.spent_usd""",
                (self._day, calls, cost_usd),
            )
            await self._db.conn.commit()
        except Exception as exc:  # noqa: BLE001 - never mask the caller's work
            log.warning(
                "Could not persist usage; the day's ledger is now behind.",
                error=f"{type(exc).__name__}: {exc}",
            )

    def snapshot(self) -> dict:
        self._roll()
        return {
            "day": self._day,
            "calls": self._calls,
            "call_cap": self.call_cap or None,
            "spent_usd": round(self._spent, 6),
            "spend_cap_usd": self.spend_cap_usd or None,
            "run_calls": self._run_calls,
            "run_spent_usd": round(self._run_spent, 6),
            "run_cap_usd": self.run_cap_usd or None,
        }


def _utc_day() -> str:
    return datetime.now(UTC).date().isoformat()


_limiter: UsageLimiter | None = None


def get_usage_limiter(settings=None) -> UsageLimiter:
    """One limiter per process, shared by the gateway and the embedder.

    A singleton rather than a constructor argument because the two callers are
    built in different places - the gateway by the API layer, the embedder by a
    module-level cache - and a cap that only covers one of them is not a cap.
    """
    global _limiter
    if _limiter is None:
        from app.config import get_settings

        settings = settings or get_settings()
        _limiter = UsageLimiter(
            spend_cap_usd=settings.account_daily_spend_cap_usd,
            call_cap=settings.account_daily_call_cap,
            run_cap_usd=settings.run_spend_cap_usd,
        )
        log.info(
            f"Spend caps: ${settings.run_spend_cap_usd:.2f} per run, "
            f"${settings.account_daily_spend_cap_usd:.2f} and "
            f"{settings.account_daily_call_cap} calls per day.",
            run_cap_usd=settings.run_spend_cap_usd or "unlimited",
            spend_cap_usd=settings.account_daily_spend_cap_usd or "unlimited",
            call_cap=settings.account_daily_call_cap or "unlimited",
        )
    return _limiter


def set_usage_limiter(limiter: UsageLimiter | None) -> None:
    """Used by tests, and by the CLI when it wants its own counters."""
    global _limiter
    _limiter = limiter
