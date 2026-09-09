"""Property registry and spend ledger (Layer 02 / 09).

A Property's six config fields: display name, allowed origins, contact route
for Deflections, daily spend cap, last-ingest timestamp, active flag. None of
these can come from ingestion - they are facts about the client, not about the
hotel.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from urllib.parse import urlparse

from app.config import default_spend_cap_usd
from app.storage.db import Database


def normalise_origin(origin: str) -> str:
    """Reduce an Origin header to scheme://host[:port], lowercased.

    Browsers send origins without a path, but operators onboarding a client
    will paste whatever is in their address bar - "https://Casa-Verde.com/",
    "casaverde.com/rooms". Normalising on both write and read means those all
    resolve to the same key instead of silently failing the allowlist.
    """
    value = (origin or "").strip().lower()
    if not value:
        return ""
    if "//" not in value:
        value = f"https://{value}"
    parsed = urlparse(value)
    if not parsed.hostname:
        return ""
    scheme = parsed.scheme or "https"
    netloc = parsed.hostname
    if parsed.port and parsed.port not in (80, 443):
        netloc = f"{netloc}:{parsed.port}"
    return f"{scheme}://{netloc}"


@dataclass(slots=True)
class ContactRoute:
    """Where a Deflection sends the Visitor. Configured per Property."""

    phone: str | None = None
    email: str | None = None
    url: str | None = None
    note: str | None = None

    def to_json(self) -> str:
        return json.dumps(
            {"phone": self.phone, "email": self.email, "url": self.url, "note": self.note}
        )

    @classmethod
    def from_json(cls, raw: str | None) -> ContactRoute:
        if not raw:
            return cls()
        data = json.loads(raw)
        return cls(
            phone=data.get("phone"),
            email=data.get("email"),
            url=data.get("url"),
            note=data.get("note"),
        )

    def describe(self) -> str:
        """One line the answer prompt can hand to a Visitor verbatim."""
        parts = []
        if self.phone:
            parts.append(f"call {self.phone}")
        if self.email:
            parts.append(f"email {self.email}")
        if self.url:
            parts.append(f"see {self.url}")
        if not parts:
            return "contact the property directly"
        return " or ".join(parts)


@dataclass(slots=True)
class Property:
    property_id: str
    display_name: str
    contact_route: ContactRoute = field(default_factory=ContactRoute)
    daily_spend_cap_usd: float = field(default_factory=default_spend_cap_usd)
    last_ingested_at: str | None = None
    active: bool = True
    allowed_origins: list[str] = field(default_factory=list)


class PropertyRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    # -- registry --------------------------------------------------------

    async def create(
        self,
        property_id: str,
        display_name: str,
        *,
        allowed_origins: list[str],
        contact_route: ContactRoute | None = None,
        daily_spend_cap_usd: float | None = None,
    ) -> Property:
        if daily_spend_cap_usd is None:
            daily_spend_cap_usd = default_spend_cap_usd()
        now = datetime.now(UTC).isoformat()
        await self._db.conn.execute(
            """INSERT INTO properties
               (property_id, display_name, contact_route, daily_spend_cap_usd, created_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(property_id) DO UPDATE SET
                   display_name = excluded.display_name,
                   contact_route = excluded.contact_route,
                   daily_spend_cap_usd = excluded.daily_spend_cap_usd""",
            (
                property_id,
                display_name,
                (contact_route or ContactRoute()).to_json(),
                daily_spend_cap_usd,
                now,
            ),
        )
        await self.set_origins(property_id, allowed_origins)
        await self._db.conn.commit()
        result = await self.get(property_id)
        assert result is not None
        return result

    async def set_origins(self, property_id: str, origins: list[str]) -> None:
        await self._db.conn.execute(
            "DELETE FROM property_origins WHERE property_id = ?", (property_id,)
        )
        rows = [
            (normalise_origin(o), property_id)
            for o in origins
            if normalise_origin(o)
        ]
        if rows:
            await self._db.conn.executemany(
                "INSERT OR REPLACE INTO property_origins (origin, property_id) VALUES (?, ?)",
                rows,
            )
        await self._db.conn.commit()

    async def get(self, property_id: str) -> Property | None:
        cursor = await self._db.conn.execute(
            "SELECT * FROM properties WHERE property_id = ?", (property_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        cursor = await self._db.conn.execute(
            "SELECT origin FROM property_origins WHERE property_id = ?", (property_id,)
        )
        origins = [r["origin"] for r in await cursor.fetchall()]
        return _row_to_property(row, origins)

    async def resolve_origin(self, origin: str) -> str | None:
        """Origin header -> property_id. The allowlist check (Layer 09)."""
        key = normalise_origin(origin)
        if not key:
            return None
        cursor = await self._db.conn.execute(
            "SELECT property_id FROM property_origins WHERE origin = ?", (key,)
        )
        row = await cursor.fetchone()
        return row["property_id"] if row else None

    async def list_all(self) -> list[Property]:
        cursor = await self._db.conn.execute(
            "SELECT property_id FROM properties ORDER BY created_at"
        )
        ids = [r["property_id"] for r in await cursor.fetchall()]
        out = []
        for pid in ids:
            prop = await self.get(pid)
            if prop:
                out.append(prop)
        return out

    async def mark_ingested(self, property_id: str) -> None:
        await self._db.conn.execute(
            "UPDATE properties SET last_ingested_at = ? WHERE property_id = ?",
            (datetime.now(UTC).isoformat(), property_id),
        )
        await self._db.conn.commit()

    # -- spend cap -------------------------------------------------------

    @staticmethod
    def _utc_day() -> str:
        """The ledger key. UTC, matching the schema and every other timestamp -
        local dates would shift the budget window and let a DST change reopen a
        day that was already spent."""
        return datetime.now(UTC).date().isoformat()

    async def record_spend(self, property_id: str, amount_usd: float) -> None:
        today = self._utc_day()
        await self._db.conn.execute(
            """INSERT INTO spend_ledger (property_id, day, spent_usd)
               VALUES (?, ?, ?)
               ON CONFLICT(property_id, day)
               DO UPDATE SET spent_usd = spent_usd + excluded.spent_usd""",
            (property_id, today, amount_usd),
        )
        await self._db.conn.commit()

    async def spent_today(self, property_id: str) -> float:
        cursor = await self._db.conn.execute(
            "SELECT spent_usd FROM spend_ledger WHERE property_id = ? AND day = ?",
            (property_id, self._utc_day()),
        )
        row = await cursor.fetchone()
        return float(row["spent_usd"]) if row else 0.0

    async def within_budget(self, prop: Property) -> tuple[bool, float]:
        """Checked before a turn starts, not after.

        The cap is a circuit breaker on an endpoint exposed to the open
        internet, so it is allowed to overshoot by at most one turn.
        """
        spent = await self.spent_today(prop.property_id)
        return spent < prop.daily_spend_cap_usd, spent


def _row_to_property(row, origins: list[str]) -> Property:
    return Property(
        property_id=row["property_id"],
        display_name=row["display_name"],
        contact_route=ContactRoute.from_json(row["contact_route"]),
        daily_spend_cap_usd=float(row["daily_spend_cap_usd"]),
        last_ingested_at=row["last_ingested_at"],
        active=bool(row["active"]),
        allowed_origins=origins,
    )
