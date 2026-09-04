"""Property registry, origin allowlist and spend ledger.

The allowlist is the only thing identifying which client a public, anonymous
request belongs to, and the ledger is the only thing stopping an abusive
caller running up an unbounded bill. Both are worth testing properly.
"""

from __future__ import annotations

import pytest
import pytest_asyncio

from app.storage.db import Database
from app.storage.properties import ContactRoute, PropertyRepository, normalise_origin


@pytest_asyncio.fixture
async def repo(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    await db.connect()
    yield PropertyRepository(db)
    await db.close()


class TestOriginNormalisation:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("https://casaverde.com", "https://casaverde.com"),
            ("https://CasaVerde.com/", "https://casaverde.com"),
            ("https://casaverde.com/rooms/the-barn", "https://casaverde.com"),
            ("casaverde.com", "https://casaverde.com"),
            ("http://localhost:3000", "http://localhost:3000"),
            ("https://casaverde.com:443", "https://casaverde.com"),
        ],
    )
    def test_operator_input_variants_collapse(self, raw: str, expected: str) -> None:
        """Operators paste whatever is in the address bar; browsers send a bare
        origin. Both must resolve to the same key."""
        assert normalise_origin(raw) == expected

    def test_rejects_junk(self) -> None:
        assert normalise_origin("") == ""
        assert normalise_origin("   ") == ""


class TestRegistry:
    async def test_round_trip(self, repo: PropertyRepository) -> None:
        created = await repo.create(
            "casa-verde",
            "Casa Verde",
            allowed_origins=["https://casaverde.com"],
            contact_route=ContactRoute(phone="+44 1234 567890"),
            daily_spend_cap_usd=3.0,
        )
        assert created.display_name == "Casa Verde"

        loaded = await repo.get("casa-verde")
        assert loaded is not None
        assert loaded.contact_route.phone == "+44 1234 567890"
        assert loaded.daily_spend_cap_usd == 3.0
        assert loaded.allowed_origins == ["https://casaverde.com"]

    async def test_resolve_origin_finds_the_property(self, repo: PropertyRepository) -> None:
        await repo.create(
            "casa-verde", "Casa Verde", allowed_origins=["https://casaverde.com"]
        )
        assert await repo.resolve_origin("https://casaverde.com") == "casa-verde"
        # A browser sends the bare origin even when the visitor is deep in the site.
        assert await repo.resolve_origin("https://casaverde.com/rooms") == "casa-verde"

    async def test_unregistered_origin_resolves_to_nothing(
        self, repo: PropertyRepository
    ) -> None:
        await repo.create(
            "casa-verde", "Casa Verde", allowed_origins=["https://casaverde.com"]
        )
        assert await repo.resolve_origin("https://attacker.example") is None

    async def test_origins_do_not_leak_between_properties(
        self, repo: PropertyRepository
    ) -> None:
        await repo.create("a", "A", allowed_origins=["https://a.com"])
        await repo.create("b", "B", allowed_origins=["https://b.com"])
        assert await repo.resolve_origin("https://a.com") == "a"
        assert await repo.resolve_origin("https://b.com") == "b"

    async def test_updating_origins_removes_the_old_ones(
        self, repo: PropertyRepository
    ) -> None:
        """A client changing domain must not leave the old one authorised."""
        await repo.create("a", "A", allowed_origins=["https://old.com"])
        await repo.set_origins("a", ["https://new.com"])
        assert await repo.resolve_origin("https://old.com") is None
        assert await repo.resolve_origin("https://new.com") == "a"

    async def test_multiple_origins_for_one_property(
        self, repo: PropertyRepository
    ) -> None:
        await repo.create(
            "a", "A", allowed_origins=["https://a.com", "https://www.a.com"]
        )
        assert await repo.resolve_origin("https://www.a.com") == "a"


class TestSpendLedger:
    async def test_spend_accumulates_within_the_day(self, repo: PropertyRepository) -> None:
        await repo.create("a", "A", allowed_origins=["https://a.com"])
        await repo.record_spend("a", 0.01)
        await repo.record_spend("a", 0.02)
        assert await repo.spent_today("a") == pytest.approx(0.03)

    async def test_budget_check_flips_at_the_cap(self, repo: PropertyRepository) -> None:
        prop = await repo.create(
            "a", "A", allowed_origins=["https://a.com"], daily_spend_cap_usd=0.05
        )
        within, _ = await repo.within_budget(prop)
        assert within

        await repo.record_spend("a", 0.05)
        within, spent = await repo.within_budget(prop)
        assert not within
        assert spent == pytest.approx(0.05)

    async def test_spend_is_isolated_per_property(self, repo: PropertyRepository) -> None:
        """One client's abuse must not exhaust another client's budget."""
        await repo.create("a", "A", allowed_origins=["https://a.com"])
        await repo.create("b", "B", allowed_origins=["https://b.com"])
        await repo.record_spend("a", 5.0)
        assert await repo.spent_today("b") == 0.0


class TestContactRoute:
    def test_describes_available_channels(self) -> None:
        route = ContactRoute(phone="+44 1234 567890", email="stay@casaverde.com")
        described = route.describe()
        assert "+44 1234 567890" in described
        assert "stay@casaverde.com" in described

    def test_falls_back_when_unconfigured(self) -> None:
        assert ContactRoute().describe() == "contact the property directly"

    def test_survives_a_json_round_trip(self) -> None:
        route = ContactRoute(phone="+1 555", url="https://book.example")
        restored = ContactRoute.from_json(route.to_json())
        assert restored.phone == "+1 555"
        assert restored.url == "https://book.example"
