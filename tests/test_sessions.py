"""Layer 02 - the thread key a Visitor does not get to choose.

The chat log groups a conversation by session id, so the id is the only thing
making "what did the Guide tell my guest?" answerable. A browser-chosen value
is one any browser can choose, which is why it is signed here.

Everything below is one of two claims: a token this server issued is trusted,
and anything else is quietly replaced rather than believed or refused. The
Visitor asked a question either way and should get an answer.
"""

from __future__ import annotations

import time

import pytest

from app.api.sessions import Session, issue_session, resolve_session
from app.config import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings(session_secret="test-secret", session_max_age_hours=24)


def _tamper(token: str, *, part: int, value: str) -> str:
    parts = token.split(".")
    parts[part] = value
    return ".".join(parts)


class TestIssuing:
    def test_a_fresh_session_is_marked_issued(self, settings: Settings) -> None:
        session = issue_session("casa-verde", settings)
        assert session.issued
        assert session.thread_id
        assert session.token.startswith(f"{session.thread_id}.")

    def test_no_token_gets_one(self, settings: Settings) -> None:
        for absent in (None, ""):
            assert resolve_session("casa-verde", absent, settings).issued

    def test_threads_are_unique(self, settings: Settings) -> None:
        threads = {issue_session("casa-verde", settings).thread_id for _ in range(200)}
        assert len(threads) == 200

    def test_the_token_fits_the_request_schema(self, settings: Settings) -> None:
        """ChatRequest caps session_id at 64 characters, and a token longer
        than that would be rejected as a 422 on the turn after it was issued."""
        assert len(issue_session("casa-verde", settings).token) <= 64


class TestVerifying:
    def test_an_issued_token_round_trips(self, settings: Settings) -> None:
        first = issue_session("casa-verde", settings)
        second = resolve_session("casa-verde", first.token, settings)

        assert not second.issued
        assert second.thread_id == first.thread_id
        assert second.token == first.token

    def test_a_forged_signature_is_replaced(self, settings: Settings) -> None:
        """The attack the signature exists to stop: picking somebody else's
        thread id and having your turns filed into their conversation."""
        stolen = issue_session("casa-verde", settings)
        forged = _tamper(stolen.token, part=2, value="x" * 22)

        resolved = resolve_session("casa-verde", forged, settings)
        assert resolved.issued
        assert resolved.thread_id != stolen.thread_id

    def test_a_swapped_thread_is_replaced(self, settings: Settings) -> None:
        victim = issue_session("casa-verde", settings)
        attacker = issue_session("casa-verde", settings)
        # The attacker's own valid signature over the victim's thread id.
        spliced = _tamper(attacker.token, part=0, value=victim.thread_id)

        resolved = resolve_session("casa-verde", spliced, settings)
        assert resolved.thread_id != victim.thread_id

    def test_another_propertys_token_is_replaced(self, settings: Settings) -> None:
        """The Property is signed into the token, so one client's session is
        not a key into another's log."""
        theirs = issue_session("glass-hotel", settings)
        resolved = resolve_session("casa-verde", theirs.token, settings)

        assert resolved.issued
        assert resolved.thread_id != theirs.thread_id

    def test_another_deployments_token_is_replaced(self) -> None:
        elsewhere = Settings(session_secret="a-different-secret")
        here = Settings(session_secret="test-secret")
        theirs = issue_session("casa-verde", elsewhere)

        assert resolve_session("casa-verde", theirs.token, here).issued

    @pytest.mark.parametrize(
        "junk",
        [
            "nodots",
            "two.parts",
            "far.too.many.parts.here",
            "...",
            "a.b.c",
            "  ",
        ],
    )
    def test_malformed_tokens_are_replaced(self, junk: str, settings: Settings) -> None:
        assert resolve_session("casa-verde", junk, settings).issued


class TestExpiry:
    def _aged(self, property_id: str, hours: float, settings: Settings) -> str:
        from app.api.sessions import _sign

        thread = "abcdefghijkl"
        issued_at = str(int(time.time() - hours * 3600))
        return f"{thread}.{issued_at}.{_sign(property_id, thread, issued_at, settings)}"

    def test_a_token_inside_the_window_survives(self, settings: Settings) -> None:
        token = self._aged("casa-verde", 23, settings)
        assert not resolve_session("casa-verde", token, settings).issued

    def test_an_expired_token_is_replaced(self, settings: Settings) -> None:
        token = self._aged("casa-verde", 25, settings)
        assert resolve_session("casa-verde", token, settings).issued

    def test_a_token_from_the_future_is_replaced(self, settings: Settings) -> None:
        """A clock that moved backwards, or a timestamp someone chose. Either
        way it is not an age this server can reason about."""
        token = self._aged("casa-verde", -5, settings)
        assert resolve_session("casa-verde", token, settings).issued

    def test_zero_never_expires(self) -> None:
        settings = Settings(session_secret="test-secret", session_max_age_hours=0)
        token = self._aged("casa-verde", 24 * 365, settings)
        assert not resolve_session("casa-verde", token, settings).issued

    def test_a_non_numeric_timestamp_is_replaced(self, settings: Settings) -> None:
        from app.api.sessions import _sign

        thread, issued_at = "abcdefghijkl", "not-a-time"
        signature = _sign("casa-verde", thread, issued_at, settings)
        token = f"{thread}.{issued_at}.{signature}"

        # Correctly signed, so only the age check can reject it.
        assert resolve_session("casa-verde", token, settings).issued


class TestWithoutAConfiguredSecret:
    def test_tokens_still_verify_within_one_process(self) -> None:
        """No SESSION_SECRET means a per-process key. Conversations do not
        survive a restart, but they hold for as long as the process does."""
        settings = Settings(session_secret=None)
        issued = issue_session("casa-verde", settings)

        resolved = resolve_session("casa-verde", issued.token, settings)
        assert not resolved.issued
        assert resolved.thread_id == issued.thread_id


class TestSessionShape:
    def test_the_thread_is_the_token_without_its_proof(self, settings: Settings) -> None:
        """The log stores the thread, not the token: the signature is proof of
        who may claim the thread, not part of its identity."""
        session: Session = issue_session("casa-verde", settings)
        assert session.token.split(".")[0] == session.thread_id
        assert session.thread_id not in session.token.split(".")[2]
