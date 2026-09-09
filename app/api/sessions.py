"""Session identity (Layer 02).

A Visitor is anonymous and the Widget runs on someone else's page, so there is
no one to authenticate. What there does have to be is a thread key the Visitor
cannot choose. The chat log groups a conversation by `session_id`, and a value
the browser picks is a value any browser can pick - including one already in
use by somebody else. A log whose threads can be interleaved by anyone who
types a different id does not answer "what did the Guide tell my guest?", which
is the question it exists to answer.

So the server issues it:

    <thread>.<issued_at>.<signature>

signed with a key the deployment holds, echoed back by the Widget on the next
turn, and verified before it is trusted or written. Nothing is stored - the
signature *is* the proof, so verification is one HMAC and no lookup, and
sessions stay stateless (ADR 0003).

A token that is forged, tampered with, expired, or minted for another Property
is not an error: it is replaced with a fresh one and the turn proceeds. The
Visitor asked a question and should get an answer; what they do not get is a
say in which thread it lands in.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import time
from dataclasses import dataclass

from app.config import Settings, get_settings
from app.observability.metrics import METRICS

THREAD_BYTES = 9        # 12 base64url characters
SIGNATURE_BYTES = 16    # 22 base64url characters; the full digest buys nothing


@dataclass(frozen=True, slots=True)
class Session:
    thread_id: str      # what the log groups on
    token: str          # what the Widget echoes back
    issued: bool        # True when this turn minted it


# Used only when SESSION_SECRET is unset. Per-process, so tokens do not survive
# a restart and are not shared between containers - a Visitor mid-conversation
# is quietly given a new thread. Fine for a laptop, wrong for a deployment,
# which is why the lifespan warns about it at boot.
_ephemeral_secret: str | None = None


def _secret(settings: Settings) -> str:
    global _ephemeral_secret
    if settings.session_secret:
        return settings.session_secret
    if _ephemeral_secret is None:
        _ephemeral_secret = secrets.token_urlsafe(32)
    return _ephemeral_secret


def _sign(property_id: str, thread: str, issued_at: str, settings: Settings) -> str:
    """The Property is inside the signature, not just alongside it, so a token
    minted for one client cannot be replayed against another."""
    mac = hmac.new(
        _secret(settings).encode(),
        f"{property_id}|{thread}|{issued_at}".encode(),
        hashlib.sha256,
    )
    return base64.urlsafe_b64encode(mac.digest()[:SIGNATURE_BYTES]).decode().rstrip("=")


def issue_session(property_id: str, settings: Settings | None = None) -> Session:
    settings = settings or get_settings()
    thread = secrets.token_urlsafe(THREAD_BYTES)
    issued_at = str(int(time.time()))
    signature = _sign(property_id, thread, issued_at, settings)
    return Session(
        thread_id=thread, token=f"{thread}.{issued_at}.{signature}", issued=True
    )


def resolve_session(
    property_id: str, token: str | None, settings: Settings | None = None
) -> Session:
    """Trust an echoed token, or mint a fresh one."""
    settings = settings or get_settings()
    thread = _verify(property_id, token, settings)
    if thread is not None:
        return Session(thread_id=thread, token=token or "", issued=False)
    if token:
        # Worth a counter: a rate of these means either an attempt to write
        # into someone else's thread, or a rotated SESSION_SECRET cutting
        # every conversation in flight.
        METRICS.incr("session.reissued")
    return issue_session(property_id, settings)


def _verify(property_id: str, token: str | None, settings: Settings) -> str | None:
    """Return the thread id a valid token carries, or None."""
    if not token:
        return None

    parts = token.split(".")
    if len(parts) != 3:
        return None
    thread, issued_at, signature = parts

    expected = _sign(property_id, thread, issued_at, settings)
    # Constant-time: a plain != leaks the signature one byte at a time, and a
    # forged signature is a forged thread key.
    if not hmac.compare_digest(signature, expected):
        return None

    max_age = settings.session_max_age_hours * 3600
    if max_age > 0:
        try:
            age = time.time() - int(issued_at)
        except ValueError:
            return None
        # A clock that moved backwards, or a timestamp from the future, is not
        # a valid age - treat it the way an expired token is treated.
        if age < 0 or age > max_age:
            return None

    return thread
