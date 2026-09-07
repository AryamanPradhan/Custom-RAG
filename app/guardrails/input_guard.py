"""Layer 05 - input screening and context sanitisation.

Three untrusted inputs reach the model on every turn:

  1. the Visitor's message
  2. the conversation history - which, because sessions are stateless, arrives
     from the browser and can say anything
  3. the retrieved Chunks, which came out of owner-supplied documents

(3) is the one that gets skipped. Retrieved text is pasted into the prompt and
read with the same weight as operator instructions unless something explicitly
marks it as data. That is what sanitise_context is for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from app.guardrails.patterns import (
    detect_injection,
    detect_pii,
    neutralise_injection,
    redact_pii,
)
from app.models.domain import ScoredChunk
from app.observability.metrics import METRICS

MAX_MESSAGE_CHARS = 4000
MAX_HISTORY_TURNS = 20


class GuardOutcome(StrEnum):
    ALLOW = "allow"
    REDACTED = "redacted"
    REJECT = "reject"


@dataclass(slots=True)
class GuardVerdict:
    outcome: GuardOutcome
    message: str = ""
    reason: str = ""
    flags: list[str] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return self.outcome is GuardOutcome.REJECT


def check_message(message: str) -> GuardVerdict:
    """Screen the Visitor's message.

    Injection is rejected. PII is redacted but the question still gets answered
    - someone pasting a card number is making a mistake, not an attack, and
    refusing them outright is worse service than answering the question with
    the number stripped.
    """
    text = (message or "").strip()

    if not text:
        return GuardVerdict(GuardOutcome.REJECT, reason="empty message")

    if len(text) > MAX_MESSAGE_CHARS:
        return GuardVerdict(
            GuardOutcome.REJECT,
            reason=f"message too long ({len(text)} > {MAX_MESSAGE_CHARS} chars)",
        )

    flags = detect_injection(text)
    if flags:
        METRICS.incr("guard.injection_blocked", source="message")
        return GuardVerdict(
            GuardOutcome.REJECT,
            reason=f"prompt injection detected: {', '.join(flags)}",
            flags=flags,
        )

    pii = detect_pii(text)
    if pii:
        METRICS.incr("guard.pii_redacted", source="message")
        return GuardVerdict(
            GuardOutcome.REDACTED,
            message=redact_pii(text),
            reason=f"redacted {', '.join(pii)}",
            flags=pii,
        )

    return GuardVerdict(GuardOutcome.ALLOW, message=text)


def check_history(history: list[dict]) -> tuple[GuardVerdict, list[dict]]:
    """Screen client-supplied conversation history.

    A forged 'assistant' turn is the cheapest way to plant an instruction the
    model will treat as its own prior output, so both roles get screened, not
    just the user's.

    History is also redacted, not only inspected. The widget echoes the raw
    turn back on the next request, so a card number that check_message stripped
    from turn 1 would otherwise arrive verbatim in turn 2 and be forwarded to
    every provider the pipeline touches.

    Returns (verdict, redacted_history).
    """
    if not history:
        return GuardVerdict(GuardOutcome.ALLOW), []

    if len(history) > MAX_HISTORY_TURNS:
        return (
            GuardVerdict(
                GuardOutcome.REJECT,
                reason=f"history too long ({len(history)} > {MAX_HISTORY_TURNS} turns)",
            ),
            [],
        )

    cleaned: list[dict] = []
    redacted_any = False
    for turn in history:
        content = turn.get("content", "") or ""
        flags = detect_injection(content)
        if flags:
            METRICS.incr("guard.injection_blocked", source="history")
            return (
                GuardVerdict(
                    GuardOutcome.REJECT,
                    reason=f"prompt injection in {turn.get('role', '?')} history: "
                    f"{', '.join(flags)}",
                    flags=flags,
                ),
                [],
            )
        if detect_pii(content):
            content = redact_pii(content)
            redacted_any = True
        cleaned.append({**turn, "content": content})

    if redacted_any:
        METRICS.incr("guard.pii_redacted", source="history")
    return (
        GuardVerdict(
            GuardOutcome.REDACTED if redacted_any else GuardOutcome.ALLOW
        ),
        cleaned,
    )


def sanitise_context(chunks: list[ScoredChunk]) -> str:
    """Render retrieved Chunks as clearly-delimited, numbered data.

    Three defences, in order of importance:
      - every chunk is fenced and numbered, so the model can cite [n] and can
        tell page content apart from operator instruction
      - injected instructions inside the content are excised, not passed on
      - the ingest date rides along, because a Corpus changes only when the
        owner sends new material, and an answer
        may be citing a page that has since changed
    """
    if not chunks:
        return "(no sources retrieved)"

    blocks: list[str] = []
    for i, scored in enumerate(chunks, start=1):
        chunk = scored.chunk
        clean, fired = neutralise_injection(chunk.text)
        if fired:
            METRICS.incr("guard.injection_neutralised", source="retrieved")

        header = " > ".join(p for p in [chunk.title, *chunk.heading_path] if p)
        meta = [f"source={chunk.uri}"]
        if chunk.fetched_at:
            meta.append(f"published={chunk.fetched_at}")
        if chunk.unit:
            meta.append(f"unit={chunk.unit}")

        blocks.append(
            f"<source index=\"{i}\" {' '.join(meta)}>\n"
            f"[{i}] {header}\n{clean}\n"
            f"</source>"
        )

    return "\n\n".join(blocks)
