"""Layer 05 - detection primitives.

Pure functions, no I/O, no model calls. Everything here runs on every turn, so
it has to be cheap; the expensive semantic checks live in output_guard.

Precision matters more than recall for the injection rules. A false positive
refuses a paying client's Visitor a real answer, which is worse than letting a
weak attempt through to a model that is separately instructed to treat
retrieved content as data.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Prompt injection
# ---------------------------------------------------------------------------

# Each rule is (name, pattern). Ordinary hotel questions must not match:
# "can I ignore the road noise", "does it have an air conditioning system".
_INJECTION_RULES: list[tuple[str, re.Pattern[str]]] = [
    (
        # "ignore previous instructions", "disregard all prior rules"
        "override_instructions",
        re.compile(
            r"\b(ignore|disregard|override|bypass|forget)\b[^.\n]{0,40}?"
            r"\b(previous|prior|above|earlier|all|any)\b[^.\n]{0,40}?"
            r"\b(instruction|prompt|rule|direction|command|guideline)s?\b",
            re.I,
        ),
    ),
    (
        # "forget everything above", "ignore all of the above"
        "discard_context",
        re.compile(
            r"\b(forget|ignore|disregard)\b[^.\n]{0,30}?\b(everything|all)\b"
            r"[^.\n]{0,25}?\b(above|before|prior|previous|so far)\b",
            re.I,
        ),
    ),
    (
        # persona hijack
        "persona_override",
        re.compile(
            r"\byou are (now|no longer)\b"
            r"|\bfrom now on,? you\b"
            r"|\bact as (if you are )?(an? )?(dan|jailbroken|unrestricted|developer mode)\b"
            r"|\bpretend (to be|you are)\b"
            r"|\bdeveloper mode\b",
            re.I,
        ),
    ),
    (
        # attempts to exfiltrate the system prompt
        "prompt_extraction",
        re.compile(
            r"\b(reveal|show|print|repeat|output|display|tell me|what is)\b[^.\n]{0,30}?"
            r"\b(your|the)\b[^.\n]{0,25}?"
            r"\b(system prompt|initial prompt|instructions|system message)\b",
            re.I,
        ),
    ),
    (
        # chat-template markers smuggled into content
        "chat_template",
        re.compile(
            r"<\|(im_start|im_end|system|user|assistant|endoftext)\|>"
            r"|\[/?INST\]|<<SYS>>|</?s>",
            re.I,
        ),
    ),
    (
        # a fake role header at the start of a line
        "role_marker",
        re.compile(r"^[ \t]*(?:#{1,6}[ \t]*)?(system|assistant)[ \t]*:", re.I | re.M),
    ),
]


def detect_injection(text: str) -> list[str]:
    """Return the names of every injection rule that fired. Empty means clean."""
    if not text:
        return []
    return [name for name, pattern in _INJECTION_RULES if pattern.search(text)]


def neutralise_injection(text: str) -> tuple[str, list[str]]:
    """Replace injected instructions with a marker, keeping the rest intact.

    Used on retrieved Chunks. Dropping the whole chunk would lose legitimate
    content sitting next to the hostile string; excising only the matched span
    keeps the page usable and makes the tampering visible in the prompt.
    """
    fired: list[str] = []
    out = text
    for name, pattern in _INJECTION_RULES:
        if pattern.search(out):
            fired.append(name)
            out = pattern.sub("[neutralised]", out)
    return out, fired


# ---------------------------------------------------------------------------
# PII
# ---------------------------------------------------------------------------

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]{2,}")
_CARD_CANDIDATE = re.compile(r"\b(?:\d[ -]?){12,22}\d\b")
_PHONE_CANDIDATE = re.compile(r"\+?\d[\d\s().-]{7,}\d")


def _luhn(digits: str) -> bool:
    """Card checksum. Without it, any long digit run - a booking reference, a
    postcode sequence - would be redacted as a card number."""
    total = 0
    for i, char in enumerate(reversed(digits)):
        n = int(char)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def _card_spans(text: str) -> list[tuple[int, int]]:
    spans = []
    for match in _CARD_CANDIDATE.finditer(text):
        digits = re.sub(r"\D", "", match.group())
        if 13 <= len(digits) <= 19 and _luhn(digits):
            spans.append(match.span())
    return spans


def _phone_spans(text: str, exclude: list[tuple[int, int]]) -> list[tuple[int, int]]:
    spans = []
    for match in _PHONE_CANDIDATE.finditer(text):
        digits = re.sub(r"\D", "", match.group())
        # 9+ digits keeps dates (2026-08-12) and times out of the net.
        if len(digits) < 9:
            continue
        if any(s <= match.start() < e for s, e in exclude):
            continue
        spans.append(match.span())
    return spans


def pii_spans(text: str) -> list[tuple[int, int, str, str]]:
    """Every PII occurrence as (start, end, kind, value).

    Exposed so callers can make per-occurrence decisions - the answer path has
    to redact a Visitor's phone number while preserving the property's own.
    """
    if not text:
        return []
    cards = _card_spans(text)
    spans = [(s, e, "credit_card", text[s:e]) for s, e in cards]
    spans += [(m.start(), m.end(), "email", m.group()) for m in _EMAIL.finditer(text)]
    spans += [(s, e, "phone", text[s:e]) for s, e in _phone_spans(text, cards)]
    return sorted(spans, key=lambda r: r[0])


def detect_pii(text: str) -> list[str]:
    """Return the kinds of PII present. Empty means clean."""
    if not text:
        return []
    kinds: list[str] = []
    cards = _card_spans(text)
    if cards:
        kinds.append("credit_card")
    if _EMAIL.search(text):
        kinds.append("email")
    if _phone_spans(text, cards):
        kinds.append("phone")
    return kinds


def redact_pii(text: str) -> str:
    """Replace PII with typed markers.

    Applied before the text reaches a provider or a log line, so a Visitor who
    pastes a card number does not have it sent to three vendors and written to
    disk.
    """
    if not text:
        return text

    return redact_pii_except(text, keep=frozenset())


def redact_pii_except(text: str, *, keep: frozenset[str] | set[str]) -> str:
    """Redact PII except values that normalise into `keep`.

    `keep` holds digit-and-letter-normalised values that are legitimately
    public - the property's own phone and email, as they appear in retrieved
    sources. Card numbers are never kept, whatever the source says.
    """
    if not text:
        return text

    out = text
    for start, end, kind, value in sorted(pii_spans(text), key=lambda r: r[0], reverse=True):
        if kind != "credit_card" and normalise_contact(value) in keep:
            continue
        out = out[:start] + f"[redacted:{kind}]" + out[end:]
    return out


def normalise_contact(value: str) -> str:
    """Comparison key for a phone or email: case- and punctuation-insensitive,
    so "+44 7700 900123" and "447700900123" are the same contact."""
    return re.sub(r"[^a-z0-9@.]", "", value.strip().lower())


def contact_allowlist(*texts: str) -> frozenset[str]:
    """Normalised phone/email values appearing in trusted text."""
    keep: set[str] = set()
    for text in texts:
        for _, _, kind, value in pii_spans(text or ""):
            if kind != "credit_card":
                keep.add(normalise_contact(value))
    return frozenset(keep)
