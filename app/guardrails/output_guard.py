"""Layer 05 - output verification.

The answer path runs on a small, cheap model. This is the check that makes that
safe: a second model reads the finished answer against the sources it was given
and decides whether every property-specific claim is actually supported.

It is blocking by design. An ungrounded answer is not annotated and shown - it
is replaced by a Deflection. A plausible invented cancellation policy is the
one failure mode that can cost a client real money.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from app.gateway.llm_gateway import LLMGateway, Task
from app.guardrails.patterns import contact_allowlist, pii_spans, redact_pii_except
from app.logging_setup import get_logger
from app.observability.metrics import METRICS
from app.observability.tracing import span

log = get_logger(__name__)

VERIFIER_SYSTEM = """\
You verify whether an answer about a hotel or homestay is supported by the \
sources it was given.

You will receive numbered SOURCES and an ANSWER.

A claim is SUPPORTED when a source states it, or states something it follows \
from directly. A claim is UNSUPPORTED when no source establishes it, even if \
it is plausible or generally true of hotels.

Ignore, and never report as unsupported:
- conversational framing ("Happy to help", "Let me know if...")
- statements that the answer does not know something
- suggestions to contact the property
- general courtesy or hedging

Judge only concrete claims about this property: prices, times, policies, \
amenities, capacities, locations, availability, names.

Return JSON only."""

_SCHEMA = {
    "name": "grounding_verdict",
    "schema": {
        "type": "object",
        "properties": {
            "grounded": {
                "type": "boolean",
                "description": "true when every property-specific claim is supported",
            },
            "unsupported_claims": {
                "type": "array",
                "items": {"type": "string"},
                "description": "verbatim claims from the answer that no source supports",
            },
        },
        "required": ["grounded", "unsupported_claims"],
        "additionalProperties": False,
    },
}


@dataclass(slots=True)
class GroundingVerdict:
    grounded: bool
    unsupported_claims: list[str] = field(default_factory=list)
    checked: bool = True
    detail: str = ""


async def verify_grounding(
    *,
    question: str,
    answer: str,
    context: str,
    gateway: LLMGateway,
    property_id: str | None = None,
) -> GroundingVerdict:
    """Check an answer against its sources.

    On verifier failure the answer is blocked, not passed. If the check cannot
    run, we have no evidence the answer is grounded - and this pipeline's whole
    safety argument rests on that check running.
    """
    if not answer.strip():
        return GroundingVerdict(grounded=False, detail="empty answer")

    with span("verify_grounding") as s:
        try:
            result = await gateway.complete(
                Task.VERIFY,
                system=VERIFIER_SYSTEM,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"SOURCES:\n{context}\n\n"
                            f"QUESTION: {question}\n\n"
                            f"ANSWER:\n{answer}"
                        ),
                    }
                ],
                max_tokens=700,
                json_schema=_SCHEMA,
                property_id=property_id,
            )
            data = json.loads(result.text)
            verdict = GroundingVerdict(
                grounded=bool(data.get("grounded", False)),
                unsupported_claims=list(data.get("unsupported_claims", []) or []),
            )
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            METRICS.incr("guard.verifier_unparseable")
            log.error("guard.verifier_unparseable", error=str(exc))
            return GroundingVerdict(
                grounded=False, checked=False, detail=f"verifier returned unusable output: {exc}"
            )
        except Exception as exc:  # noqa: BLE001 - provider-specific failures
            METRICS.incr("guard.verifier_failed")
            log.error("guard.verifier_failed", error=f"{type(exc).__name__}: {exc}")
            return GroundingVerdict(
                grounded=False, checked=False, detail=f"verifier unavailable: {exc}"
            )

        s.attributes["grounded"] = verdict.grounded
        s.attributes["unsupported"] = len(verdict.unsupported_claims)

    METRICS.incr(
        "guard.grounding_pass" if verdict.grounded else "guard.grounding_fail"
    )
    if not verdict.grounded:
        log.warning(
            "guard.ungrounded_answer",
            property_id=property_id,
            claims=verdict.unsupported_claims[:5],
        )
    return verdict


def scrub_answer(answer: str, *, trusted_text: str = "") -> str:
    """Last pass before the answer leaves the process.

    A Deflection is supposed to hand the Visitor the property's phone number,
    so blanket PII redaction would destroy the single most useful thing the
    Guide says. Contacts that appear in `trusted_text` - the retrieved sources
    and the configured contact route - are preserved; anything else, and every
    card number regardless of source, is redacted.
    """
    if not answer:
        return answer
    spans = pii_spans(answer)
    if not spans:
        return answer

    keep = contact_allowlist(trusted_text)
    scrubbed = redact_pii_except(answer, keep=keep)
    if scrubbed != answer:
        METRICS.incr("guard.pii_redacted", source="answer")
    return scrubbed
