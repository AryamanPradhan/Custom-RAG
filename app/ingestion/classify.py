"""Categorise Sources, and detect which unit they describe.

Category drives the retrieval filter; unit drives the "answer for all cottages"
behaviour. Both are decided once at ingestion rather than per query, because
ingestion happens rarely and queries happen constantly.

Cheap heuristics run first and settle most pages - a URL ending /rooms/ or a
title containing "Cancellation Policy" needs no model call. The model is only
asked about the pages the heuristics cannot place.
"""

from __future__ import annotations

import json
import re

from app.gateway.llm_gateway import LLMGateway, Task
from app.logging_setup import get_logger
from app.models.domain import DocCategory, Document

log = get_logger(__name__)

# Ordered: the first pattern to match wins, so put the specific before the general.
_RULES: list[tuple[DocCategory, re.Pattern[str]]] = [
    (DocCategory.POLICIES, re.compile(
        r"\b(polic|terms|conditions|cancellation|refund|house rules|check[- ]?in"
        r"|check[- ]?out|pets?|smoking|deposit)\b", re.I)),
    (DocCategory.RATES, re.compile(r"\b(rate|pricing|prices|tariff|cost per night)\b", re.I)),
    (DocCategory.ROOMS, re.compile(
        r"\b(rooms?|suites?|cottages?|villas?|apartments?|accommodation|cabins?"
        r"|bedrooms?|stay)\b", re.I)),
    (DocCategory.DINING, re.compile(
        r"\b(dining|restaurant|breakfast|menu|bar|caf[eé]|food|kitchen)\b", re.I)),
    (DocCategory.AMENITIES, re.compile(
        r"\b(amenit|facilit|pool|spa|gym|wifi|wi-fi|parking|laundry|services)\b", re.I)),
    (DocCategory.ACTIVITIES, re.compile(
        r"\b(activit|experience|tour|trek|excursion|things to do|attractions?"
        r"|sightseeing)\b", re.I)),
    (DocCategory.LOCATION, re.compile(
        r"\b(location|directions|how to (get|reach)|getting here|map|address"
        r"|nearby|transport|airport)\b", re.I)),
    (DocCategory.FAQ, re.compile(r"\b(faq|frequently asked|questions)\b", re.I)),
    (DocCategory.CONTACT, re.compile(r"\b(contact|enquir|inquir|reach us|get in touch)\b", re.I)),
]

_CLASSIFY_SYSTEM = """\
You categorise a page from a hotel or homestay's own documents.

Pick the single category that best describes what the page is FOR. A rooms page \
that mentions breakfast is still "rooms".

If the page is about one specific named room, cottage, villa or apartment, put \
that name in "unit" exactly as the page writes it. If it covers the whole \
property or several units, leave unit null.

Return JSON only."""

_SCHEMA = {
    "name": "page_classification",
    "schema": {
        "type": "object",
        "properties": {
            "category": {"type": "string", "enum": [str(c) for c in DocCategory]},
            "unit": {"type": ["string", "null"]},
        },
        "required": ["category", "unit"],
        "additionalProperties": False,
    },
}


def classify_heuristic(doc: Document) -> DocCategory | None:
    """Match against URL path and title only.

    Body text is deliberately excluded: nearly every page on a hotel site
    mentions rooms and breakfast, so matching on body turns everything into
    DocCategory.ROOMS.
    """
    haystack = f"{doc.uri} {doc.title}"
    for category, pattern in _RULES:
        if pattern.search(haystack):
            return category
    return None


async def classify_document(
    doc: Document,
    gateway: LLMGateway | None = None,
    *,
    detect_units: bool = True,
) -> tuple[DocCategory, str | None]:
    """Return (category, unit). Heuristics first, model only when needed.

    The model is asked in two cases: the heuristics could not place the page,
    or the page looks like it describes rooms and a unit name is worth
    extracting. Everything else - policies, dining, directions, contact - is
    settled for free. Without this, a 300-page document set makes 300 billed
    calls.

    When the heuristic already placed the page, the model is answering a
    narrower question than it thinks: only `unit` is taken from it. It once
    supplied the category as well, and relabelled `02-rooms-and-rate-card` from
    `rates` to `rooms` - defensible in isolation, except it left the corpus
    with no `rates` Source at all. A title matched by an explicit rule is the
    stronger evidence; the model is here for the name it can read off the page.
    """
    guess = classify_heuristic(doc)

    if gateway is None:
        return guess or DocCategory.OTHER, None

    needs_unit = detect_units and guess in (DocCategory.ROOMS, DocCategory.RATES)
    if guess is not None and not needs_unit:
        return guess, None

    try:
        result = await gateway.complete(
            Task.REWRITE,  # same cheap model tier as query rewriting
            system=_CLASSIFY_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"URL: {doc.uri}\nTITLE: {doc.title}\n\n"
                        f"CONTENT (truncated):\n{doc.text[:1500]}"
                    ),
                }
            ],
            max_tokens=200,
            json_schema=_SCHEMA,
            property_id=doc.property_id,
        )
        data = json.loads(result.text)
        raw = data.get("category", "other")
        model_category = DocCategory(raw) if raw in set(DocCategory) else DocCategory.OTHER
        unit = (data.get("unit") or "").strip() or None
        return guess or model_category, unit
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "Classifier call failed; falling back to the title heuristic.",
            uri=doc.uri,
            error=f"{type(exc).__name__}: {exc}",
        )
        return guess or DocCategory.OTHER, None
