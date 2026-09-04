# Hotel AI Guide

An informational assistant added to homestay and hotel websites. It answers a
visitor's questions using only that property's own published content. It does
not transact.

## Language

### The product

**Guide**:
The assistant itself — the thing a visitor talks to on a property's website.
_Avoid_: bot, chatbot, agent, concierge

**Widget**:
The embeddable front-end that hosts the Guide on a property's website.
_Avoid_: plugin, embed, iframe

**Property**:
One hotel or homestay whose content the Guide answers for. Owns exactly one
corpus and one Widget.
_Avoid_: hotel, homestay, client, tenant, site

**Visitor**:
A person talking to the Guide on a property's website. Anonymous, unauthenticated,
and never assumed to have a booking.
_Avoid_: guest, user, customer, lead

### Content

**Corpus**:
Everything the Guide is allowed to answer from, for one Property.
_Avoid_: knowledge base, index, dataset

**Source**:
One artifact that went into the Corpus — a crawled page or an uploaded file.
_Avoid_: document, page, file

**Chunk**:
The retrievable unit a Source is split into. Carries the heading trail it came
from so a citation can name where the answer lives.
_Avoid_: passage, segment, node

**Heading path**:
The trail of headings above a Chunk in its Source, e.g. `Rooms > Deluxe Sea View`.
Used both as a retrieval signal and as the visible citation label.
_Avoid_: breadcrumb, hierarchy, section path

### Answering

**Grounded**:
An answer is grounded when every property-specific claim in it traces to a
retrieved Chunk. Ungrounded answers are not shown.
_Avoid_: accurate, correct, factual

**Citation**:
The Source a grounded claim came from, shown to the Visitor so a claim can be
checked against the property's own page.
_Avoid_: reference, link, footnote

**Deflection**:
What the Guide does when the Corpus cannot answer: it declines and points to the
property's own contact route. Not a failure state — the designed response to a
gap.
_Avoid_: fallback, refusal, escalation, handoff

## Deliberately out of scope

The Guide never quotes live availability, holds or makes a booking, takes
payment, or captures contact details. Those are the booking engine's job, and
the Guide points at it rather than wrapping it.
