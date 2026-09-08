"""Core domain objects shared by ingestion, retrieval and the agent.

Layer 04 (Context & Memory) of the reference architecture: these types are the
contract between what we ingest and what the agent is allowed to say.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum


class SourceKind(StrEnum):
    """Where a document came from. Drives trust level and refresh policy.

    Both kinds are supplied by the property owner. Nothing is fetched from the
    open web, so the Guide can only ever answer from material someone chose to
    hand over.
    """

    UPLOAD = "upload"          # PDF/DOCX/MD/HTML the owner uploaded
    STRUCTURED = "structured"  # PMS / rate feed, injected as text


class DocCategory(StrEnum):
    """Hotel-domain taxonomy. Used as a retrieval filter and to route intent."""

    ROOMS = "rooms"
    AMENITIES = "amenities"
    POLICIES = "policies"          # cancellation, check-in, pets, smoking
    DINING = "dining"
    LOCATION = "location"          # directions, transport, neighbourhood
    ACTIVITIES = "activities"      # tours, experiences, local attractions
    RATES = "rates"
    FAQ = "faq"
    CONTACT = "contact"
    OTHER = "other"


@dataclass(slots=True)
class Document:
    """One retrieved source artifact before chunking."""

    property_id: str
    source_kind: SourceKind
    uri: str                       # upload://<filename>, or a feed's own id
    title: str
    text: str
    category: DocCategory = DocCategory.OTHER
    # Which accommodation unit this Source is about, when the Property has
    # several (Casa Verde's "The Barn", "The Loft"). None = property-wide.
    unit: str | None = None
    lang: str = "en"
    fetched_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict = field(default_factory=dict)

    @property
    def doc_id(self) -> str:
        return hashlib.sha256(f"{self.property_id}|{self.uri}".encode()).hexdigest()[:32]

    @property
    def content_hash(self) -> str:
        """Lets the pipeline skip re-embedding pages that have not changed."""
        return hashlib.sha256(self.text.encode()).hexdigest()


@dataclass(slots=True)
class Chunk:
    """An embeddable unit. `heading_path` is what makes citations readable."""

    chunk_id: str
    doc_id: str
    property_id: str
    text: str
    uri: str
    title: str
    heading_path: list[str]
    category: DocCategory
    source_kind: SourceKind
    position: int
    token_estimate: int
    unit: str | None = None
    # When the Source was ingested. Shown to the Visitor as "as published
    # on ..." because a Corpus only changes when the owner sends new material,
    # so it can lag what the property actually does today.
    fetched_at: str = ""
    metadata: dict = field(default_factory=dict)

    def to_payload(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "property_id": self.property_id,
            "text": self.text,
            "uri": self.uri,
            "title": self.title,
            "heading_path": self.heading_path,
            "category": str(self.category),
            "source_kind": str(self.source_kind),
            "position": self.position,
            "unit": self.unit,
            "fetched_at": self.fetched_at,
            **self.metadata,
        }


@dataclass(slots=True)
class ScoredChunk:
    """A chunk plus the scores that got it here. Kept separate so the reranker
    can rewrite `score` without losing the retrieval provenance."""

    chunk: Chunk
    score: float
    dense_score: float | None = None
    sparse_score: float | None = None
    rerank_score: float | None = None

    @property
    def citation_label(self) -> str:
        trail = " › ".join(self.chunk.heading_path[:2]) if self.chunk.heading_path else ""
        return f"{self.chunk.title}{' — ' + trail if trail else ''}"


@dataclass(slots=True)
class Citation:
    """What the answer actually leaned on. Surfaced to the guest and to evals."""

    index: int
    uri: str
    label: str
    snippet: str
    published_on: str = ""
    unit: str | None = None


@dataclass(slots=True)
class TurnRecord:
    """One completed exchange, as it is written to the chat log.

    Built by the pipeline at the end of a turn and handed to whatever is
    recording - so it carries the *screened* question, never the raw one. A
    turn that was deflected or blocked is still a turn: the reason it did not
    answer is the most useful column in the table.
    """

    property_id: str
    session_id: str
    trace_id: str
    question: str
    answer: str
    mode: str                      # json | stream
    intent: str = "informational"
    deflected: bool = False
    grounded: bool = True
    blocked: bool = False          # rejected by the input guard, never reached a model
    reason: str = ""
    citations: list[dict] = field(default_factory=list)
    latency_ms: float = 0.0
