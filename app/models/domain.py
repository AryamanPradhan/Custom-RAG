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
    """Where a document came from. Drives trust level and refresh policy."""

    WEBSITE = "website"      # crawled from the property's own site
    UPLOAD = "upload"        # PDF/DOCX the owner uploaded
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
    uri: str                       # URL or upload://<filename>
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
    # When the Source was last fetched. Shown to the Visitor as "as published
    # on ..." because re-crawls are manual and a corpus can lag the live site.
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
