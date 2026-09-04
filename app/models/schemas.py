"""Request/response contracts for the HTTP surface (Layer 01/02)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class ChatRequest(BaseModel):
    """Sessions are stateless: the widget carries the transcript and sends it
    back each turn. Everything here is therefore untrusted input, and is
    screened by the guardrails before it reaches a model."""

    message: str = Field(min_length=1, max_length=4000)
    history: list[ChatTurn] = Field(default_factory=list, max_length=20)
    session_id: str | None = Field(default=None, max_length=64)

    @field_validator("message")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("message must not be blank")
        return v


class CitationOut(BaseModel):
    index: int
    uri: str
    label: str
    snippet: str
    published_on: str = ""
    unit: str | None = None


class ChatResponse(BaseModel):
    answer: str
    citations: list[CitationOut] = Field(default_factory=list)
    deflected: bool = False
    grounded: bool = True
    trace_id: str = ""


# -- operator surface ------------------------------------------------------


class ContactRouteIn(BaseModel):
    phone: str | None = None
    email: str | None = None
    url: str | None = None
    note: str | None = None


class PropertyIn(BaseModel):
    property_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{1,62}$")
    display_name: str = Field(min_length=1, max_length=120)
    allowed_origins: list[str] = Field(min_length=1)
    contact_route: ContactRouteIn = Field(default_factory=ContactRouteIn)
    daily_spend_cap_usd: float = Field(default=5.0, gt=0, le=1000)


class PropertyOut(BaseModel):
    property_id: str
    display_name: str
    allowed_origins: list[str]
    daily_spend_cap_usd: float
    spent_today_usd: float
    last_crawled_at: str | None
    active: bool
    indexed_chunks: int | None = None


class CrawlRequest(BaseModel):
    start_url: str
    max_pages: int | None = Field(default=None, ge=1, le=2000)
    max_depth: int | None = Field(default=None, ge=1, le=10)
    include_paths: list[str] = Field(default_factory=list)
    exclude_paths: list[str] = Field(default_factory=list)


class IngestSummary(BaseModel):
    property_id: str
    documents: int
    chunks: int
    skipped_unchanged: int
    errors: list[str] = Field(default_factory=list)
    duration_seconds: float


class DriftReportOut(BaseModel):
    property_id: str
    checked: int
    changed: list[str]
    unreachable: list[str]
    is_stale: bool
    summary: str
    checked_at: str


class FeedbackRequest(BaseModel):
    rating: Literal["up", "down"]
    trace_id: str | None = Field(default=None, max_length=64)
    question: str | None = Field(default=None, max_length=1000)
    comment: str | None = Field(default=None, max_length=1000)
