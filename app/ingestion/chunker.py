"""Structure-aware chunking.

Naive fixed-window chunking is the single biggest quality killer in hotel RAG.
Property sites are dense with short, self-contained facts under headings -
"Check-in / Check-out", "Pet policy", "Deluxe Sea View" - and a blind 500-token
window happily splits a room's rate away from its name, or merges the pet
policy into the cancellation policy.

So: split on heading structure first, pack sections into token windows second,
and stamp every chunk with its heading path. The heading path does double duty
as a retrieval signal (it is prepended to the embedded text) and as the
human-readable citation label.
"""

from __future__ import annotations

import hashlib
import re

from app.models.domain import Chunk, DocCategory, Document

# Rough char-per-token ratio for English prose. Good enough for packing
# decisions; we never bill against it.
CHARS_PER_TOKEN = 4

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
_WS = re.compile(r"[ \t]+")
_BLANKS = re.compile(r"\n{3,}")


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def normalise(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS.sub(" ", text)
    text = _BLANKS.sub("\n\n", text)
    return text.strip()


class Section:
    __slots__ = ("heading_path", "text")

    def __init__(self, heading_path: list[str], text: str) -> None:
        self.heading_path = heading_path
        self.text = text


def split_sections(markdown: str) -> list[Section]:
    """Split markdown into sections, tracking the live heading stack."""
    matches = list(_HEADING.finditer(markdown))
    if not matches:
        return [Section([], markdown.strip())] if markdown.strip() else []

    sections: list[Section] = []
    stack: list[tuple[int, str]] = []

    preamble = markdown[: matches[0].start()].strip()
    if preamble:
        sections.append(Section([], preamble))

    for i, match in enumerate(matches):
        level = len(match.group(1))
        title = match.group(2).strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))

        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        body = markdown[match.end() : end].strip()
        if body:
            sections.append(Section([t for _, t in stack], body))
    return sections


def _split_oversized(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    """Break a section that is too large for one chunk, preferring paragraph
    boundaries and falling back to sentence boundaries."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    pieces: list[str] = []
    buffer = ""

    for para in paragraphs:
        if len(para) > max_chars:
            if buffer:
                pieces.append(buffer)
                buffer = ""
            sentences = re.split(r"(?<=[.!?])\s+", para)
            sent_buf = ""
            for sentence in sentences:
                if len(sent_buf) + len(sentence) + 1 > max_chars and sent_buf:
                    pieces.append(sent_buf.strip())
                    sent_buf = sent_buf[-overlap_chars:] if overlap_chars else ""
                sent_buf = f"{sent_buf} {sentence}".strip()
            if sent_buf:
                pieces.append(sent_buf.strip())
        elif len(buffer) + len(para) + 2 > max_chars and buffer:
            pieces.append(buffer)
            buffer = f"{buffer[-overlap_chars:]}\n\n{para}".strip() if overlap_chars else para
        else:
            buffer = f"{buffer}\n\n{para}".strip() if buffer else para

    if buffer:
        pieces.append(buffer)
    return pieces


def chunk_document(
    doc: Document,
    *,
    target_tokens: int = 450,
    overlap_tokens: int = 80,
) -> list[Chunk]:
    """Turn a document into embeddable chunks.

    Small adjacent sections are packed together up to the target size so that
    a page of six one-line FAQ answers does not become six near-useless chunks.
    """
    max_chars = target_tokens * CHARS_PER_TOKEN
    overlap_chars = overlap_tokens * CHARS_PER_TOKEN

    sections = split_sections(normalise(doc.text))
    if not sections:
        return []

    chunks: list[Chunk] = []
    position = 0
    buffer_text = ""
    buffer_path: list[str] = []

    def flush() -> None:
        nonlocal buffer_text, buffer_path, position
        if not buffer_text.strip():
            buffer_text = ""
            return
        chunks.append(_make_chunk(doc, buffer_text.strip(), buffer_path, position))
        position += 1
        buffer_text = ""

    for section in sections:
        if len(section.text) > max_chars:
            flush()
            for piece in _split_oversized(section.text, max_chars, overlap_chars):
                chunks.append(_make_chunk(doc, piece, section.heading_path, position))
                position += 1
            continue

        # Pack with the previous section only when they share a heading branch;
        # merging unrelated sections is what creates cross-topic chunks.
        same_branch = (
            buffer_path
            and section.heading_path
            and buffer_path[0] == section.heading_path[0]
        )
        if buffer_text and (
            len(buffer_text) + len(section.text) + 2 > max_chars or not same_branch
        ):
            flush()

        if not buffer_text:
            buffer_path = section.heading_path
        buffer_text = (
            f"{buffer_text}\n\n{section.text}".strip() if buffer_text else section.text
        )

    flush()
    return chunks


def _make_chunk(
    doc: Document, text: str, heading_path: list[str], position: int
) -> Chunk:
    chunk_id = hashlib.sha256(
        f"{doc.doc_id}|{position}|{text[:256]}".encode()
    ).hexdigest()[:32]
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc.doc_id,
        property_id=doc.property_id,
        text=text,
        uri=doc.uri,
        title=doc.title,
        heading_path=heading_path,
        category=doc.category if isinstance(doc.category, DocCategory) else DocCategory.OTHER,
        source_kind=doc.source_kind,
        position=position,
        token_estimate=estimate_tokens(text),
        unit=doc.unit,
        fetched_at=doc.fetched_at.date().isoformat(),
        metadata={"content_hash": doc.content_hash, "lang": doc.lang},
    )


def embedding_text(chunk: Chunk) -> str:
    """What actually gets embedded.

    Prefixing the title and heading path restores the context that chunking
    stripped away - without it, a chunk reading "11:00 AM" is unretrievable.
    """
    header_parts = [chunk.title, *chunk.heading_path]
    header = " > ".join(p for p in header_parts if p)
    return f"{header}\n\n{chunk.text}" if header else chunk.text
