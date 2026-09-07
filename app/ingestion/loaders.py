"""Layer 03 - Document loaders for owner uploads.

The only way content enters a Corpus. Hotels keep what matters in exactly these
formats - the rate card PDF, the house-rules DOCX, the breakfast menu - and
they are higher-trust than a marketing site precisely because someone chose to
hand each one over.
"""

from __future__ import annotations

import io
from pathlib import Path

import trafilatura

from app.logging_setup import get_logger
from app.models.domain import Document, SourceKind

log = get_logger(__name__)

SUPPORTED_SUFFIXES = {".pdf", ".docx", ".md", ".markdown", ".txt", ".html", ".htm"}


class UnsupportedDocument(ValueError):
    pass


def load_bytes(
    property_id: str, filename: str, data: bytes, *, title: str | None = None
) -> Document:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise UnsupportedDocument(
            f"{filename}: unsupported type {suffix or '(none)'}. "
            f"Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
        )

    if suffix == ".pdf":
        text = _load_pdf(data)
    elif suffix == ".docx":
        text = _load_docx(data)
    elif suffix in (".html", ".htm"):
        text = _load_html(data)
    else:
        text = data.decode("utf-8", errors="replace")

    text = text.strip()
    if len(text) < 40:
        raise UnsupportedDocument(
            f"{filename}: extracted only {len(text)} characters. "
            "If this is a scanned PDF it needs OCR before ingestion."
        )

    return Document(
        property_id=property_id,
        source_kind=SourceKind.UPLOAD,
        uri=f"upload://{filename}",
        title=title or Path(filename).stem.replace("_", " ").replace("-", " ").title(),
        text=text,
    )


def _load_pdf(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages: list[str] = []
    for i, page in enumerate(reader.pages, start=1):
        content = (page.extract_text() or "").strip()
        if content:
            # Page headings give the chunker structure to split on and make
            # citations precise enough to be checkable ("Rate Card, page 3").
            pages.append(f"## Page {i}\n\n{content}")
    return "\n\n".join(pages)


def _load_docx(data: bytes) -> str:
    import docx

    document = docx.Document(io.BytesIO(data))
    lines: list[str] = []

    for para in document.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        # python-docx returns None for a paragraph with no explicit style.
        style = ((para.style.name if para.style else None) or "").lower()
        if style.startswith("heading"):
            level = "".join(c for c in style if c.isdigit()) or "2"
            lines.append(f"{'#' * min(int(level) + 1, 6)} {text}")
        else:
            lines.append(text)

    # Tables carry the rate grids and amenity matrices - render them as
    # markdown so the chunker keeps rows intact.
    for table in document.tables:
        rows = [
            " | ".join(cell.text.strip().replace("\n", " ") for cell in row.cells)
            for row in table.rows
        ]
        rows = [r for r in rows if r.replace("|", "").strip()]
        if len(rows) >= 2:
            header, *body = rows
            separator = " | ".join("---" for _ in header.split(" | "))
            lines.append("\n".join(["", header, separator, *body, ""]))

    return "\n\n".join(lines)


def _load_html(data: bytes) -> str:
    html = data.decode("utf-8", errors="replace")
    extracted = trafilatura.extract(
        html, output_format="markdown", include_tables=True, favor_recall=True
    )
    return extracted or ""
