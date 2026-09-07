"""Ingestion orchestration: Sources in, Chunks indexed.

    load -> classify -> chunk -> embed -> upsert -> record state

Everything indexed here was handed over by the property owner - an uploaded
document or pasted text. Nothing is fetched from the web.

Content hashes are what make re-sending a document cheap: an upload whose text
has not changed since the last run is skipped before it reaches the embedding
API, which is where the money is. Owners routinely re-send a whole folder to
change one page in one file.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from app.config import Settings
from app.gateway.llm_gateway import LLMGateway
from app.ingestion.chunker import chunk_document, embedding_text
from app.ingestion.classify import classify_document
from app.ingestion.loaders import load_bytes
from app.ingestion.source_state import record_source_state
from app.logging_setup import get_logger
from app.models.domain import Document
from app.observability.metrics import METRICS
from app.observability.tracing import span
from app.retrieval.embeddings import GeminiEmbedder, SparseEncoder
from app.retrieval.store import VectorStore
from app.storage.db import Database

log = get_logger(__name__)

_EMBED_BATCH = 64


def _label(doc: Document) -> str:
    """What to call this document in a log line. `upload://rates.pdf` is noise."""
    return doc.uri.removeprefix("upload://")


@dataclass
class IngestReport:
    property_id: str
    documents: int = 0
    chunks: int = 0
    skipped_unchanged: int = 0
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0


class IngestionPipeline:
    def __init__(
        self,
        *,
        settings: Settings,
        store: VectorStore,
        dense: GeminiEmbedder,
        sparse: SparseEncoder,
        db: Database,
        gateway: LLMGateway | None = None,
    ) -> None:
        self._settings = settings
        self._store = store
        self._dense = dense
        self._sparse = sparse
        self._db = db
        # Optional: without it, classification falls back to URL/title
        # heuristics and no unit detection. Ingestion still works.
        self._gateway = gateway

    async def ingest_upload(
        self, property_id: str, filename: str, data: bytes
    ) -> IngestReport:
        started = time.perf_counter()
        report = IngestReport(property_id=property_id)
        try:
            with span("load", file=filename, bytes=len(data)) as s:
                doc = load_bytes(property_id, filename, data)
                s.attributes["chars"] = len(doc.text)
                s.summary = f"Read {filename}: {len(doc.text):,} characters of text."
        except ValueError as exc:
            report.errors.append(str(exc))
            report.duration_seconds = round(time.perf_counter() - started, 2)
            log.warning(f"Could not read {filename}.", file=filename, error=str(exc))
            return report

        report = await self.ingest_documents(property_id, [doc])
        report.duration_seconds = round(time.perf_counter() - started, 2)
        return report

    async def ingest_documents(
        self, property_id: str, documents: list[Document]
    ) -> IngestReport:
        """One span for the run, with a span per document step inside it.

        The wrapper exists so that every step below shares a single parent:
        an ingest reads as one nested trace rather than a dozen unrelated ones.
        """
        with span("ingest", property_id=property_id, documents=len(documents)) as s:
            report = await self._ingest(property_id, documents)
            s.attributes.update(
                chunks=report.chunks,
                skipped=report.skipped_unchanged,
                errors=len(report.errors),
            )
            s.summary = (
                f"Indexed {report.documents} documents as {report.chunks} chunks"
                f" ({report.skipped_unchanged} unchanged, {len(report.errors)} failed)."
            )
            return report

    async def _ingest(
        self, property_id: str, documents: list[Document]
    ) -> IngestReport:
        report = IngestReport(property_id=property_id)
        if not documents:
            return report

        started = time.perf_counter()
        log.info(
            f"Ingesting {len(documents)} documents for {property_id}.",
            property_id=property_id,
            documents=len(documents),
        )

        with span("prepare_index") as s:
            await self._store.ensure_collection(self._dense.dim)
            known = await self._store.doc_hashes(property_id)
            s.attributes["already_indexed"] = len(known)
            s.summary = (
                f"Collection {self._settings.qdrant_collection} ready; "
                f"{len(known)} documents already indexed."
            )

        # Chunk first, embed second, and only then touch the index. Deleting a
        # document's chunks before its replacements are embedded would leave the
        # document unretrievable whenever the embedding call fails - and
        # recording source state up front would claim it was indexed anyway.
        prepared: list[tuple[Document, list]] = []
        for doc in documents:
            if known.get(doc.doc_id) == doc.content_hash:
                report.skipped_unchanged += 1
                continue
            try:
                with span("classify", file=_label(doc)) as s:
                    category, unit = await classify_document(doc, self._gateway)
                    doc.category = category
                    doc.unit = unit
                    s.attributes.update(category=category.value, unit=unit or "-")
                    s.summary = f"Classified {_label(doc)} as {category.value}."

                with span("chunk", file=_label(doc)) as s:
                    chunks = chunk_document(
                        doc,
                        target_tokens=self._settings.chunk_target_tokens,
                        overlap_tokens=self._settings.chunk_overlap_tokens,
                    )
                    s.attributes["chunks"] = len(chunks)
                    s.summary = f"Split {_label(doc)} into {len(chunks)} chunks."
                if chunks:
                    prepared.append((doc, chunks))
                else:
                    log.warning(f"{_label(doc)} produced no chunks; skipping.",
                                file=_label(doc))
            except Exception as exc:  # noqa: BLE001 - one bad file must not stop the run
                report.errors.append(f"{doc.uri}: {type(exc).__name__}: {exc}")
                log.warning(
                    f"Could not prepare {_label(doc)}; the rest of the run continues.",
                    uri=doc.uri,
                    error=f"{type(exc).__name__}: {exc}",
                )

        for doc, chunks in prepared:
            try:
                with span("embed", file=_label(doc), chunks=len(chunks)) as s:
                    s.summary = f"Embedded {len(chunks)} chunks of {_label(doc)}."
                    vectors: list[list[float]] = []
                    sparse: list[tuple[list[int], list[float]]] = []
                    for i in range(0, len(chunks), _EMBED_BATCH):
                        batch = chunks[i : i + _EMBED_BATCH]
                        texts = [embedding_text(c) for c in batch]
                        vectors.extend(await self._dense.embed_documents(texts))
                        sparse.extend(await self._sparse.encode_documents(texts))

                with span("index", file=_label(doc)) as s:
                    # Replace rather than merge: a shortened page would
                    # otherwise leave its removed chunks in the index forever,
                    # still retrievable and still citable.
                    await self._store.delete_document(property_id, doc.doc_id)
                    indexed = await self._store.upsert(chunks, vectors, sparse)
                    s.attributes["chunks"] = indexed
                    s.summary = f"Indexed {indexed} chunks of {_label(doc)}."
                report.chunks += indexed
                report.documents += 1

                # Only now is the claim "this document is indexed at this
                # hash" true.
                await record_source_state(
                    self._db, property_id, doc.uri, doc.content_hash
                )

            except Exception as exc:  # noqa: BLE001
                report.errors.append(f"{doc.uri}: {type(exc).__name__}: {exc}")
                log.error(
                    f"Failed to index {_label(doc)}; it stays at its previous version.",
                    uri=doc.uri,
                    error=f"{type(exc).__name__}: {exc}",
                )

        METRICS.add("ingest.chunks", report.chunks, property_id=property_id)
        report.duration_seconds = round(time.perf_counter() - started, 2)
        return report
