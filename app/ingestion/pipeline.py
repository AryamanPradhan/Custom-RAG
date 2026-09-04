"""Ingestion orchestration: Sources in, Chunks indexed.

    crawl / load -> classify -> chunk -> embed -> upsert -> record state

Content hashes are the reason a re-crawl is cheap: a page whose text has not
changed since the last run is skipped before it reaches the embedding API,
which is where the money is. On a typical property site a re-crawl after a
small edit touches two or three pages out of eighty.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from app.config import Settings
from app.gateway.llm_gateway import LLMGateway
from app.ingestion.chunker import chunk_document, embedding_text
from app.ingestion.classify import classify_document
from app.ingestion.crawler import SiteCrawler
from app.ingestion.drift import record_source_state
from app.ingestion.loaders import load_bytes
from app.logging_setup import get_logger
from app.models.domain import Document
from app.observability.metrics import METRICS
from app.retrieval.embeddings import GeminiEmbedder, SparseEncoder
from app.retrieval.store import VectorStore
from app.storage.db import Database

log = get_logger(__name__)

_EMBED_BATCH = 64


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

    async def ingest_site(
        self,
        property_id: str,
        start_url: str,
        *,
        max_pages: int | None = None,
        max_depth: int | None = None,
        include_paths: list[str] | None = None,
        exclude_paths: list[str] | None = None,
    ) -> IngestReport:
        started = time.perf_counter()
        crawler = SiteCrawler(self._settings, property_id)
        result = await crawler.crawl(
            start_url,
            max_pages=max_pages,
            max_depth=max_depth,
            include_paths=include_paths,
            exclude_paths=exclude_paths,
        )
        report = await self.ingest_documents(property_id, result.documents)
        report.errors.extend(result.errors)
        report.duration_seconds = round(time.perf_counter() - started, 2)
        return report

    async def ingest_upload(
        self, property_id: str, filename: str, data: bytes
    ) -> IngestReport:
        started = time.perf_counter()
        report = IngestReport(property_id=property_id)
        try:
            doc = load_bytes(property_id, filename, data)
        except ValueError as exc:
            report.errors.append(str(exc))
            report.duration_seconds = round(time.perf_counter() - started, 2)
            return report

        report = await self.ingest_documents(property_id, [doc])
        report.duration_seconds = round(time.perf_counter() - started, 2)
        return report

    async def ingest_documents(
        self, property_id: str, documents: list[Document]
    ) -> IngestReport:
        report = IngestReport(property_id=property_id)
        if not documents:
            return report

        await self._store.ensure_collection(self._dense.dim)
        known = await self._store.doc_hashes(property_id)

        # Chunk first, embed second, and only then touch the index. Deleting a
        # document's chunks before its replacements are embedded would leave the
        # page unretrievable whenever the embedding call fails - and recording
        # source state up front would make the drift detector report that same
        # page as happily indexed.
        prepared: list[tuple[Document, list]] = []
        for doc in documents:
            if known.get(doc.doc_id) == doc.content_hash:
                report.skipped_unchanged += 1
                continue
            try:
                category, unit = await classify_document(doc, self._gateway)
                doc.category = category
                doc.unit = unit

                chunks = chunk_document(
                    doc,
                    target_tokens=self._settings.chunk_target_tokens,
                    overlap_tokens=self._settings.chunk_overlap_tokens,
                )
                if chunks:
                    prepared.append((doc, chunks))
            except Exception as exc:  # noqa: BLE001 - one bad page must not stop the run
                report.errors.append(f"{doc.uri}: {type(exc).__name__}: {exc}")
                log.warning(
                    "ingest.document_failed",
                    uri=doc.uri,
                    error=f"{type(exc).__name__}: {exc}",
                )

        for doc, chunks in prepared:
            try:
                vectors: list[list[float]] = []
                sparse: list[tuple[list[int], list[float]]] = []
                for i in range(0, len(chunks), _EMBED_BATCH):
                    batch = chunks[i : i + _EMBED_BATCH]
                    texts = [embedding_text(c) for c in batch]
                    vectors.extend(await self._dense.embed_documents(texts))
                    sparse.extend(await self._sparse.encode_documents(texts))

                # Replace rather than merge: a shortened page would otherwise
                # leave its removed chunks in the index forever, still
                # retrievable and still citable.
                await self._store.delete_document(property_id, doc.doc_id)
                report.chunks += await self._store.upsert(chunks, vectors, sparse)
                report.documents += 1

                # Only now is the claim "this page is indexed at this hash" true.
                await record_source_state(
                    self._db,
                    property_id,
                    doc.uri,
                    doc.content_hash,
                    etag=doc.metadata.get("etag"),
                    last_modified=doc.metadata.get("last_modified"),
                )
            except Exception as exc:  # noqa: BLE001
                report.errors.append(f"{doc.uri}: {type(exc).__name__}: {exc}")
                log.error(
                    "ingest.index_failed",
                    uri=doc.uri,
                    error=f"{type(exc).__name__}: {exc}",
                )

        METRICS.add("ingest.chunks", report.chunks, property_id=property_id)
        log.info(
            "ingest.done",
            property_id=property_id,
            documents=report.documents,
            chunks=report.chunks,
            skipped=report.skipped_unchanged,
            errors=len(report.errors),
        )
        return report
