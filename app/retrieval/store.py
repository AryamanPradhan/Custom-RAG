"""Layer 04 - Vector Store (Qdrant, hybrid).

One collection, two named vectors per point, server-side RRF fusion. Multi-
tenancy is a mandatory payload filter on property_id, indexed as a tenant key
so Qdrant co-locates each property's points on disk.

QDRANT_URL selects the backend. An http(s) URL talks to a server - Docker
locally, or a managed cluster. Anything else is read as an embedded store: a
directory path, or ":memory:" for a scratch one that dies with the process.
Embedded runs the same query path in-process with no server at all, which is
what makes a first upload testable before any infrastructure exists.

Tenant isolation is enforced here rather than by the caller: search() takes
property_id as a required positional argument, so there is no code path that
can accidentally query across properties.
"""

from __future__ import annotations

import uuid

from qdrant_client import AsyncQdrantClient, models

from app.config import Settings, get_settings
from app.logging_setup import get_logger
from app.models.domain import Chunk, DocCategory, ScoredChunk, SourceKind

log = get_logger(__name__)

DENSE = "dense"
SPARSE = "sparse"

_NAMESPACE = uuid.UUID("6f1b0b1e-9f0f-4f7a-9a2f-2c9a0f3e5d11")


def point_id(chunk_id: str) -> str:
    """Qdrant point IDs must be UUIDs or unsigned ints; our chunk ids are hex
    strings. A deterministic uuid5 keeps upserts idempotent across re-uploads."""
    return str(uuid.uuid5(_NAMESPACE, chunk_id))


class VectorStore:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        target = self.settings.qdrant_url
        self.embedded = not target.startswith(("http://", "https://"))
        if self.embedded:
            # One process only - the store takes a lock on the directory - and
            # payload indexes are ignored, so it is for development, not for
            # serving. It is a real Qdrant otherwise: same client, same query
            # path, same fusion.
            self._client = AsyncQdrantClient(
                path=None if target == ":memory:" else target,
                location=":memory:" if target == ":memory:" else None,
            )
            log.warning(
                f"Embedded Qdrant at {target} - single process, no payload indexes.",
                location=target,
            )
        else:
            self._client = AsyncQdrantClient(
                url=target,
                api_key=self.settings.qdrant_api_key or None,
                timeout=30,
            )
        self.collection = self.settings.qdrant_collection

    @property
    def client(self) -> AsyncQdrantClient:
        return self._client

    async def close(self) -> None:
        await self._client.close()

    # -- schema ----------------------------------------------------------

    async def ensure_collection(self, dense_dim: int) -> None:
        exists = await self._client.collection_exists(self.collection)
        if not exists:
            await self._client.create_collection(
                collection_name=self.collection,
                vectors_config={
                    DENSE: models.VectorParams(
                        size=dense_dim, distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)
                },
            )
            log.info(
                f"Created collection {self.collection} ({dense_dim}-dim).",
                collection=self.collection,
                dim=dense_dim,
            )

        # Payload indexes. property_id is declared as a tenant key so Qdrant
        # groups each tenant's vectors together - a large win once you host
        # more than a handful of properties.
        # Index creation is not idempotent across all Qdrant versions, and
        # ensure_collection() runs on every boot - so tolerate "already exists"
        # rather than crash-looping the service on restart.
        if self.embedded:
            # Embedded Qdrant ignores payload indexes and warns about each one.
            # Nothing to create, so do not ask.
            return

        async def _index(field: str, schema) -> None:
            try:
                await self._client.create_payload_index(
                    self.collection, field_name=field, field_schema=schema
                )
            except Exception as exc:  # noqa: BLE001 - vendor-specific error types
                log.debug("store.index_exists", field=field, detail=str(exc))

        await _index(
            "property_id",
            models.KeywordIndexParams(
                type=models.KeywordIndexType.KEYWORD, is_tenant=True
            ),
        )
        for field in ("category", "source_kind", "doc_id", "unit"):
            await _index(field, models.PayloadSchemaType.KEYWORD)

    # -- writes ----------------------------------------------------------

    async def upsert(
        self,
        chunks: list[Chunk],
        dense_vectors: list[list[float]],
        sparse_vectors: list[tuple[list[int], list[float]]],
    ) -> int:
        if not chunks:
            return 0
        points = [
            models.PointStruct(
                id=point_id(chunk.chunk_id),
                vector={
                    DENSE: dense,
                    SPARSE: models.SparseVector(indices=sparse[0], values=sparse[1]),
                },
                payload=chunk.to_payload(),
            )
            for chunk, dense, sparse in zip(chunks, dense_vectors, sparse_vectors, strict=True)
        ]
        await self._client.upsert(self.collection, points=points, wait=True)
        return len(points)

    async def delete_document(self, property_id: str, doc_id: str) -> None:
        """Remove a document's chunks - used when a document is replaced or its
        content changed and produced fewer chunks than the previous run."""
        await self._client.delete(
            self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="property_id", match=models.MatchValue(value=property_id)
                        ),
                        models.FieldCondition(
                            key="doc_id", match=models.MatchValue(value=doc_id)
                        ),
                    ]
                )
            ),
            wait=True,
        )

    async def delete_property(self, property_id: str) -> None:
        """Layer 09 - retention / right-to-erasure for one tenant."""
        await self._client.delete(
            self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="property_id", match=models.MatchValue(value=property_id)
                        )
                    ]
                )
            ),
            wait=True,
        )

    # -- reads -----------------------------------------------------------

    def _filter(self, property_id: str) -> models.Filter:
        """Tenant isolation, and nothing else.

        Category once narrowed this too. It no longer does: a filter chosen
        before retrieval can only remove Chunks, and the ones it removed were
        sometimes the answer. Relevance is decided by the reranker, which has
        read the candidates. See ADR 0005.
        """
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="property_id", match=models.MatchValue(value=property_id)
                )
            ]
        )

    async def hybrid_search(
        self,
        property_id: str,
        *,
        dense_vector: list[float],
        sparse_vector: tuple[list[int], list[float]],
        limit: int = 40,
        prefetch_multiplier: int = 3,
    ) -> list[ScoredChunk]:
        """Dense + BM25 in one round trip, fused server-side with RRF.

        RRF fuses by rank rather than score, which is what makes it safe to
        combine a cosine similarity with a BM25 score - two quantities that are
        not on remotely the same scale.
        """
        query_filter = self._filter(property_id)
        prefetch_limit = limit * prefetch_multiplier

        response = await self._client.query_points(
            collection_name=self.collection,
            prefetch=[
                models.Prefetch(
                    query=dense_vector,
                    using=DENSE,
                    limit=prefetch_limit,
                    filter=query_filter,
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_vector[0], values=sparse_vector[1]
                    ),
                    using=SPARSE,
                    limit=prefetch_limit,
                    filter=query_filter,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        return [_to_scored(p) for p in response.points]

    async def count(self, property_id: str) -> int:
        result = await self._client.count(
            self.collection,
            count_filter=self._filter(property_id),
            exact=True,
        )
        return result.count

    async def doc_hashes(self, property_id: str) -> dict[str, str]:
        """doc_id -> content_hash for every indexed doc of this property, so
        the ingestion pipeline can skip pages that have not changed."""
        hashes: dict[str, str] = {}
        offset = None
        while True:
            points, offset = await self._client.scroll(
                self.collection,
                scroll_filter=self._filter(property_id),
                limit=512,
                offset=offset,
                with_payload=["doc_id", "content_hash"],
                with_vectors=False,
            )
            for p in points:
                payload = p.payload or {}
                if payload.get("doc_id") and payload.get("content_hash"):
                    hashes[payload["doc_id"]] = payload["content_hash"]
            if offset is None:
                break
        return hashes


def _to_scored(point) -> ScoredChunk:
    payload = point.payload or {}
    chunk = Chunk(
        chunk_id=payload.get("chunk_id", str(point.id)),
        doc_id=payload.get("doc_id", ""),
        property_id=payload.get("property_id", ""),
        text=payload.get("text", ""),
        uri=payload.get("uri", ""),
        title=payload.get("title", ""),
        heading_path=payload.get("heading_path", []) or [],
        category=DocCategory(payload.get("category", "other")),
        source_kind=SourceKind(payload.get("source_kind", "upload")),
        position=payload.get("position", 0),
        token_estimate=len(payload.get("text", "")) // 4,
        unit=payload.get("unit"),
        fetched_at=payload.get("fetched_at", ""),
        metadata={
            k: v
            for k, v in payload.items()
            if k
            not in {
                "chunk_id", "doc_id", "property_id", "text", "uri", "title",
                "heading_path", "category", "source_kind", "position",
                "unit", "fetched_at",
            }
        },
    )
    return ScoredChunk(chunk=chunk, score=float(point.score))


_store: VectorStore | None = None


def get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store
