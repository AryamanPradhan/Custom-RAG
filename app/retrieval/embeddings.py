"""Layer 04 - Embeddings.

Two vector families, because hybrid retrieval needs both:

  dense  - semantic. Catches "can I bring my dog" -> "pet policy".
  sparse - lexical BM25. Catches "Wi-Fi password", "check-out at 11", room
           names and other rare tokens that dense models blur away.

Property content is dense with proper nouns - room names, restaurants, local
landmarks - that dense-only retrieval reliably fumbles, which is why the sparse
half is not optional here.

Dense vectors come from Gemini; sparse vectors are computed locally, so the
lexical half of retrieval costs nothing and works offline.
"""

from __future__ import annotations

import asyncio
from typing import Any

from app.config import Settings, get_settings
from app.gateway.budget import get_usage_limiter
from app.gateway.pricing import cost_usd
from app.logging_setup import get_logger

log = get_logger(__name__)

# Conservative batch size: the embed endpoint limits both request count and
# total tokens per call, and our chunks run ~450 tokens each.
_BATCH = 32

# gemini-embedding-001 takes an explicit task_type and needs manual L2
# normalisation below 3072 dims. gemini-embedding-2 drops task_type and
# auto-normalises truncated dimensions.
_LEGACY_MODEL = "gemini-embedding-001"


class GeminiEmbedder:
    def __init__(self, settings: Settings) -> None:
        from google import genai

        if not settings.google_api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY is unset - the dense half of retrieval cannot run. "
                "Set it in .env, or the Guide can only do lexical search."
            )
        self._client = genai.Client(api_key=settings.google_api_key)
        self._model = settings.dense_model
        self.dim = settings.dense_dim
        self._legacy = self._model == _LEGACY_MODEL

    def _config(self, task_type: str):
        from google.genai import types

        kwargs: dict[str, Any] = {"output_dimensionality": self.dim}
        if self._legacy:
            kwargs["task_type"] = task_type
        return types.EmbedContentConfig(**kwargs)

    @staticmethod
    def _as_contents(texts: list[str]) -> list[Any]:
        """One Content per text - the difference between 32 vectors and 1.

        google-genai coerces a plain `list[str]` into a *single* Content with
        one part per string, so the API returns one embedding for the whole
        batch. Wrapping each text in its own Content is what makes it a batch.
        """
        from google.genai import types

        return [types.Content(parts=[types.Part(text=t)]) for t in texts]

    async def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        out: list[list[float]] = []
        limiter = get_usage_limiter()
        for i in range(0, len(texts), _BATCH):
            batch = texts[i : i + _BATCH]
            # Embeddings are cheap per call and ruinous in bulk: one careless
            # re-index of every property is thousands of calls that no
            # per-property cap sees. Checked per batch, not per run.
            await limiter.check("embed")
            result = await self._client.aio.models.embed_content(
                model=self._model,
                contents=self._as_contents(batch),
                config=self._config(task_type),
            )
            returned = result.embeddings or []
            if len(returned) != len(batch):
                raise RuntimeError(
                    f"embedding provider returned {len(returned)} vectors for "
                    f"{len(batch)} inputs - refusing to misalign chunks and vectors"
                )
            vectors = [list(e.values or []) for e in returned]
            if self._legacy and self.dim != 3072:
                vectors = [_normalise(v) for v in vectors]
            out.extend(vectors)

            # The embed endpoint reports no usage, so bill the same ~4
            # chars-per-token estimate the gateway uses for abandoned streams.
            tokens = sum(len(t) for t in batch) // 4
            await limiter.record(cost_usd(self._model, input_tokens=tokens))
        return out

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._embed(texts, "RETRIEVAL_DOCUMENT")

    async def embed_query(self, text: str) -> list[float]:
        return (await self._embed([text], "RETRIEVAL_QUERY"))[0]


def _normalise(vector: list[float]) -> list[float]:
    """L2-normalise to unit length.

    Only needed for gemini-embedding-001 truncated below 3072: cosine distance
    in Qdrant assumes unit vectors, and Matryoshka truncation breaks that.
    """
    magnitude = sum(v * v for v in vector) ** 0.5
    if magnitude == 0:
        return vector
    return [v / magnitude for v in vector]


class SparseEncoder:
    """BM25 sparse vectors via fastembed. Qdrant applies the IDF modifier
    server-side, so what we send is term frequencies, not final BM25 weights."""

    def __init__(self, settings: Settings) -> None:
        from fastembed import SparseTextEmbedding

        self._model = SparseTextEmbedding(model_name=settings.sparse_model)

    def _sync_docs(self, texts: list[str]) -> list[tuple[list[int], list[float]]]:
        return [
            (emb.indices.tolist(), emb.values.tolist()) for emb in self._model.embed(texts)
        ]

    def _sync_query(self, text: str) -> tuple[list[int], list[float]]:
        emb = next(iter(self._model.query_embed(text)))
        return emb.indices.tolist(), emb.values.tolist()

    async def encode_documents(self, texts: list[str]) -> list[tuple[list[int], list[float]]]:
        # fastembed is synchronous CPU work - keep it off the event loop.
        return await asyncio.to_thread(self._sync_docs, texts)

    async def encode_query(self, text: str) -> tuple[list[int], list[float]]:
        return await asyncio.to_thread(self._sync_query, text)


_dense: GeminiEmbedder | None = None
_sparse: SparseEncoder | None = None


def get_dense_embedder(settings: Settings | None = None) -> GeminiEmbedder:
    global _dense
    if _dense is None:
        settings = settings or get_settings()
        _dense = GeminiEmbedder(settings)
        log.info(
            f"Gemini embeddings ready ({settings.dense_model}, {_dense.dim}-dim).",
            model=settings.dense_model,
            dim=_dense.dim,
        )
    return _dense


def get_sparse_encoder(settings: Settings | None = None) -> SparseEncoder:
    global _sparse
    if _sparse is None:
        _sparse = SparseEncoder(settings or get_settings())
    return _sparse
