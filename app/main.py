"""FastAPI application (Layer 02 - request handling).

Wiring lives here and nowhere else: the lifespan builds the store, embedders,
gateway and pipeline once and hangs them on app.state, so request handlers do
no construction and tests can substitute any piece.

CORS is deliberately permissive at the middleware layer and strict at the
dependency layer. The widget is embedded on client domains we do not enumerate
in config, so the real check is the Origin-to-Property lookup in deps.py, which
runs against the database on every request and returns 403 for anything
unregistered.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from app.api import routes_admin, routes_chat
from app.api.deps import build_gateway
from app.config import get_settings
from app.gateway.budget import get_usage_limiter
from app.ingestion.pipeline import IngestionPipeline
from app.logging_setup import configure_logging, get_logger
from app.observability.logfire_setup import instrument_fastapi
from app.pipeline.answer import AnswerPipeline
from app.retrieval.embeddings import get_dense_embedder, get_sparse_encoder
from app.retrieval.retriever import Retriever
from app.retrieval.store import VectorStore
from app.storage.db import close_db, init_db
from app.storage.properties import PropertyRepository

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level, service="guide-api")

    db = await init_db(settings.database_path)
    repo = PropertyRepository(db)
    # Counters live in the database so a restart cannot reset the day's budget.
    get_usage_limiter(settings).attach(db)

    store = VectorStore(settings)
    dense = get_dense_embedder(settings)
    sparse = get_sparse_encoder(settings)
    await store.ensure_collection(dense.dim)

    gateway = build_gateway(settings, repo)
    gateway.preflight()
    retriever = Retriever(store, dense, sparse, top_k=settings.retrieve_top_k)

    app.state.settings = settings
    app.state.store = store
    app.state.repo = repo
    app.state.gateway = gateway
    app.state.pipeline = AnswerPipeline(
        gateway=gateway,
        retriever=retriever,
        rerank_top_n=settings.rerank_top_n,
        min_rerank_score=settings.min_rerank_score,
        min_rerank_relevance=settings.min_rerank_relevance,
    )
    app.state.ingestion = IngestionPipeline(
        settings=settings,
        store=store,
        dense=dense,
        sparse=sparse,
        db=db,
        gateway=gateway,
    )

    log.info(
        f"Ready. Answering with {settings.answer_model}, "
        f"verifying with {settings.verifier_model}.",
        answer_model=settings.answer_model,
        verifier_model=settings.verifier_model,
        dense_model=settings.dense_model,
        dense_dim=settings.dense_dim,
        run_spend_cap_usd=settings.run_spend_cap_usd or "unlimited",
        account_spend_cap_usd=settings.account_daily_spend_cap_usd or "unlimited",
        account_call_cap=settings.account_daily_call_cap or "unlimited",
    )
    try:
        yield
    finally:
        await store.close()
        await close_db()


def create_app() -> FastAPI:
    # Before the app object exists: Logfire instruments FastAPI by adding
    # middleware, and Starlette refuses new middleware once the app has
    # started - which the lifespan already has.
    configure_logging(get_settings().log_level, service="guide-api")

    app = FastAPI(
        title="Hotel AI Guide",
        version="0.1.0",
        description=(
            "Grounded informational assistant embedded on hotel and homestay "
            "websites. Answers only from a property's own published content."
        ),
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"https?://.*",
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    app.include_router(routes_chat.router)
    app.include_router(routes_admin.router)
    instrument_fastapi(app)

    @app.get("/health", tags=["ops"])
    async def health() -> JSONResponse:
        return JSONResponse({"status": "ok"})

    widget_file = Path(__file__).resolve().parent.parent / "widget" / "src" / "guide.js"

    @app.get("/guide.js", tags=["widget"], include_in_schema=False)
    async def guide_js() -> FileResponse:
        """Serve the widget from the same origin as the API.

        One <script> tag per client site is the whole integration story, so the
        bundle ships from here rather than requiring a CDN step.
        """
        return FileResponse(
            widget_file,
            media_type="application/javascript",
            headers={"Cache-Control": "public, max-age=300"},
        )

    return app


app = create_app()
