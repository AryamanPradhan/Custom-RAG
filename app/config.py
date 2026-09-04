"""Application settings, loaded from environment / .env.

Every tunable in the pipeline is surfaced here so that ingestion, retrieval and
the gateway never read os.environ directly.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ---- Providers (Layer 06) -------------------------------------------
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    google_api_key: str | None = None

    answer_model: str = "gpt-4o-mini"
    verifier_model: str = "claude-haiku-4-5"
    rewrite_model: str = "claude-haiku-4-5"
    rerank_model: str = "claude-sonnet-5"
    eval_model: str = "claude-sonnet-5"

    # ---- Embeddings -----------------------------------------------------
    # Changing either of these invalidates every indexed corpus: the dimension
    # is fixed at Qdrant collection creation and the vector space is not
    # comparable across models.
    dense_model: str = "gemini-embedding-2"
    dense_dim: int = 1536
    sparse_model: str = "Qdrant/bm25"

    # ---- Vector store ---------------------------------------------------
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "hotel_chunks"

    # ---- Relational storage ---------------------------------------------
    database_path: str = "data/guide.db"

    # ---- Ingestion ------------------------------------------------------
    crawl_max_pages: int = 300
    crawl_max_depth: int = 4
    crawl_concurrency: int = 6
    crawl_delay_seconds: float = 0.5
    crawl_respect_robots: bool = True
    crawl_user_agent: str = "hotel-ai-guide/0.1"
    upload_dir: str = "data/uploads"
    max_upload_mb: int = 25

    # ---- Chunking + retrieval -------------------------------------------
    chunk_target_tokens: int = 450
    chunk_overlap_tokens: int = 80
    retrieve_top_k: int = 40
    rerank_top_n: int = 8
    # Below this rerank score nothing is considered relevant and the Guide
    # deflects instead of answering.
    min_rerank_score: int = 4

    # ---- Serving --------------------------------------------------------
    log_level: str = "INFO"
    rate_limit_per_minute: int = 12
    rate_limit_burst: int = 5
    default_daily_spend_cap_usd: float = 5.0
    admin_api_key: str = "change-me-before-deploying"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
