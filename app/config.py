"""Application settings, loaded from environment / .env.

Every tunable in the pipeline is surfaced here so that ingestion, retrieval and
the gateway never read os.environ directly.
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ---- Providers (Layer 06) -------------------------------------------
    # Every chat task runs on OpenAI. The gateway is still provider-agnostic,
    # so pointing a single task at another vendor stays a one-line env change -
    # but nothing requires an Anthropic key to boot, serve or evaluate.
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None  # optional; unused by the defaults below
    google_api_key: str | None = None
    cohere_api_key: str | None = None

    answer_model: str = "gpt-4o-mini"
    # Verification is blocking, so it stays a tier above the cheap paths only
    # where it pays for itself: mini is enough for a claim-by-claim check
    # against supplied sources, which is extraction, not reasoning.
    verifier_model: str = "gpt-4o-mini"
    rewrite_model: str = "gpt-4o-mini"
    # A dedicated reranker, not a chat model: one call, no prompt, ~100ms, and
    # a relevance score instead of a rubric an LLM has to be talked through.
    rerank_model: str = "rerank-v3.5"
    eval_model: str = "gpt-4o"
    # Applies to the answer only. Every other task runs at 0 - see
    # LLMGateway.temperature_for. Low rather than zero: the answer is prose
    # read by a guest, and it still cannot leave its sources.
    answer_temperature: float = 0.2

    # ---- Embeddings -----------------------------------------------------
    # Changing either of these invalidates every indexed corpus: the dimension
    # is fixed at Qdrant collection creation and the vector space is not
    # comparable across models.
    dense_model: str = "gemini-embedding-2"
    dense_dim: int = 1536
    sparse_model: str = "Qdrant/bm25"

    # ---- Vector store ---------------------------------------------------
    # An http(s) URL is a Qdrant server. Anything else is an embedded store:
    # a directory path, or ":memory:". See app/retrieval/store.py.
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "hotel_chunks"

    # ---- Relational storage ---------------------------------------------
    database_path: str = "data/guide.db"

    # ---- Chat log (Layer 09) --------------------------------------------
    # One row per served turn: what was asked, what was answered, why it
    # deflected. Off makes the service keep nothing about a Visitor at all,
    # which is the right setting for a client who asks for exactly that.
    # Retention is a ceiling, not a promise of freshness: rows are swept at
    # startup, so a service that never restarts never sweeps. 0 keeps forever.
    chat_log_enabled: bool = True
    chat_log_retention_days: int = 90

    # ---- Ingestion ------------------------------------------------------
    # Uploads are the only way content enters a Corpus.
    upload_dir: str = "data/uploads"
    max_upload_mb: int = 25

    # ---- Chunking + retrieval -------------------------------------------
    chunk_target_tokens: int = 450
    chunk_overlap_tokens: int = 80
    retrieve_top_k: int = 40
    rerank_top_n: int = 8
    # Below the threshold nothing is considered relevant and the Guide deflects
    # instead of answering. Two of them, because the two rerankers do not speak
    # the same units and silently reading one as the other would either deflect
    # everything or nothing:
    #   min_rerank_score      0-10 rubric, when RERANK_MODEL is a chat model
    #   min_rerank_relevance  0-1 relevance, when it is a Cohere reranker
    # Cohere's scores are not calibrated across queries, so treat 0.2 as a
    # starting point to tune against the eval set, not a validated value.
    min_rerank_score: int = 4
    min_rerank_relevance: float = 0.2

    # ---- Serving --------------------------------------------------------
    log_level: str = "INFO"
    # console: one short human line per event. json: the same fields as JSON,
    # for a log shipper. Neither affects what is sent to Logfire.
    log_format: Literal["console", "json"] = "console"

    # ---- Logfire (Layer 08) ---------------------------------------------
    # A write token turns export on; without one every span and log still
    # happens locally and nothing leaves the machine. Create one with
    # `logfire projects new` after `logfire auth`.
    logfire_token: str | None = None
    # Keeps a laptop run out of the same view as production traffic.
    logfire_environment: str = "dev"
    rate_limit_per_minute: int = 12
    rate_limit_burst: int = 5
    default_daily_spend_cap_usd: float = 5.0

    # Ceilings under *every* model call - chat, ingestion, eval, embeddings.
    # The per-property cap above bills a property; ingestion and evals bill
    # none, so these are what bound a testing session. Any at 0 = unlimited.
    #
    # The run cap is scoped to one process and starts at zero every time, so a
    # single `guide eval` or server session cannot spend more than this no
    # matter what the day's ledger says. The daily ones are cumulative across
    # processes and persisted.
    run_spend_cap_usd: float = 0.15
    account_daily_spend_cap_usd: float = 0.50
    account_daily_call_cap: int = 2000
    # Retries inside the provider SDK, on top of the first attempt. Each one
    # is a real billed call on a timeout or a 5xx, so this is a spending knob
    # as much as a reliability one: 3 retries can quadruple the cost of a bad
    # minute. One is enough to ride out a blip without funding a storm.
    provider_max_retries: int = 1
    admin_api_key: str = "change-me-before-deploying"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
