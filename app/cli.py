"""Operator CLI.

Everything an operator does by hand: onboard a client, index the documents they
supplied, ask the Guide a question, run the eval set.

    guide onboard casa-verde "Casa Verde" --origin https://casaverde.com \\
          --phone "+44 1234 567890"
    guide upload casa-verde ./house-rules.pdf
    guide upload casa-verde ./data/demo-corpus        # a folder works too
    guide ask casa-verde "can I bring my dog?"
    guide eval casa-verde ./evals/casa-verde.json
    guide logs casa-verde --deflected        # what the corpus could not answer
    guide origins casa-verde --add http://localhost:8000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from app.config import get_settings
from app.evals.harness import load_cases, run_eval
from app.gateway.budget import get_usage_limiter
from app.gateway.llm_gateway import LLMGateway
from app.ingestion.loaders import SUPPORTED_SUFFIXES, load_bytes
from app.ingestion.pipeline import IngestionPipeline
from app.logging_setup import configure_logging
from app.observability.tracing import span
from app.pipeline.answer import AnswerPipeline
from app.retrieval.embeddings import get_dense_embedder, get_sparse_encoder
from app.retrieval.retriever import Retriever
from app.retrieval.store import VectorStore
from app.storage.chat_log import ChatLog
from app.storage.db import close_db, init_db
from app.storage.properties import ContactRoute, PropertyRepository, normalise_origin


class _Context:
    def __init__(self, settings, db, repo, store, dense, sparse, gateway) -> None:
        self.settings = settings
        self.db = db
        self.repo = repo
        self.store = store
        self.dense = dense
        self.sparse = sparse
        self.gateway = gateway

    def answer_pipeline(self, *, record_turns: bool = False) -> AnswerPipeline:
        """`record_turns` files each turn in the chat log.

        On for `ask`, which is a real question against a real Corpus, and off
        for `eval`, which would otherwise file a sweep of invented questions
        as though visitors had asked them.
        """
        chat_log = ChatLog(self.db)
        return AnswerPipeline(
            gateway=self.gateway,
            retriever=Retriever(
                self.store, self.dense, self.sparse, top_k=self.settings.retrieve_top_k
            ),
            rerank_top_n=self.settings.rerank_top_n,
            min_rerank_score=self.settings.min_rerank_score,
            min_rerank_relevance=self.settings.min_rerank_relevance,
            on_turn=(
                chat_log.record
                if record_turns and self.settings.chat_log_enabled
                else None
            ),
        )

    def ingestion(self) -> IngestionPipeline:
        return IngestionPipeline(
            settings=self.settings,
            store=self.store,
            dense=self.dense,
            sparse=self.sparse,
            db=self.db,
            gateway=self.gateway,
        )


async def _build(*, need_embeddings: bool = True, need_rerank: bool = True) -> _Context:
    settings = get_settings()
    configure_logging(settings.log_level, service="guide-cli")
    db = await init_db(settings.database_path)
    repo = PropertyRepository(db)
    # Shares the day's budget with the running service - a CLI eval sweep and
    # live traffic spend the same money.
    get_usage_limiter(settings).attach(db)

    async def record(property_id: str, amount: float) -> None:
        await repo.record_spend(property_id, amount)

    gateway = LLMGateway(settings, on_spend=record)
    if need_embeddings:
        # `onboard` touches no model, so it stays usable before any key is set,
        # and `upload` never reranks - each command checks only what it uses.
        gateway.preflight(needs_rerank=need_rerank)
    store = VectorStore(settings)
    dense = get_dense_embedder(settings) if need_embeddings else None
    sparse = get_sparse_encoder(settings) if need_embeddings else None
    return _Context(settings, db, repo, store, dense, sparse, gateway)


async def _onboard(args) -> int:
    ctx = await _build(need_embeddings=False)
    try:
        prop = await ctx.repo.create(
            args.property_id,
            args.display_name,
            allowed_origins=args.origin,
            contact_route=ContactRoute(
                phone=args.phone, email=args.email, url=args.url
            ),
            daily_spend_cap_usd=args.cap,
        )
        print(f"Registered {prop.property_id} ({prop.display_name})")
        print(f"  origins: {', '.join(prop.allowed_origins)}")
        print(f"  contact: {prop.contact_route.describe()}")
        print(f"  cap:     ${prop.daily_spend_cap_usd:.2f}/day")
        print(f"\nEmbed on the site:\n"
              f'  <script src="/guide.js" data-property-id="{prop.property_id}"></script>')
        return 0
    finally:
        await close_db()


def _collect(paths: list[str]) -> list[Path]:
    """Files to ingest, from a mix of file and directory arguments.

    Directories are walked rather than globbed by the caller, because the shell
    that expands `*.pdf` on one machine leaves it literal on another.
    """
    found: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            found += sorted(
                p for p in path.rglob("*")
                if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
            )
        else:
            found.append(path)
    return found


async def _upload(args) -> int:
    ctx = await _build(need_rerank=False)
    try:
        paths = _collect(args.paths)
        if not paths:
            print(f"Nothing to ingest under {', '.join(args.paths)}", file=sys.stderr)
            return 1

        # Loaded up front so the whole set is one ingest: one collection check,
        # one hash lookup, one report - and one process load of the BM25 model.
        documents = []
        failures = 0
        for path in paths:
            try:
                documents.append(load_bytes(args.property_id, path.name, path.read_bytes()))
            except (OSError, ValueError) as exc:
                failures += 1
                print(f"  ! {exc}", file=sys.stderr)

        report = await ctx.ingestion().ingest_documents(args.property_id, documents)
        await ctx.repo.mark_ingested(args.property_id)
        print(
            f"{report.documents} documents -> {report.chunks} chunks "
            f"({report.skipped_unchanged} unchanged, {report.duration_seconds}s)"
        )
        for error in report.errors:
            print(f"  ! {error}", file=sys.stderr)
        # A run where every document was already indexed is a success, not an
        # empty one - re-sending a folder to change one file is the normal case.
        indexed_or_current = report.chunks or report.skipped_unchanged
        return 0 if indexed_or_current and not failures and not report.errors else 1
    finally:
        await ctx.store.close()
        await close_db()


async def _ask(args) -> int:
    ctx = await _build()
    try:
        prop = await ctx.repo.get(args.property_id)
        if prop is None:
            print(f"No such property: {args.property_id}", file=sys.stderr)
            return 1
        # One parent span, so a question reads as a single trace: plan,
        # retrieve, rerank, generate, verify. Served requests get this from
        # the FastAPI instrumentation instead.
        with span("ask", question=args.question) as s:
            result = await ctx.answer_pipeline(record_turns=True).answer(
                prop, args.question
            )
            s.summary = (
                f"Deflected: {result.reason}."
                if result.deflected
                else f"Answered from {len(result.citations)} cited sources."
            )
        print(f"\n{result.answer}\n")
        for citation in result.citations:
            stamp = f" (as published {citation.published_on})" if citation.published_on else ""
            print(f"  [{citation.index}] {citation.label}{stamp}\n      {citation.uri}")
        if result.deflected:
            print(f"\n(deflected: {result.reason})", file=sys.stderr)
        return 0
    finally:
        await ctx.store.close()
        await close_db()


async def _eval(args) -> int:
    ctx = await _build()
    try:
        prop = await ctx.repo.get(args.property_id)
        if prop is None:
            print(f"No such property: {args.property_id}", file=sys.stderr)
            return 1
        cases = load_cases(args.cases)
        if not cases:
            print(f"No cases in {args.cases}", file=sys.stderr)
            return 1
        run = await run_eval(
            prop=prop,
            cases=cases,
            pipeline=ctx.answer_pipeline(),
            db=ctx.db,
            label=args.label or "",
        )
        print(json.dumps(run.summary(), indent=2))
        for result in run.results:
            if not result.passed:
                print(f"  FAIL {result.case_id}: {result.question}\n"
                      f"       {result.detail}", file=sys.stderr)
        return 0 if run.rate("passed") == 1.0 else 1
    finally:
        await ctx.store.close()
        await close_db()


async def _origins(args) -> int:
    """Show, add to or remove from a Property's origin allowlist.

    The allowlist is the only thing identifying which client an anonymous
    request belongs to, so it needed a way to change after onboarding - a
    client moving to a new domain, or a test page served from a new port.
    """
    ctx = await _build(need_embeddings=False)
    try:
        prop = await ctx.repo.get(args.property_id)
        if prop is None:
            print(f"No such property: {args.property_id}", file=sys.stderr)
            return 1

        origins = list(prop.allowed_origins)
        for raw in args.add or []:
            origin = normalise_origin(raw)
            if not origin:
                print(f"Not a usable origin: {raw}", file=sys.stderr)
                return 1
            if origin not in origins:
                origins.append(origin)
        for raw in args.remove or []:
            origin = normalise_origin(raw)
            if origin in origins:
                origins.remove(origin)

        if not origins:
            # Removing the last one would leave the Property unreachable by
            # every caller, which is a mistake rather than a configuration.
            print("A property needs at least one origin.", file=sys.stderr)
            return 1

        if origins != prop.allowed_origins:
            await ctx.repo.set_origins(args.property_id, origins)

        print(f"{prop.property_id} answers requests from:")
        for origin in origins:
            print(f"  {origin}")
        return 0
    finally:
        await ctx.store.close()
        await close_db()


async def _logs(args) -> int:
    """Read the chat log back. No models, no vector store - just the table."""
    ctx = await _build(need_embeddings=False)
    try:
        turns = await ChatLog(ctx.db).recent(
            args.property_id,
            limit=args.limit,
            session_id=args.session,
            deflected_only=args.deflected,
        )
        if not turns:
            print("No turns recorded yet.")
            return 0
        for turn in turns:
            mark = "✗" if turn["deflected"] else ("!" if turn["blocked"] else "✓")
            stamp = turn["created_at"][:19].replace("T", " ")
            print(f"\n{mark} {stamp}  {turn['mode']}  {turn['latency_ms']:.0f}ms")
            print(f"   Q: {turn['question']}")
            print(f"   A: {turn['answer'][:300]}")
            if turn["reason"]:
                print(f"   → {turn['reason']}")
            for citation in turn["citations"]:
                print(f"      [{citation['index']}] {citation['label']}")
        print(f"\n{len(turns)} turn{'' if len(turns) == 1 else 's'}.")
        return 0
    finally:
        # _build opens one either way, and an embedded Qdrant holds a
        # lock on its directory until it is closed.
        await ctx.store.close()
        await close_db()


def main() -> int:
    parser = argparse.ArgumentParser(prog="guide", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("onboard", help="register a property")
    p.add_argument("property_id")
    p.add_argument("display_name")
    p.add_argument("--origin", action="append", required=True,
                   help="allowed origin; repeat for more than one")
    p.add_argument("--phone")
    p.add_argument("--email")
    p.add_argument("--url")
    # No default here: None reaches the repository, which resolves it from
    # DEFAULT_DAILY_SPEND_CAP_USD like every other caller.
    p.add_argument(
        "--cap", type=float, default=None,
        help="daily spend cap in USD (default: DEFAULT_DAILY_SPEND_CAP_USD)",
    )
    p.set_defaults(func=_onboard)

    p = sub.add_parser("upload", help="index documents the owner supplied")
    p.add_argument("property_id")
    p.add_argument("paths", nargs="+", help="files or directories to index")
    p.set_defaults(func=_upload)

    p = sub.add_parser("ask", help="ask the Guide a question")
    p.add_argument("property_id")
    p.add_argument("question")
    p.set_defaults(func=_ask)

    p = sub.add_parser("origins", help="show or change a property's origin allowlist")
    p.add_argument("property_id")
    p.add_argument("--add", action="append", help="origin to allow; repeatable")
    p.add_argument("--remove", action="append", help="origin to drop; repeatable")
    p.set_defaults(func=_origins)

    p = sub.add_parser("logs", help="read the chat log for a property")
    p.add_argument("property_id")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--session", help="only this session id")
    p.add_argument("--deflected", action="store_true",
                   help="only turns the corpus could not answer")
    p.set_defaults(func=_logs)

    p = sub.add_parser("eval", help="run an eval case file")
    p.add_argument("property_id")
    p.add_argument("cases")
    p.add_argument("--label")
    p.set_defaults(func=_eval)

    args = parser.parse_args()
    return asyncio.run(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
