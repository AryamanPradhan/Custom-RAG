"""Operator CLI.

Everything an operator does by hand: onboard a client, crawl their site, check
whether their corpus has drifted, ask the Guide a question, run the eval set.

    guide onboard casa-verde "Casa Verde" --origin https://casaverde.com \\
          --phone "+44 1234 567890"
    guide crawl casa-verde https://casaverde.com
    guide upload casa-verde ./house-rules.pdf
    guide ask casa-verde "can I bring my dog?"
    guide drift casa-verde
    guide eval casa-verde ./evals/casa-verde.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from app.config import get_settings
from app.evals.harness import load_cases, run_eval
from app.gateway.llm_gateway import LLMGateway
from app.ingestion.drift import check_drift
from app.ingestion.pipeline import IngestionPipeline
from app.logging_setup import configure_logging
from app.pipeline.answer import AnswerPipeline
from app.retrieval.embeddings import get_dense_embedder, get_sparse_encoder
from app.retrieval.retriever import Retriever
from app.retrieval.store import VectorStore
from app.storage.db import close_db, init_db
from app.storage.properties import ContactRoute, PropertyRepository


class _Context:
    def __init__(self, settings, db, repo, store, dense, sparse, gateway) -> None:
        self.settings = settings
        self.db = db
        self.repo = repo
        self.store = store
        self.dense = dense
        self.sparse = sparse
        self.gateway = gateway

    def answer_pipeline(self) -> AnswerPipeline:
        return AnswerPipeline(
            gateway=self.gateway,
            retriever=Retriever(
                self.store, self.dense, self.sparse, top_k=self.settings.retrieve_top_k
            ),
            rerank_top_n=self.settings.rerank_top_n,
            min_rerank_score=self.settings.min_rerank_score,
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


async def _build(*, need_embeddings: bool = True) -> _Context:
    settings = get_settings()
    configure_logging(settings.log_level)
    db = await init_db(settings.database_path)
    repo = PropertyRepository(db)

    async def record(property_id: str, amount: float) -> None:
        await repo.record_spend(property_id, amount)

    gateway = LLMGateway(settings, on_spend=record)
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


async def _crawl(args) -> int:
    ctx = await _build()
    try:
        report = await ctx.ingestion().ingest_site(
            args.property_id, args.start_url, max_pages=args.max_pages
        )
        await ctx.repo.mark_crawled(args.property_id)
        print(
            f"{report.documents} documents -> {report.chunks} chunks "
            f"({report.skipped_unchanged} unchanged, {report.duration_seconds}s)"
        )
        for error in report.errors[:10]:
            print(f"  ! {error}", file=sys.stderr)
        return 0 if report.chunks or report.skipped_unchanged else 1
    finally:
        await ctx.store.close()
        await close_db()


async def _upload(args) -> int:
    ctx = await _build()
    try:
        path = Path(args.path)
        report = await ctx.ingestion().ingest_upload(
            args.property_id, path.name, path.read_bytes()
        )
        print(f"{report.documents} documents -> {report.chunks} chunks")
        for error in report.errors:
            print(f"  ! {error}", file=sys.stderr)
        return 0 if report.chunks else 1
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
        result = await ctx.answer_pipeline().answer(prop, args.question)
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


async def _drift(args) -> int:
    ctx = await _build(need_embeddings=False)
    try:
        report = await check_drift(ctx.db, ctx.settings, args.property_id)
        print(report.summary())
        for uri in report.changed:
            print(f"  changed: {uri}")
        for uri in report.unreachable:
            print(f"  unreachable: {uri}")
        return 0
    finally:
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
    p.add_argument("--cap", type=float, default=5.0, help="daily spend cap in USD")
    p.set_defaults(func=_onboard)

    p = sub.add_parser("crawl", help="crawl and index a property website")
    p.add_argument("property_id")
    p.add_argument("start_url")
    p.add_argument("--max-pages", type=int, default=None)
    p.set_defaults(func=_crawl)

    p = sub.add_parser("upload", help="index a PDF or DOCX")
    p.add_argument("property_id")
    p.add_argument("path")
    p.set_defaults(func=_upload)

    p = sub.add_parser("ask", help="ask the Guide a question")
    p.add_argument("property_id")
    p.add_argument("question")
    p.set_defaults(func=_ask)

    p = sub.add_parser("drift", help="check whether the corpus has fallen behind")
    p.add_argument("property_id")
    p.set_defaults(func=_drift)

    p = sub.add_parser("eval", help="run an eval case file")
    p.add_argument("property_id")
    p.add_argument("cases")
    p.add_argument("--label")
    p.set_defaults(func=_eval)

    args = parser.parse_args()
    return asyncio.run(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
