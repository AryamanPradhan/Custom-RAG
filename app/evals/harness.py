"""Layer 08 - offline evaluation.

Scaffolding now, cases per client later. Three things get scored, because they
fail for different reasons and a single "quality" number hides which:

  retrieval_hit  Did the right Source reach the answer model? A miss here is a
                 chunking, embedding or reranking problem.
  grounded       Did the answer stay inside its sources? A miss here is the
                 answer model overreaching - the thing a small model does.
  deflection     Did the Guide decline when it should have? Scored separately
                 because a Guide that answers everything scores well on
                 groundedness right up until it invents a refund policy.

Cases live in a JSON file per property so a client's questions can be added
without touching code.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from app.logging_setup import get_logger
from app.pipeline.answer import AnswerPipeline
from app.storage.db import Database
from app.storage.properties import Property

log = get_logger(__name__)


@dataclass(slots=True)
class EvalCase:
    case_id: str
    question: str
    # A substring of the URI the answer should cite. Substring rather than
    # exact match so cases survive a site restructure that keeps the slug.
    expects_source: str | None = None
    # True when the corpus genuinely cannot answer and deflecting is correct.
    expects_deflection: bool = False
    # Substrings the answer must contain, matched case-insensitively. For the
    # cases where "grounded" is not enough: an answer can stay inside its
    # sources and still tell the visitor the opposite of what they needed,
    # which is exactly what happens when a general rule is retrieved in place
    # of the specific exception to it.
    expects_answer_contains: list[str] = field(default_factory=list)
    note: str = ""


@dataclass(slots=True)
class CaseResult:
    case_id: str
    question: str
    retrieved_hit: bool
    grounded: bool
    deflected: bool
    passed: bool
    answer: str = ""
    detail: str = ""


@dataclass(slots=True)
class EvalRun:
    run_id: str
    property_id: str
    label: str
    started_at: str
    results: list[CaseResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    def rate(self, attr: str) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if getattr(r, attr)) / len(self.results)

    def summary(self) -> dict:
        return {
            "run_id": self.run_id,
            "property_id": self.property_id,
            "label": self.label,
            "cases": self.total,
            "pass_rate": round(self.rate("passed"), 3),
            "retrieval_hit_rate": round(self.rate("retrieved_hit"), 3),
            "grounded_rate": round(self.rate("grounded"), 3),
            "deflection_rate": round(self.rate("deflected"), 3),
        }


def load_cases(path: str | Path) -> list[EvalCase]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        EvalCase(
            case_id=item.get("case_id") or f"case-{i}",
            question=item["question"],
            expects_source=item.get("expects_source"),
            expects_deflection=bool(item.get("expects_deflection", False)),
            expects_answer_contains=list(item.get("expects_answer_contains", []) or []),
            note=item.get("note", ""),
        )
        for i, item in enumerate(data.get("cases", []), start=1)
    ]


async def run_eval(
    *,
    prop: Property,
    cases: list[EvalCase],
    pipeline: AnswerPipeline,
    db: Database | None = None,
    label: str = "",
) -> EvalRun:
    """Run every case through the real pipeline.

    Deliberately not mocked: the whole point is to measure what a Visitor would
    actually get, including the guardrails and the verifier. This spends real
    tokens - one full turn per case.
    """
    run = EvalRun(
        run_id=uuid.uuid4().hex[:12],
        property_id=prop.property_id,
        label=label or "unlabelled",
        started_at=datetime.now(UTC).isoformat(),
    )

    for case in cases:
        result = await pipeline.answer(prop, case.question, [])

        cited = " ".join(c.uri for c in result.citations)
        hit = bool(case.expects_source and case.expects_source in cited)

        if case.expects_deflection:
            # A refusal the answer model writes itself - "I'm unable to make
            # bookings; please contact the property" - does the same job as the
            # pipeline's Deflection and should score the same. What the two
            # have in common is citing nothing: an informational answer in this
            # system always carries the sources it came from, so an uncited
            # answer is a non-answer however it was phrased.
            passed = result.deflected or not result.citations
            detail = "" if passed else "answered a question it should have deflected"
        elif result.deflected:
            passed = False
            detail = f"deflected unexpectedly: {result.reason}"
        else:
            lowered = result.answer.lower()
            missing = [
                phrase
                for phrase in case.expects_answer_contains
                if phrase.lower() not in lowered
            ]
            passed = result.grounded and not missing
            detail = "" if passed else (result.reason or "")
            if case.expects_source:
                passed = passed and hit
                if not hit:
                    detail = f"expected a citation matching {case.expects_source!r}"
            if missing:
                detail = f"answer did not mention {missing!r}"

        run.results.append(
            CaseResult(
                case_id=case.case_id,
                question=case.question,
                retrieved_hit=hit,
                grounded=result.grounded,
                deflected=result.deflected,
                passed=passed,
                answer=result.answer[:500],
                detail=detail,
            )
        )

    if db is not None:
        await _persist(db, run)

    log.info(
        f"Eval finished: {run.rate('passed'):.0%} passed of {len(run.results)} cases.",
        **run.summary(),
    )
    return run


async def _persist(db: Database, run: EvalRun) -> None:
    await db.conn.execute(
        """INSERT INTO eval_runs
           (run_id, property_id, label, started_at, cases, retrieval_hits,
            grounded, correct_deflect)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            run.run_id,
            run.property_id,
            run.label,
            run.started_at,
            run.total,
            sum(1 for r in run.results if r.retrieved_hit),
            sum(1 for r in run.results if r.grounded),
            sum(1 for r in run.results if r.deflected),
        ),
    )
    await db.conn.executemany(
        """INSERT INTO eval_results
           (run_id, case_id, question, retrieved_hit, grounded, deflected,
            answer, detail)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (
                run.run_id,
                r.case_id,
                r.question,
                int(r.retrieved_hit),
                int(r.grounded),
                int(r.deflected),
                r.answer,
                r.detail,
            )
            for r in run.results
        ],
    )
    await db.conn.commit()
