#!/usr/bin/env python3
"""Patello-bench runner: evaluate the private corpus against the custom judges.

Usage:
    OPENROUTER_API_KEY=sk-or-... .venv/bin/python bench/run_bench.py \
        --judge-model openrouter/google/gemini-2.5-flash [--dry-run]

Reads data/private/corpus.jsonl (git-ignored), runs the appropriate custom
judge per row, writes per-row verdicts to results/<timestamp>.jsonl and
prints a scorecard: agreement with human labels per failure class.
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from judges import Jury  # noqa: E402
from judges.classifiers.patello import (  # noqa: E402
    KcalBudgetFraming,
    SwedishFabrication,
    TransitConnectionFeasibility,
)

JUDGES = {
    "swedish_fabrication": SwedishFabrication,
    "kcal_budget_framing": KcalBudgetFraming,
    "transit_connection_feasibility": TransitConnectionFeasibility,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--judge-model",
        default="openrouter/google/gemini-3.7-flash",
        help="instructor provider/model string for the judge (default: %(default)s)",
    )
    ap.add_argument(
        "--jury",
        default=None,
        help="comma-separated instructor model slugs; overrides --judge-model with a Jury (majority vote)",
    )
    ap.add_argument(
        "--cascade",
        action="store_true",
        help="two cheap judges in parallel; on disagreement, gemini tie-breaker. "
        "Configure via --cheap-judges / --tie-breaker.",
    )
    ap.add_argument(
        "--cheap-judges",
        default="openrouter/deepseek/deepseek-v4-flash,openrouter/qwen/qwen3.5-flash",
        help="comma-separated cheap tier (default: %(default)s)",
    )
    ap.add_argument(
        "--tie-breaker",
        default="openrouter/google/gemini-3.7-flash",
        help="tie-breaker judge for --cascade (default: %(default)s)",
    )
    ap.add_argument("--corpus", default=str(REPO / "data/private/corpus.jsonl"))
    ap.add_argument("--dry-run", action="store_true", help="load corpus and judges only")
    args = ap.parse_args()

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        print(f"corpus not found: {corpus_path}", file=sys.stderr)
        return 1
    rows = [json.loads(l) for l in corpus_path.read_text().splitlines() if l.strip()]
    print(f"corpus: {len(rows)} rows")

    unknown = {r["judge"] for r in rows} - set(JUDGES)
    if unknown:
        print(f"unknown judge keys in corpus: {unknown}", file=sys.stderr)
        return 1
    print("judge keys ok:", ", ".join(sorted(JUDGES)))

    if args.dry_run:
        print("dry run ok")
        return 0

    if not os.environ.get("OPENROUTER_API_KEY"):
        print(
            "OPENROUTER_API_KEY not set — judges call OpenRouter via instructor.",
            file=sys.stderr,
        )
        return 1

    jury_models = args.jury.split(",") if args.jury else None
    results_dir = REPO / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.jsonl"

    tp = fp = tn = fn = 0
    with out_path.open("w") as out:
        def judge_row(judge_cls, model, inp, outp):
            return judge_cls(model=model).judge(input=inp, output=outp)

        for r in rows:
            if getattr(args, "cascade", False):
                cheap = [m.strip() for m in args.cheap_judges.split(",")]
                with ThreadPoolExecutor(max_workers=len(cheap)) as ex:
                    futures = {
                        m: ex.submit(judge_row, JUDGES[r["judge"]], m, r["input"], r["output"])
                        for m in cheap
                    }
                    cheap_judgments = {m: f.result() for m, f in futures.items()}
                cheap_scores = {m: bool(jd.score) for m, jd in cheap_judgments.items()}
                distinct = set(cheap_scores.values())
                tie_broken = None
                tie_reasoning = None
                if len(distinct) > 1:
                    tb = judge_row(JUDGES[r["judge"]], args.tie_breaker, r["input"], r["output"])
                    tie_broken = bool(tb.score)
                    tie_reasoning = tb.reasoning
                    predicted_fail = tie_broken
                else:
                    predicted_fail = next(iter(distinct))
                reasoning = " || ".join(
                    f"[{m}] {jd.reasoning}" for m, jd in cheap_judgments.items()
                )
                if tie_reasoning is not None:
                    reasoning += f" || [TIE-BREAKER {args.tie_breaker}] {tie_reasoning}"
                per_judge = dict(cheap_scores)
                if tie_broken is not None:
                    per_judge[f"{args.tie_breaker} (tie-breaker)"] = tie_broken
                human_fail = bool(r["label"])
                rec = {
                    "id": r["id"],
                    "judge": r["judge"],
                    "mode": "cascade",
                    "human_label_fail": human_fail,
                    "predicted_fail": predicted_fail,
                    "match": predicted_fail == human_fail,
                    "judge_reasoning": reasoning,
                    "per_judge": per_judge,
                    "model": r.get("model"),
                    "host_window": r.get("host_window"),
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                tie = " tie->" + ("F" if tie_broken else "P") if tie_broken is not None else ""
                print(f"  {r['id']}: pred={'FAIL' if predicted_fail else 'PASS'} "
                      f"human={'FAIL' if human_fail else 'PASS'} "
                      f"{'ok' if rec['match'] else 'MISMATCH'}{tie}")
                tp += predicted_fail and human_fail
                fp += predicted_fail and not human_fail
                tn += not predicted_fail and not human_fail
                fn += not predicted_fail and human_fail
                continue
            if jury_models:
                jury = Jury(
                    judges=[JUDGES[r["judge"]](model=m.strip()) for m in jury_models],
                    voting_method="majority",
                )
                verdict = jury.vote(input=r["input"], output=r["output"])
                predicted_fail = bool(verdict.score)
                reasoning = " || ".join(
                    f"[{m}] {jd.reasoning}"
                    for m, jd in zip(jury_models, verdict.judgments)
                )
                per_judge = {
                    m.strip(): bool(jd.score)
                    for m, jd in zip(jury_models, verdict.judgments)
                }
            else:
                j = JUDGES[r["judge"]](model=args.judge_model)
                judgment = j.judge(input=r["input"], output=r["output"])
                predicted_fail = bool(judgment.score)
                reasoning = judgment.reasoning
                per_judge = None
            human_fail = bool(r["label"])
            rec = {
                "id": r["id"],
                "judge": r["judge"],
                "human_label_fail": human_fail,
                "predicted_fail": predicted_fail,
                "match": predicted_fail == human_fail,
                "judge_reasoning": reasoning,
                "per_judge": per_judge,
                "model": r.get("model"),
                "host_window": r.get("host_window"),
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"  {r['id']}: pred={'FAIL' if predicted_fail else 'PASS'} "
                  f"human={'FAIL' if human_fail else 'PASS'} "
                  f"{'ok' if rec['match'] else 'MISMATCH'}")
            tp += predicted_fail and human_fail
            fp += predicted_fail and not human_fail
            tn += not predicted_fail and not human_fail
            fn += not predicted_fail and human_fail

    total = len(rows)
    print(f"\nScorecard — {out_path.name}")
    print(f"  agreement: {tp+tn}/{total}")
    print(f"  true fails caught: {tp}/{tp+fn}  false alarms: {fp}/{fp+tn}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
