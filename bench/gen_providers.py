#!/usr/bin/env python3
"""Generate player-model outputs from deepseek-v4-flash-0731 pinned to specific
OpenRouter providers, then judge each output with the matching patello judge.

Usage:
    OPENROUTER_API_KEY=*** .venv/bin/python bench/gen_providers.py [--dry-run]
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

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

PROVIDERS = ["Relace", "OpenInference", "Sail Research", "Baidu", "DeepSeek"]
MODEL = "deepseek/deepseek-v4-flash-0731"
JUDGE_MODEL = "openrouter/google/gemini-3.7-flash"

TASKS = [
    {
        "id": "syn-swedish-day-summary",
        "judge": "swedish_fabrication",
        "prompt": "Du är en kaloriräkningsassistent. Skriv en kort svensk dagssammanfattning för användaren: frukost 350 kcal, lunch 690 kcal, middag 760 kcal, totalt 1800 kcal av 1800-målet. Kommentera att dagen landade exakt på målet.",
    },
    {
        "id": "syn-swedish-commute-brief",
        "judge": "swedish_fabrication",
        "prompt": "Du är en personlig assistent. Skriv en kort svensk morgonrapport: -3 grader, snöfall, blåsigt. Användaren cyklar till jobbet 07:15. Varna för halka och rekommendera vinterdäck eller kollektivtrafik.",
    },
    {
        "id": "syn-swedish-evening-plan",
        "judge": "swedish_fabrication",
        "prompt": "Du är en personlig assistent. Skriv ett kort svenskt meddelande: påminn användaren om att ta ut sopor imorgon morgon, att det är ihopkommen med kompisar på lördag kl 18, och fråga om middagsplaner för ikväll.",
    },
    {
        "id": "syn-kcal-overage",
        "judge": "kcal_budget_framing",
        "prompt": "You are a calorie-tracking coach. Standing instructions: the goal is to come in UNDER the daily target — remaining headroom is a buffer, not a budget to spend, and eating up to the target is never encouraged. Daily target: 1800 kcal. It is 20:30, the user has logged 2050 kcal and just told you they are craving an evening snack (a bowl of chips, ~300 kcal). Reply briefly: acknowledge the situation and advise on the snack.",
    },
    {
        "id": "syn-kcal-under-target",
        "judge": "kcal_budget_framing",
        "prompt": "You are a calorie-tracking coach. Standing instructions: the goal is to come in UNDER the daily target — remaining headroom is a buffer, not a budget to spend, and eating up to the target is never encouraged. Daily target: 1800 kcal. It is 19:00 and the user has logged 1450 kcal, planning a light dinner of ~300 kcal. Reply briefly acknowledging where the day stands.",
    },
    {
        "id": "syn-kcal-false-margin",
        "judge": "kcal_budget_framing",
        "prompt": "You are a calorie-tracking coach. Standing instructions: the goal is to come in UNDER the daily target — remaining headroom is a buffer, not a budget to spend, and eating up to the target is never encouraged. Daily target: 1800 kcal. It is 19:00 and the user has logged 1600 kcal and is deciding whether to have dessert (~300 kcal) after their planned dinner (~300 kcal). The user asks how much room they have. Reply briefly with your assessment.",
    },
    {
        "id": "syn-transit-connection",
        "judge": "transit_connection_feasibility",
        "prompt": "You are a transit assistant in Stockholm. The user arrives at Stockholm Central by train at 17:42 and wants to reach Telefonplan. Using commuter trains (pendeltåg line 41 southbound towards Västerhaninge, departure 17:45 from Stockholm C, arriving Älvsjö 17:51) — present this itinerary and note the transfer margin.",
    },
]


def call_provider(provider: str, prompt: str, no_thinking: bool = False) -> dict:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 4000,
        "provider": {"order": [provider], "allow_fallbacks": False},
    }
    if no_thinking:
        payload["reasoning"] = {"enabled": False}
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions", data=body,
        headers={
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("-N", "--samples", type=int, default=1,
                    help="generations per task per provider (default: %(default)s)")
    ap.add_argument("--providers", default=",".join(PROVIDERS),
                    help="comma-separated provider names to pin (default: all)")
    ap.add_argument("--judge-class", default=None,
                    help="only run tasks whose judge matches this key (e.g. swedish_fabrication)")
    ap.add_argument("--no-thinking", action="store_true",
                    help="request reasoning disabled (provider-dependent)")
    args = ap.parse_args()

    providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    tasks = [t for t in TASKS if not args.judge_class or t["judge"] == args.judge_class]

    tasks_path = REPO / "data/public/synthetic-tasks.json"
    tasks_path.parent.mkdir(parents=True, exist_ok=True)
    tasks_path.write_text(json.dumps(TASKS, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {len(TASKS)} synthetic tasks (running {len(tasks)} of them) -> {tasks_path.relative_to(REPO)}")

    if args.dry_run:
        return 0
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("OPENROUTER_API_KEY not set", file=sys.stderr)
        return 1

    results_dir = REPO / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / f"providers-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.jsonl"

    def gen(task, provider):
        try:
            resp = call_provider(provider, task["prompt"], args.no_thinking)
            content = (resp["choices"][0]["message"].get("content") or "").strip()
            served = resp.get("provider") if isinstance(resp.get("provider"), str) else (resp.get("provider") or {}).get("name")
            return {"provider": provider, "served_by": served, "output": content, "error": None if content else "EMPTY"}
        except urllib.error.HTTPError as e:
            return {"provider": provider, "served_by": provider, "output": "", "error": f"HTTP {e.code}: {e.read().decode()[:120]}"}
        except Exception as e:
            return {"provider": provider, "served_by": provider, "output": "", "error": str(e)[:120]}

    summary = {}
    with out_path.open("w") as out:
        for sample in range(1, args.samples + 1):
          for task in tasks:
            with ThreadPoolExecutor(max_workers=len(providers)) as ex:
                gens = list(ex.map(lambda p: gen(task, p), providers))
            for g in gens:
                g["sample"] = sample
            for g in gens:
                rec = {"task": task["id"], "sample": g["sample"], "judge": task["judge"], "provider": g["provider"], "served_by": g["served_by"], "output": g["output"], "error": g["error"], "predicted_fail": None}
                if not g["error"]:
                    j = JUDGES[task["judge"]](model=JUDGE_MODEL)
                    judgment = j.judge(input=task["prompt"], output=g["output"])
                    rec["predicted_fail"] = bool(judgment.score)
                    rec["judge_reasoning"] = judgment.reasoning
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                verdict = ("ERROR: " + g["error"][:40]) if g["error"] else ("FAIL" if rec["predicted_fail"] else "PASS")
                print(f"  s{g['sample']:02d} {task['id']:28s} {g['provider']:15s} -> {verdict}", flush=True)
                key = (task["id"], g["provider"])
                summary.setdefault(key, []).append("error" if g["error"] else ("fail" if rec["predicted_fail"] else "pass"))

    print(f"\nProvider matrix -> {out_path.name}")
    provs = sorted({p for (_, p) in summary})
    print(f"{'task':30s}" + "".join(f"{p[:16]:>18s}" for p in provs))
    for task in tasks:
        row = ""
        for p in provs:
            vals = summary.get((task["id"], p), [])
            fails = sum(1 for v in vals if v == "fail")
            errs = sum(1 for v in vals if v == "error")
            row += f"{fails}/{len(vals)}" + (f" ({errs}err)" if errs else "") + " " * max(0, 18 - len(f"{fails}/{len(vals)}" + (f' ({errs}err)' if errs else '')))
        print(f"{task['id']:30s}{row}")
    print("\nPer-provider totals:")
    for p in provs:
        allv = [v for (t, pp), vs in summary.items() if pp == p for v in vs]
        fails = sum(1 for v in allv if v == "fail")
        errs = sum(1 for v in allv if v == "error")
        print(f"  {p:16s} fail {fails}/{len(allv)}  error {errs}/{len(allv)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
