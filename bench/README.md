# Patello-bench

Personal LLM benchmark built on [databricks/judges](https://github.com/databricks/judges).
Evaluates models against real failure modes observed in production personal
assistants (Swedish-language).

## Custom judges

`judges/classifiers/patello.py` — three boolean classifiers, each grounded in
real logged incidents:

| Judge | Catches | Ground truth |
|---|---|---|
| `SwedishFabrication` | Invented words ("honestare", "tyngder"), diacritic corruption ("üverskattad"), Danish/Norwegian/German intrusions ("Bilde"), uncontrolled code-switching | 2026-08-24 health-agent session |
| `KcalBudgetFraming` | Treating remaining kcal/time as reassurance when the goal is to come in *under* target | 2026-07-20 "allt är bakvänt" episode |
| `TransitConnectionFeasibility` | Connections departing before arrival, <5 min transfers, wrong direction labels | 2026-05-19 SL evening briefing |

## OpenRouter setup

Judges call models through [instructor](https://python.useinstructor.com/),
which has first-class OpenRouter support — pass any
`openrouter/<vendor>/<model>` slug as the judge model:

```bash
export OPENROUTER_API_KEY=sk-or-...
.venv/bin/python bench/run_bench.py --judge-model openrouter/google/gemini-2.5-flash
```

To evaluate a *player* model (the model under test), generate outputs with
the same slug through any OpenAI-compatible client pointed at
`https://openrouter.ai/api/v1`, then add rows to the corpus.

## Corpus

`data/private/corpus.jsonl` — **git-ignored** (private transcripts; never
commit). Schema per row:

```json
{
  "id": "...", "judge": "swedish_fabrication",
  "input": "...", "output": "...",
  "label": 1,  // 1 = failure, 0 = pass
  "feedback": "why",
  "model": "...", "host_window": "..."
}
```

Regenerate/extend from `../private-transcripts/*.md` (also private).

## Runner

```bash
.venv/bin/python bench/run_bench.py --dry-run   # validate corpus + imports
.venv/bin/python bench/run_bench.py             # full run, needs API key
```

Writes per-row verdicts (incl. judge reasoning) to `results/<ts>.jsonl`
(git-ignored) and prints agreement vs human labels.

## Tests

```bash
.venv/bin/python -m pytest tests/test_patello.py -q
```
