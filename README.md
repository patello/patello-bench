# patello-bench ⚖️

A personal LLM benchmark for failure modes that matter in real assistant workloads — built as a fork of [databricks/judges](https://github.com/databricks/judges) (v0.1.1, branch `patello-bench`).

It targets three failure classes observed in production agents (calorie coaching, commute briefs, transit planning), then runs a **provider matrix**: the same model pinned to different OpenRouter providers, to separate model-level weaknesses from serving-stack/quantization damage.

## What it tests

| Task | Judge | Failure class |
|---|---|---|
| `syn-swedish-day-summary` | `SwedishFabrication` | Invented Swedish words, Nordic-language intrusions, word salad |
| `syn-swedish-commute-brief` | `SwedishFabrication` | Same, in winter-driving vocabulary ("piggdäck"-type chimera words) |
| `syn-swedish-evening-plan` | `SwedishFabrication` | Same, in everyday-reminder register |
| `syn-kcal-overage` | `KcalBudgetFraming` | Advising eating when the daily budget is already exceeded |
| `syn-kcal-under-target` | `KcalBudgetFraming` | Framing remaining deficit as a budget to "spend" on treats |
| `syn-transit-connection` | `TransitConnectionFeasibility` | Presenting implausibly tight (<5 min) or impossible transfer margins as fine routes |

The judges are LLM-as-judge boolean classifiers grounded in dictionary criteria (SAOL/SO for Swedish, explicit budget/transfer-margin rules). Prompts are synthetic but modeled on real failure episodes; raw private transcripts are **not** included (see `data/private/`, git-ignored).

## Layout

```
bench/gen_providers.py        # provider-pinned generation + judging matrix
judges/classifiers/patello.py # the three custom judges
data/public/synthetic-tasks.json  # the six task prompts
data/private/                 # git-ignored: private transcript corpus + goldens
results/                      # git-ignored: run outputs (JSONL per run)
```

## Running

Requires an OpenRouter API key (generation + judge model are billed per run).

```bash
python -m venv .venv && .venv/bin/pip install -e .
echo 'OPENROUTER_API_KEY=sk-or-...' > .env
set -a && . ./.env && set +a

# Full matrix, all default providers, 10 samples each
.venv/bin/python bench/gen_providers.py -N 10 --no-thinking

# Single provider, single task class
.venv/bin/python bench/gen_providers.py \
    --providers "OpenInference,Relace" \
    --judge-class swedish_fabrication \
    -N 10 --no-thinking

# Offline validation without any API calls
.venv/bin/python bench/gen_providers.py --dry-run
.venv/bin/pytest tests/ -q
```

Each run writes `results/providers-<UTC-timestamp>.jsonl` — one row per (sample, task, provider) with the raw output, judge verdict, and reasoning — then prints a task × provider fail matrix.

### Pinned model

Provider runs pin `deepseek/deepseek-v4-flash-0731` with `allow_fallbacks: false`, so every row is attributable to a specific serving stack (Relace, OpenInference, Baidu, DeepSeek official, ...). Provider names must match OpenRouter's current endpoint list; delisted providers 404 immediately.

## Interpreting results

- Compare providers per task, not just on aggregate totals — one systematic failure on a single task can dominate an otherwise-flat matrix.
- At N=10 per task, differences need Fisher exact (or larger N) before you trust them; ~N=30 per contested task is the minimum for a defensible call.
- Cross-provider recurrence of the *same* fabricated word (e.g. "piggdäck") indicates model-level contamination; a failure unique to one provider indicates serving-stack damage.
