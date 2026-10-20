# CLAUDE.md

This file provides guidance to Claude Code when working with this BTC/USD prediction experiment using Google's TimesFM 3.0.

## Repository Overview

A 14-day automated experiment predicting BTC/USD 24h ahead (daily at 12:23 UTC per workflow), verifying against actuals, recording in SQLite ledger, exporting JSON, and generating `reports/bitcoin_report.md`. **TimesFM 3.0 only** — no ARIMA fallback.

## Key Directories

| Directory | Purpose |
|-----------|---------|
| `src/timesfm3/` | TimesFM 3.0 source (PyTorch) |
| `evaluation/forward/` | **Production pipeline** (CLI, pipeline, contracts, storage, model adapter, providers) |
| `evaluation/trading/` | Trading strategy backtesting |
| `scripts/` | Data download, oracle break-even analysis |
| `data/` | SQLite ledger (`forward.sqlite`) + exported JSON |
| `reports/` | Generated reports (`bitcoin_report.md`, `data.json`) |

## Critical Files

- `evaluation/forward/contracts.py` — Validation rules, temporal constraints, experiment constants (EXPERIMENT, CHECKPOINT, REVISION)
- `evaluation/forward/pipeline.py` — Generation/verification orchestration
- `evaluation/forward/model.py` — TimesFM 3.0 adapter with preflight checks (enforces `timesfm==3.0.1`)
- `evaluation/forward/storage.py` — SQLite schema and operations
- `.github/workflows/bitcoin-evaluation.yml` — CI/CD (hourly verify, daily generate; see workflow for times)

## Common Commands

```bash
# Install from source (required for timesfm3)
pip install -e ".[torch]"

# Run forward pipeline locally
python -m evaluation.forward.cli generate   # requires CI evidence first
python -m evaluation.forward.cli verify
python -m evaluation.forward.cli report

# Run tests — use the quality gate command from .github/workflows/bitcoin-evaluation.yml

# Download data
python -m scripts.download_daily_ohlc --years 5 --out research/data/btc_usd_1d.csv
```

## Do Not

1. **Never modify the pinned checkpoint** (`CHECKPOINT`/`REVISION` in `contracts.py`) without explicit approval — breaks reproducibility
2. **Never delete `data/forward.sqlite`** without explicit approval — the experiment ledger; CI commits it as evidence
3. **Never run `generate` without CI evidence** — requires `reports/provider-checks-ci.json` with both providers passing
4. **Never install `timesfm` from PyPI** — install only with `pip install -e ".[torch]"`; `model.py` enforces `timesfm==3.0.1`
5. **Never remove or bypass `preflight()` in `model.py`** — validates RAM (≥4 GB) and disk (≥3 GB) before loading
6. **Never run `verify` or `report` locally on the real ledger** — they write to `data/` and `reports/`; use CI or a copy

## Workflow

1. **Plan** with `/plan-feature` (planner subagent, model: opus) — step-by-step plan, no edits
2. **Tests first** with `/verify` and `test-writer` subagent — write failing tests before implementation
3. **Implement** in small steps — one commit per step
4. **Review** with `/review-changes` (reviewer subagent) — lists bugs/security/missing tests by severity
5. **Clear** with `/clear` between tasks; `/compact` around 70% context
6. **Max 2 subagents in parallel** (free-tier rate limits)
7. **Never claim success** without running `/verify` and citing its exit code