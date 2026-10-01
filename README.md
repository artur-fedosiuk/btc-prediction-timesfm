# TimesFM

## BTC forward experiment status

The BTC collector now has a separate immutable SQLite H24 implementation in
`evaluation/forward`. The 12 historical CSV rows remain `legacy_h1` and are
invalid for H24 evaluation. Candidate providers are Coinbase BTC/USD (primary)
and Kraken BTC/USD (cross-check); adoption is gated on an actual Actions probe.

Local tests and a real pinned TimesFM 3.0 smoke test are verified. CI execution
and the first live forward prediction are **not yet verified**. Do not infer
predictive edge, profitability, or production readiness.

See [P0–P4 acceptance report](reports/p0-p4-verification-2026-09-27/VERIFICATION.md)
and [temporal/model/operational contract](reports/p0-p4-verification-2026-09-27/pipeline.md).
The upstream model documentation follows.

TimesFM (Time Series Foundation Model) is a pretrained time-series foundation
model developed by Google Research for time-series forecasting.

*   Paper:
    [A decoder-only foundation model for time-series forecasting](https://arxiv.org/abs/2310.10688),
    ICML 2024.
*   <span style="color:red">(NEW!)</span> TimesFM 3.0 Checkpoint:
    [`google/timesfm-3.0-pytorch`](https://huggingface.co/google/timesfm-3.0-pytorch).
*   Checkpoints (up to 2.5):
    [TimesFM Hugging Face Collection](https://huggingface.co/collections/google/timesfm-release-66e4be5fdb56e960c1e482a6).
*   [Google Research blog](https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/)
    (New blog post for TimesFM 3.0 coming soon!).
*   TimesFM in Google 1P Products:
    *   [BigQuery ML](https://cloud.google.com/bigquery/docs/timesfm-model):
        Enterprise level SQL queries for scalability and reliability.
    *   [Google Sheets](https://workspaceupdates.googleblog.com/2026/02/forecast-data-in-connected-sheets-BigQueryML-TimesFM.html):
        For your daily spreadsheet.
    *   [Vertex Model Garden](https://pantheon.corp.google.com/vertex-ai/publishers/google/model-garden/timesfm):
        Dockerized endpoint for agentic calling.

This open version is not an officially supported Google product.

**Latest Model Version:** TimesFM 3.0

**Archived Model Versions:**

-   2.5: relevant code under `src/timesfm`.
-   1.0 and 2.0: relevant code archived in the subdirectory `v1`. You can `pip
    install timesfm==1.3.0` to install an older version of this package to load
    them.

--------------------------------------------------------------------------------

## Update — August 2026

**TimesFM 3.0 is out!**

TimesFM 3.0 introduces native **multivariate time-series forecasting**, flexible
**covariate support** (both past-only and past-and-future covariates), superior
zero-shot generalist capabilities, and top performance across all three major
time-series foundation model benchmarks.

### Key Highlights:

-   **Native Multivariate & Univariate Forecasting with Covariates**: Seamlessly
    forecast multi-channel multivariate series as well as individual univariate
    series, with native support for past-only and past-and-future dynamic
    covariates without per-task tuning.
-   **Top Benchmark Performance**:
    -   🥇 **fev-bench**: **Rank #1 overall** across 100 diverse real-world
        forecasting tasks.
    -   🥇 **TIME Benchmark**: **Rank #1 overall** across 50 domain datasets and
        98 evaluation tasks.
    -   🥇 **GIFT-Eval**: **Rank #1 among all foundation models**.

### License notice for pretrained weights

> **Important:** The TimesFM source code in this repository is licensed under
> Apache-2.0, and model weights up to version 2.5 remain Apache-2.0. However,
> for the time being, TimesFM 3.0 pretrained weights are distributed under the
> separate `timesfm-non-commercial-license-v1.0` license and are restricted to
> non-commercial, non-production use. Commercial or production use of the
> default pretrained weights is **not permitted**.

--------------------------------------------------------------------------------

## Update - July 2, 2026

Updated PyPI to `timesfm=2.0.2`. See
[Install](https://github.com/google-research/timesfm#from-pypi).

## Update - Apr. 9, 2026

Added fine-tuning example using HuggingFace Transformers + PEFT (LoRA) — see
[`timesfm-forecasting/examples/finetuning/`](timesfm-forecasting/examples/finetuning/).
Also added unit tests (`tests/`) and incorporated several community fixes.

Shoutout to [@kashif](https://github.com/kashif) and
[@darkpowerxo](https://github.com/darkpowerxo).

## Update - Mar. 19, 2026

Huge shoutout to [@borealBytes](https://github.com/borealBytes) for adding the
support for
[AGENTS](https://github.com/google-research/timesfm/blob/master/AGENTS.md)!
TimesFM
[SKILL.md](https://github.com/google-research/timesfm/tree/master/timesfm-forecasting)
is out.

## Update - Oct. 29, 2025

Added back the covariate support through XReg for TimesFM 2.5.

## Update - Sept. 15, 2025

TimesFM 2.5 is out!

Comparing to TimesFM 2.0, this new 2.5 model:

-   uses 200M parameters, down from 500M.
-   supports up to 16k context length, up from 2048.
-   supports continuous quantile forecast up to 1k horizon via an optional 30M
    quantile head.
-   gets rid of the `frequency` indicator.
-   has a couple of new forecasting flags.

Since the Sept. 2025 launch, the following improvements have been completed for
TimesFM 2.5:

1.  ✅ Flax version of the model for faster inference.
2.  ✅ Covariate support via XReg (see Oct. 2025 update).
3.  ✅ Documentation, examples, and agent skill (see `timesfm-forecasting/`).
4.  ✅ Fine-tuning example with LoRA via HuggingFace Transformers + PEFT (see
    `timesfm-forecasting/examples/finetuning/`).
5.  ✅ Unit tests for core layers, configs, and utilities (see `tests/`).

### Install

#### From `PyPI`

```shell
# Install TimesFM with PyTorch
pip install timesfm[torch]
```

#### Local Install

1.  Clone the repository:

    ```shell
    git clone https://github.com/google-research/timesfm.git
    cd timesfm
    ```

2.  Create a virtual environment and install with PyTorch:

    ```shell
    # Using uv
    uv venv
    source .venv/bin/activate

     # Install the package in editable mode with torch
    uv pip install -e .[torch]
    ```

--------------------------------------------------------------------------------

### Code Examples: TimesFM 3.0

#### 1. Univariate Forecasting (Variable Lengths)

Pass a batch of 1D NumPy arrays of different context lengths to forecast
univariate time series:

```python
import numpy as np
from timesfm3 import TimesFM3Evaluator, ModelConfig

# Initialize TimesFM 3.0
config = ModelConfig(
    checkpoint_path="google/timesfm-3.0-pytorch",
    per_core_batch_size=32,
    device="cuda"
)
forecaster = TimesFM3Evaluator(config)

# Two univariate series of different lengths (100 and 72 steps)
ts1 = np.linspace(0, 1, 100).astype(np.float32)
ts2 = np.sin(np.linspace(0, 24, 72)).astype(np.float32)

# Generate forecast (point predictions + 9 quantiles: 0.1 to 0.9)
outputs = list(forecaster.predict_batch([ts1, ts2], horizon=12, return_quantiles=True, use_symmetric_averaging=False))

print("Series 1 forecast shape:", outputs[0].forecast.shape)   # (12,)
print("Series 1 quantiles shape:", outputs[0].quantiles.shape) # (12, 9)

print("Series 2 forecast shape:", outputs[1].forecast.shape)   # (12,)
print("Series 2 quantiles shape:", outputs[1].quantiles.shape) # (12, 9)
```

#### 2. Multivariate Forecasting with Covariates

Pass a 2D array of shape `(num_variates, context_length)` along with optional
past-only and past-and-future covariates:

```python
import numpy as np
from timesfm3 import TimesFM3Evaluator, ModelConfig

# Initialize TimesFM 3.0
config = ModelConfig(
    checkpoint_path="google/timesfm-3.0-pytorch",
    per_core_batch_size=16,
    device="cuda"
)
forecaster = TimesFM3Evaluator(config)

context_len = 128
horizon = 24

# 3 target variates across past context: (3, 128)
target = np.random.randn(3, context_len).astype(np.float32)

# 1 past-only covariate channel across past context: (1, 128)
past_only_cov = np.random.randn(1, context_len).astype(np.float32)

# 2 past-and-future covariate channels across context + horizon: (2, 152)
past_future_cov = np.random.randn(2, context_len + horizon).astype(np.float32)

# Generate joint forecast across all 3 target variates
outputs = list(
    forecaster.predict_batch(
        contexts=[target],
        horizon=horizon,
        past_only_covariates=[past_only_cov],
        past_future_covariates=[past_future_cov],
        return_quantiles=True,
        use_symmetric_averaging=False,
    )
)

print("Multivariate forecast shape:", outputs[0].forecast.shape)   # (3, 24)
print("Multivariate quantiles shape:", outputs[0].quantiles.shape) # (3, 24, 9)
```

--------------------------------------------------------------------------------

## Bitcoin Prediction Experiment (BTC/USD)

> ⚠️ **Disclaimer**: This is a technical experiment to evaluate TimesFM's
> forecasting ability on cryptocurrency data. It is **not** financial advice.

### What It Does

A 14-day automated experiment that:

1.  **Predicts** the BTC/USD price 24 hours ahead (daily at 12:00 UTC).
2.  **Verifies** yesterday's prediction against the actual price.
3.  **Records** all results in
    [`data/bitcoin_predictions.csv`](data/bitcoin_predictions.csv).
4.  **Generates** a Markdown report at
    [`reports/bitcoin_report.md`](reports/bitcoin_report.md).

**Prediction methods:**

-   **TimesFM 3.0** — the only method used by the forward pipeline
    (`evaluation/forward/`). Requires `torch` + the package installed from
    source (see below). The pipeline **never falls back** to ARIMA; if the
    model cannot be loaded, the run fails loudly and is recorded as a failed
    event in the ledger.
-   **ARIMA / Persistence / Drift baselines** — computed offline from the
    same input context as TimesFM via `evaluation/forward/baselines.py` and
    embedded in `reports/data.json` for fair comparison. These are never used
    as live predictions.

### Configure the Experiment

The experiment window is controlled by environment variables:

```bash
# In .github/workflows/bitcoin-evaluation.yml:
EXPERIMENT_START_DATE: '2026-09-03'   # ISO date, UTC
EXPERIMENT_DURATION_DAYS: '14'         # Total days
```

The workflow automatically stops after the duration expires.

### Start Manually

1.  Go to the **Actions** tab in your GitHub repository.
2.  Select **"Bitcoin Prediction Evaluation"** from the left sidebar.
3.  Click **"Run workflow"** → **"Run workflow"**.

### Change the Schedule

Edit the `cron` expression in
[`.github/workflows/bitcoin-evaluation.yml`](.github/workflows/bitcoin-evaluation.yml):

```yaml
schedule:
  - cron: '0 12 * * *'   # Current: daily at 12:00 UTC
  # Examples:
  # - cron: '0 8 * * *'  # Daily at 08:00 UTC
  # - cron: '0 */12 * * *'  # Every 12 hours
```

### View the Report

The report is automatically committed to
[`reports/bitcoin_report.md`](reports/bitcoin_report.md) after each run.
It includes:

-   Summary statistics (MAE, RMSE, direction accuracy).
-   Daily results table with all metrics.
-   A cautious conclusion.

### Stop the Experiment

Three options:

1.  **Wait** — it stops automatically after `EXPERIMENT_DURATION_DAYS`.
2.  **Disable the workflow** — Go to Actions → Bitcoin Prediction Evaluation →
    ⋯ → Disable workflow.
3.  **Delete the workflow file** — Remove
    `.github/workflows/bitcoin-evaluation.yml`.

### Run Locally with TimesFM 3.0

> **Important:** `pip install timesfm[torch]` installs **TimesFM 2.0.2** from
> PyPI, which does **not** include `timesfm3` and will raise `ImportError`.
> Install from source instead:

```bash
# Clone and install in editable mode (includes timesfm3)
git clone https://github.com/google-research/timesfm.git
cd timesfm
pip install -e ".[torch]"

# Run prediction with the real model
python -m evaluation.forward.cli generate

# Verify matured predictions (run after their 24h window closes)
python -m evaluation.forward.cli verify
```

### Run Tests

The exact command used by the CI quality gate (see `.github/workflows/bitcoin-evaluation.yml`):

```bash
# Install dependencies first (editable install required for timesfm3)
pip install -e ".[torch]"
pip install pytest statsmodels pandas requests

# BTC pipeline regressions + mocked E2E
python -m pytest tests/forward tests/test_btc_metrics.py tests/test_btc_csv_manager.py \
  tests/test_btc_simulation.py tests/test_binance.py tests/test_price_verifier.py \
  tests/test_forecast_path.py tests/test_trading_engine.py -q

# TimesFM 3.0 source-level unit tests
python -m pytest src/timesfm3 -q
```

