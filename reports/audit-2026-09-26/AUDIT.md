# BTC / TimesFM technical audit — 26 September 2026

**Verdict: the system does not satisfy the requested completion criteria. The live collector is currently failing, the evaluation target is wrong, and the HTML dashboard uses demonstration data. Existing metrics cannot establish predictive or trading edge.**

This document completes the initial investigation and proposes bounded corrective work. It does **not** certify the system, claim that bugs were fixed, or label the full requested implementation complete. No application code, experiment data, dependency, configuration, workflow, remote service, or production state was changed. Only audit documents were added.

Audit snapshot: branch `master`, commit `3f6782ce8aef432df0e79852dc289c506bf465e5`. Working tree was initially clean. Repository: `artur-fedosiuk/btc-prediction-timesfm`. Live API sample: 2026-09-26 10:38:52 UTC. Findings refer to this snapshot; external job/provider state can change.

## 1. Architecture discovered

```text
GitHub Actions: push to master / daily schedule / manual dispatch
  → evaluation.run_predict --use-timesfm
  → CoinGecko /simple/price                  [separate entry-price snapshot]
  → CoinGecko /market_chart?days=90          [timestamp + USD price samples]
  → discard all history timestamps
  → float64 array → last 512 values → float32
  → installed timesfm3.TimesFM3Evaluator, checkpoint google/timesfm-3.0-pytorch
      └─ exception → ARIMA(5,1,0), last 168 samples, distinctly named fallback
  → 24 sample forecasts + q10/q90 paths      [current source only]
  → ForecastPathMetrics                    [indices correct, times not attached]
  → append 38 fields to legacy CSV          [BREAK: legacy header has 16 fields]
  → evaluation.run_verify                  [BREAK: ParserError]
      → intended target generated_at + 24h
      → CoinGecko USD samples + Binance USDT hourly closes
      → average over time per provider → mean of two averages if within 1%
      → write this TWAP into actual_price_24h
      → compute endpoint errors against that temporal average [WRONG TARGET]
      → overwrite entire CSV → regenerate reports/bitcoin_report.md
  → git commit and push                    [SKIPPED after current failure]

evaluation/trading/*                       [library exercised by tests only]
  signal → position size → costs → simulated trades → metrics
  NO production caller, CSV adapter, persisted trades, or scheduled paper broker

reports/index.html                        [separate, ignored by Git]
  eight hardcoded rows → five artificial H24 paths → JS analytics/trading → charts
  NO CSV read, backend API, report JSON, or scheduled generation

chat_app/app.py                            [separate interactive demonstration]
  Yahoo Finance daily OHLC → TimesFM → response JSON
  NO experiment persistence or verification; missing static/index.html
```

The requested Market Data → Prediction → Verification → Trading → Metrics → Dashboard chain **does not exist as a connected working system**. The Markdown report is connected to CSV data; the HTML dashboard and Python trading engine are not connected to each other or to the scheduled experiment.

### Repository coverage and active files

The inventory contains all **166 tracked files**, plus the ignored HTML dashboard, with SHA-256 hashes in [repository-inventory.json](repository-inventory.json). All **104 Python files** parsed successfully with `ast.parse`.

| Files / directory | Role and audit treatment |
|---|---|
| `.github/workflows/bitcoin-evaluation.yml` | Actual live entry points, installation, experiment window, persistence and scheduling reviewed. |
| `evaluation/config.py`, `run_predict.py`, `predictor.py`, `path_metrics.py` | Prediction contract, array construction, model configuration, output mapping and serialization reviewed. |
| `evaluation/coingecko.py`, `binance.py`, `price_verifier.py`, `run_verify.py` | Source identity, timestamps, HTTP behavior, target semantics and verification reviewed. |
| `evaluation/csv_manager.py`, `metrics.py`, `report.py` | Storage, legacy behavior, calculated errors and Markdown provenance reviewed. |
| `evaluation/trading/{types,signal,equity,metrics,backtest_engine,baselines}.py`, both `__init__.py` files | Entire trading call chain and its absence from production reviewed. |
| `data/bitcoin_predictions.csv`, `reports/bitcoin_report.md` | Read-only row audit, Git history checks and arithmetic reconciliation. |
| `reports/index.html` | Ignored local artifact; full embedded analytics inspected and executed with a stub DOM. No browser visual QA claimed. |
| `chat_app/app.py` | Separate pipeline, calendar, API outputs and Flask root route checked. |
| `src/timesfm3/*` | Actual runtime package; relevant input/output, normalization, decode and checkpoint-loading internals reviewed; all 42 small-model/library tests run. |
| `src/timesfm/*` | TimesFM 2.5 library, not the BTC predictor; lightweight utility tests run. |
| `v1/*` | Archived legacy models, tests, notebooks and experiments inventoried; not reachable from BTC workflow. Not a line-by-line review of all archived research code. |
| `timesfm3-usage/*` | Benchmark/notebook examples, no BTC production call path. |
| `timesfm-forecasting/*` | Separate skill, CLI and example outputs; not evidence that BTC forecasting is valid. Skill contradictions noted below. |
| `tests/*`, `pyproject.toml`, `requirements.txt`, build/publish workflows, README | Tests, dependency drift, packaging and documentation checked. No live deployment or package publish performed. |

The unused `coingecko.get_btc_price_at`, `binance.get_btc_price_at` and `binance.get_btc_daily_close` functions are not called by the current verifier. Their existence does not prove endpoint verification. Trading APIs are called by tests and by other trading functions, but not by the scheduled pipeline. `VerifiedPrice.price_high`, `price_low`, `num_hourly_points` and `discrepancy_pct` are calculated and then discarded by `run_verify`.

## 2. Audit findings

Status meanings: **FAILED** = demonstrated wrong behavior; **PARTIAL** = some implementation exists but required guarantees are absent; **MISSING** = not connected/implemented; **VERIFIED LIMITED** = the explicitly described narrow claim passed. Severity ranks system impact, not implementation effort.

| ID | Component | Status | Evidence | Problems | Severity | Required Fix |
|---|---|---|---|---|---|---|
| F01 | Live CSV compatibility | FAILED | `csv_manager.py:60-72`; CI run 36228960065; `legacy_append` probe | Appends 38 columns beneath 16-column header; verifier raises `Expected 16 fields in line 14, saw 38`; commit step never runs. | CRITICAL | Schema-aware atomic append/migration or separate versioned ledger; preserve legacy values; test legacy file → new prediction → verification. |
| F02 | Forecast target / actual price | FAILED | `predictor.py:94`; `price_verifier.py:40-51,97-109,147-167`; `run_verify.py:75,92-112` | Endpoint prediction compared to temporal average, then reused as simulated exit. Synthetic endpoint 123 becomes actual 111.5. | CRITICAL | Persist endpoint observations at explicit T+h separately from TWAP and path extrema; never treat a TWAP as an executable endpoint price. |
| F03 | HTML dashboard provenance | FAILED | `reports/index.html:242-252,377-390`; actual JS execution | Eight hardcoded demo rows; five invented H24 paths; synthetic paths disagree with their own endpoints. No backend connection. | CRITICAL | Replace data source with validated Python report JSON; remove demo fallback; show unavailable/invalid states honestly. |
| F04 | Forecast origin and cutoff | FAILED | `run_predict.py:62-76,105-107`; live API sample; `cutoff_and_freshness` probe | Timestamp-free model input; no regular-grid/cutoff/freshness checks; future and 10-day-old history accepted. `generated_at` is assigned after inference, unrelated to last sample time. | CRITICAL | Central market-data validator, explicit information cutoff and forecast origin, regular UTC grid, point timestamps and `LookAheadBiasError`. |
| F05 | Portfolio temporal ordering | FAILED | `backtest_engine.py:104-182`; `future_equity_position_size` probe | First trade's next-day realized gain sizes a second trade entered one hour later: $1,000 becomes $1,020 before profit exists. | CRITICAL | Event-ordered fills/closures and cash reservations; distinguish independent hypothetical trades from an investable portfolio. |
| F06 | Binance candle boundaries | FAILED | `binance.py:126-132,210-236`; `binance_window` probe; live unclosed candle | Associates close with open timestamp, ignores close_time, permits 25 candles for 24h. Target-time candle may close an hour later. Endpoint helper also chooses nearest open then uses its close. | HIGH | Preserve OHLCV and open/close/availability times; reject incomplete and out-of-window bars; target last eligible close with explicit tolerance. |
| F07 | Source identity and independent verification | PARTIAL | `binance.py:24,217`; `coingecko.py:133-145`; CSV; successful CI log | USD aggregate and USDT pair averaged without conversion metadata. All 10 verified rows contain only CoinGecko; Binance returned HTTP 451 in CI. | HIGH | Same-quote sources or explicit measured conversion; persist provider/venue/pair/quote, exact source timestamps and values. Validate availability in runner environment. |
| F08 | Market/API quality | FAILED | `coingecko.py:113-145,226-240`; `price_verifier.py:147-167`; invalid/incomplete probes | Zero, negative, NaN, infinity, duplicate and future values accepted. One sample per source earns HIGH confidence. No missing-candle or staleness gate. | HIGH | Validate schema, positivity, finiteness, ordering, spacing, completeness, freshness, divergence and minimum source coverage; reject or quarantine. |
| F09 | Prediction immutability and identifiers | PARTIAL | `csv_manager.py:67-72,115-141`; schema; Git inspection | No prediction ID, input/forecast hashes, append-only enforcement, uniqueness or tamper checks; timestamp alone is identity. Verification rewrites full file and can overwrite an existing result. | HIGH | Immutable prediction records plus separate versioned verification records; transactional writes, unique IDs and content hashes. |
| F10 | Reproducibility | MISSING | `predictor.py:68-76`; schema; `pyproject.toml`; workflow line 48 | Input snapshots absent; checkpoint revision follows main; package/config/git/device/seed not recorded; precision rounded in CSV. | HIGH | Store exact model input, canonical hash, model revision/checksum, code SHA, resolved dependencies/config, device and deterministic settings. Document tolerance across hardware. |
| F11 | Historical experiment semantics | FAILED | `git show 3f6782c^:evaluation/predictor.py:74`; current CSV; `run_verify.py:75`; `report.py:42-75` | All 12 stored rows are single-step legacy; still judged at +24h TWAP. New report groups by method, not experiment/horizon. HTML relabels five old rows H24. | HIGH | Preserve legacy rows as legacy with explicit invalid target semantics; separate experiments; never regenerate old forecasts or promote them to live H24. |
| F12 | Trading/report integration | MISSING | Workflow lines 74-99; caller search; `report.py`; absent `reports/data.json` | No production CSV→PredictionRecord adapter, signal/trade persistence, broker loop or backend report artifact. | HIGH | Connect only validated forecasts/results to versioned analytics and paper-trading pipeline; add full E2E reconciliation. |
| F13 | Duplicate prediction IDs | FAILED | `backtest_engine.py:97,109-123`; `tests/test_trading_engine.py:635-643`; probe | Lookup keeps last duplicate; first trade uses second outcome (90 instead of 110). Existing test checks only trade count. | HIGH | Enforce identity uniqueness before simulation/persistence and test actual matched prices. |
| F14 | Timezones / overlap | FAILED | `signal.py:92-126`; `backtest_engine.py:51-60`; timezone probe | Single-position mode compares timestamp strings. 13:00+02:00 is accepted after a 12:00Z deadline even though it is 11:00Z. Naive values accepted; formatting may attach Z without conversion. | HIGH | Parse/validate aware timestamps, normalize UTC before comparison/addition, test CET/CEST and both DST transitions. |
| F15 | Execution and risk model | PARTIAL | `metrics.py:72-93`; `equity.py:13-26`; `backtest_engine.py:125-160`; types | Costs are subtracted, but both fees use entry notional; maker fee and risk_per_trade_pct unused. No fill model, funding/borrow policy, capital reservation, or reversal mode. TP/SL requires unavailable true highs/lows. | HIGH | Explicit instrument/fill/accounting contract; per-leg notionals, costs, conditional funding/borrow, event order and OHLC ambiguity policy. |
| F16 | Zero-threshold / Naive strategy | FAILED | `signal.py:55-58`; `baselines.py:238-256`; probe | Flat forecast at threshold 0 opens LONG. Naive comparison becomes an always-long strategy despite description. | HIGH | Define FLAT/NO_TRADE independently of threshold; separately named Always Long/Short benchmarks. |
| F17 | Forecast shape / validity | PARTIAL | `path_metrics.py:44-67`; mutable `PredictionResult`; nonfinite probe | Index mapping and length check exist, but no finite/positive/quantile-order validation or invalid status; NaN path produces NaN metrics. | HIGH | Validated immutable common ForecastResult, complete time axes, output sanity gates and status exclusion. |
| F18 | Statistical conclusions | FAILED | `trading/metrics.py:183-249`; `report.py:241-265`; HTML line 473 | n=2 produces Sharpe 40.53 and Sortino infinity; no confidence intervals, OOS split, dependence-aware inference or minimum-sample policy. HTML calls ≥30 trades Reliable without an edge test. | HIGH | Missing/insufficient-sample states, explicit pre-registered inference criteria, temporal holdouts and dependence-aware uncertainty; no profitability status. |
| F19 | Forecast-model comparison | MISSING | `predictor.py`; `trading/baselines.py` | ARIMA is a fallback, not a concurrent baseline. Naive/Momentum are trading experiments, not forecasting providers. Drift, MA, ETS and ForecastProvider interface absent. | HIGH | Same-cutoff independent forecasts with common result contract, paired per-horizon errors and frozen model settings. |
| F20 | Operational failure visibility | FAILED | `run_verify.py:94-99,129-151`; CI log; workflow lines 74-86 | All verification fetches can fail and exit 0. Logging uses invalid `$%,.2f`. Prediction failure prevents verification; experiment expiry also prevents draining pending results. | HIGH | Persist health/failures, accurate exit state, separate generation/verification jobs, verify matured predictions after collection ends. |
| F21 | Durable storage / concurrent runs | PARTIAL | `csv_manager.py:70,141`; workflow lacks concurrency control | Non-atomic rewrite, no lock/backup, no transaction; concurrent runs may conflict or lose updates. Failed runner has no artifact rescue. | HIGH | Single writer/concurrency control, atomic durable writes and immutable backup/artifact before risky later stages. |
| F22 | Model fallback provenance | PARTIAL | `run_predict.py:81-92`; `predictor.py:167-169` | Actual method is correctly distinct, but requested_model, fallback_used/reason and model version are absent; local ARIMA dependency is missing. | MEDIUM | Preserve requested/actual model and reason, validate fallback output, keep experiment and metrics separate. |
| F23 | Path/range analysis and uncertainty | PARTIAL | `path_metrics.py`; verifier fields; CSV | Predicted path extrema/slope exist; actual paths not stored; no shape classifier or path errors. q10/q90 are nominal 80% pointwise bounds, not path extrema or 95% confidence. | MEDIUM | Persist full actual path/OHLC; define deterministic shapes; evaluate per-horizon errors/coverage; avoid unsupported 50/90/95% coverage claims. |
| F24 | Monte Carlo and baselines | PARTIAL | `baselines.py:124-207,355-389` | 1,000 seeded runs exist, but only means retained; no median/5th/95th percentiles. All inherit engine bias; B&H allocates 100%, strategy defaults to 10%. | MEDIUM | Fix engine first; return distributions; report matched exposure/risk and Always Long/Short; validate comparable sample windows. |
| F25 | Separate chat application | FAILED | `chat_app/app.py:131-183,275-277,295`; Flask/calendar probes | Friday's next BTC date becomes Monday; naive date use; q20/q80 named q25/q75; root returns 404 because static HTML is missing. | HIGH | Either explicitly retire/exclude demo or fix calendar/output/routes separately; do not present as experiment UI. |
| F26 | Skill / docs / test entry points | FAILED | Skill lines 58-59,458-474; checker lines 223-258; README; test results | Skill targets 2.5, mislabels q10–q90 as 90% in one section, and preflight passes low available RAM based on total RAM. README claims ARIMA CI but workflow runs TimesFM. Recommended v1 tests do not match installed API. | MEDIUM | Version-specific docs and requirements, honest memory preflight, current test commands and CI correctness gate. |

## 3. Actual runtime and TimesFM contract

| Property | Verified evidence and conclusion |
|---|---|
| Installed distribution | Local `timesfm==3.0.1`, Python 3.13.1, torch 2.14.0, NumPy 2.4.6, pandas 3.0.5. `timesfm3` is packaged inside the `timesfm` distribution, not a separate distribution. |
| Import path | `/Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/timesfm3/`. All 16 Python files match `src/timesfm3` byte-for-byte. Local imports are installed copies, not editable source. |
| CI runtime | Python 3.11; successful run 36163764194 installed timesfm 3.0.1, torch 2.14.0, pandas 3.0.6, statsmodels 0.15.0. Dependency resolution differs from local and ignores `requirements.txt`. |
| Actual checkpoint | `google/timesfm-3.0-pytorch`; cached and CI-resolved revision `43046b85ec22d584a13f8098c2ed39c889e129c2`. Cached safetensors size 1,322,898,824 bytes. Application does not pin/record revision. |
| Actual execution | CI 36163764194 logged model loading and $83,819.64 output matching stored row `2026-09-25T16:56:59Z`; no ARIMA fallback in that run. CI 36228960065 generated $84,070.94 at 08:12:32Z but lost durable publication at CSV parsing. |
| Input variable | Raw CoinGecko `prices` observations in USD. **Not OHLC close, not average candle price, not returns/log returns.** Only price array survives; API timestamp, volume and market cap are discarded. |
| Context | Current application uses last min(512, n) samples. At strictly hourly cadence 512 samples cover approximately 21.3 days of observations (511 hours between first/last timestamps). Requests 90 days; minimum accepted 30 samples. ARIMA uses last 168 samples. |
| Frequency | There is **no hourly frequency argument** in TimesFM3Evaluator. It receives arrays and horizon counts; time units come exclusively from regular input sampling and application metadata. |
| Horizon | Current adapter requests 24. `model.py:619-626` extracts forecasts after the context; `model.py:638` explicitly indexes future positions 1..horizon; forecaster returns `raw[:horizon]` at lines 750-757. The pre-change adapter requested **1** with no daily aggregation. |
| Index mapping | Adapter probe and `tests/test_forecast_path.py:8-39` confirm element 0→first point, 3→fourth, 7→eighth, 11→twelfth, 23→24th. They do **not** prove wall-clock T+1h..T+24h for existing irregular inputs. |
| Batch shape | One 1D series is passed as `[context]`; per-core batch size 4. Per-series output is `(24,)`, quantiles `(24,9)` for configured q0.1..q0.9. Forecaster internally rounds decoding to 64 then slices to requested horizon. |
| Point / quantiles | Point forecast is median index 4 in TimesFM 3.0; bounds use quantile indices 0 and 8. Correct for this checkpoint. TimesFM 2.5's 10-channel indexing is different. |
| Normalization | Application casts raw values only. Evaluator defaults `use_znorm=False`, `padding_mode=none`, sorting and nonnegative clamp enabled; symmetric averaging explicitly true. Model config enables linear detrending, iterative CPM RevIN and identity input transform. `model.py:236,344,637-645` normalizes internally, reverses normalization, and adds trend back. No extra application scaling/inverse transform exists. |
| Missing values | Forecaster strips leading NaNs, interpolates internal NaNs, and converts all-NaN input to zeros (`timesfm3_forecaster.py:518-536`). This is library behavior, not a suitable market-data validation policy. It must be blocked upstream for this experiment. |
| Determinism | Inference mode/eval are present. App sets no seed or deterministic algorithm policy, stores no exact inputs/device/settings. Reproducing historical predictions is impossible from current CSV alone. Cross-device bitwise identity is not established. |
| Interval meaning | q0.1–q0.9 has nominal central mass 80%, pointwise and uncalibrated on this BTC experiment. It is not a 95% confidence interval or a guaranteed 24h min/max range. 50/90/95% bands require unavailable quantiles or an explicitly tested interpolation/calibration method. |

**Conditional time statement:** if the last input represents a fully observed point at UTC time T, and the entire input is a validated one-hour grid, then forecast[0] corresponds to T+1h and forecast[23] to T+24h. Those preconditions are not enforced here. The API sample had a final 2,250-second interval, and generated_at differs from the last sample time. Therefore the requested wall-clock mapping is **not verified for this pipeline**.

No full pretrained inference was rerun locally during this audit. The mandatory skill checker reported 24 GB total but only 1.7 GB available, yet passed a **2.5** profile based on total RAM. The skill states “blocks if below 2 GB” ([local instruction](../../timesfm-forecasting/SKILL.md)); the checker does not implement that stated safeguard. Given the larger 3.0 checkpoint, no additional full-model allocation was justified. Small test models and a mocked adapter were used; actual pretrained execution is supported by the CI logs, not by claiming a new local inference.

The [official TimesFM 3.0 model card](https://huggingface.co/google/timesfm-3.0-pytorch) uses a non-commercial model license. The repository README also states non-production restrictions. A future production/commercial deployment requires checking the actual weight license; changing to 2.5 is an alternative experiment, not an automatic substitution. No model/license migration was performed.

## 4. Market data, time and leakage evidence

CoinGecko returned HTTP 200 for the same 90-day history request used by the application: **2,161 samples**, from `2026-06-28T11:00:00Z` to `2026-09-26T10:37:30Z`. The final intervals were `3600,3600,3600,3600,3600,2250` seconds. Last sample age at request start was 82.70 seconds; no duplicates or future timestamps were observed in this particular response. Its SHA-256 was `320a0a69c6f2b20e27477d5896c02f116325e4b629eac84adf600dda9b9ac572` (response body was not retained). A fresh sample does not make its grid regular or prove prior runs were fresh. [CoinGecko's documentation](https://docs.coingecko.com/reference/coins-id-market-chart) describes automatic hourly granularity for 2–90 days; it does not replace local timestamp validation.

Binance's recent three-candle request returned HTTP 200 locally. The 10:00 UTC candle had close_time `10:59:59.999Z`, still in the future at the 10:38 audit request. The API's current `close` is therefore a provisional value. Application parsing discards this distinction. [Binance's documented kline schema](https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints) distinguishes open time at index 0, close at index 4, and close time at index 6. In CI, Binance repeatedly returned HTTP 451; no restriction bypass was attempted.

The diagnostic Binance fixture demonstrates that a 24h window can retain 25 close values and label the last value at target time although its actual candle closes almost an hour later. This is **verification target contamination**, not evidence that Binance enters the TimesFM input (it does not). A partial candle seen at 15:43 contains information available at 15:43; its final 16:00 close, reconstructed later and presented as known at 15:43, would be look-ahead. Both availability time and candle close time must be explicit.

The pipeline accepted synthetic history ending five days in the future and stale history ending 16 September, called the model, and attempted to save both. This proves the absence of a guard; it does not prove that all historical live predictions used future observations. Historical absence of leakage is **unverifiable** because inputs were not saved.

The current arithmetic mean is a TWAP approximation only for equally spaced observations with a defined boundary convention. Missing or irregular samples require explicit temporal weights or rejection. A cross-source median must use observations of the same quote and target interval. With only two sources its usual definition equals their mean; it cannot identify which source is anomalous. Preserve both exact values and mark disagreement instead of implying robustness. An independent third source can improve outlier resistance, but aggregate services may incorporate exchange feeds and are not automatically independent venues.

As zero-subscription alternatives, public Coinbase and Kraken BTC/USD OHLC requests both returned HTTP 200 locally. For the completed 09:00–10:00 UTC candle, Coinbase close was $84,012.95 and Kraken close $84,002.40. These are distinct venue closes with the same quote and interval, not identical spot ticks. [Coinbase](https://docs.cdp.coinbase.com/api-reference/exchange-api/rest-api/products/get-product-candles) documents OHLCV, up to 300 candles per request and possible gaps; [Kraken](https://docs.kraken.com/api-reference/market-data/get-ohlc-data) documents its OHLC endpoint. Their GitHub runner reachability and operating limits remain to be validated. They are candidate sources, not installed providers or verified production fallbacks.

## 5. Persistence and historical integrity

- Current CSV: **12 rows, 16 columns, all method=timesfm, 10 verified, 2 pending, zero forecast paths, zero explicit experiment versions, zero Binance actuals**. Current code expects 38 columns. Loader assigns legacy_h1/horizon=1 in memory.
- Git history confirms those 12 rows were produced before the 26 September horizon change. They cannot be treated as live H24 forecasts. No full path can be reconstructed honestly from their endpoint.
- Six original forecast fields (initial, predicted endpoint, lower, upper, direction, method) were compared across available Git revisions for all 12 current timestamps: **no changed values detected**. This is limited observed preservation, not append-only enforcement, independent timestamp attestation, or proof of exact historical inputs.
- History contains 26 unique prediction timestamps; **14 older records are absent from the current CSV**, consistent with reset commit `bdfb1ea83c237998d267dc5c8f8824d0e9ff2fa8`. They remain in Git. No restoration or deletion was performed.
- Updating a copied verified row changes its actual from 76,207.3635099679 to 1.0 and rewrites 16 columns into 38, adding legacy metadata. Existing forecasts are not regenerated by this function, but results are mutable and the whole file is rewritten non-atomically.
- Rounding predictions/path values before persistence loses exact output precision. Neither input nor forecast hash, model revision, generation duration, observation retrieval time, verification time or Git commit is stored.

Original files remained byte-identical throughout the audit:

| File | SHA-256 |
|---|---|
| `data/bitcoin_predictions.csv` | `6871717a47f9e26c203f6b3ecb0c25f674de36eeb3db781f04cd032cdcda1e50` |
| `reports/bitcoin_report.md` | `9f68bd25fc5b4b29dedcd8e78101b236183ffd9bce0d5df79a368e40362e2d0d` |
| `reports/index.html` | `689ddce1ba109b5a3c86bd66c3f77fabca25de349b9443345d7f3c7e85e0a97c` |

## 6. Reconciliation: real stored row versus dashboard

Selected actual CSV row: **2026-09-19T15:43:25Z**, which the HTML mislabels as H24 at 15:00Z.

| Quantity | Recomputed value |
|---|---:|
| Initial recorded price | $81,654.00 |
| Predicted single-step endpoint | $81,623.90 |
| Stored actual (TWAP, not endpoint) | $80,824.83227010316 |
| Predicted return | −0.0368628603620% |
| Return to stored TWAP | −1.0154649250457% |
| Python direction at threshold 0 | SHORT |
| Python starting capital / position | $10,000 / $1,000 |
| Python hypothetical gross PnL | $10.1546492504573 |
| Python entry + exit fees | $2.00 |
| Python spread / slippage | $0.50 / $1.00 |
| Python hypothetical net PnL | $6.6546492504573 |
| HTML signal-log displayed net PnL | **$66.55** |

Python applies `position × signed_return − costs`, with default round-trip cost 0.35%. HTML signal log instead allocates $10,000; its strategy simulation allocates 10% of capital. It therefore disagrees by approximately **10× with itself and Python**. This is a reconciliation of implemented arithmetic using a bad target, **not a valid trade return**.

The real embedded HTML script was executed in Node using a stub DOM/Chart renderer: it reported 8 rows, 5 H24 rows and 3 legacy rows. All five artificial paths end at values different from their claimed endpoint; the 20 September row's path even rises while its endpoint forecast falls. This execution does not constitute visual browser QA, which is unnecessary to establish the demonstrated data disconnection.

The Markdown report **does reconcile numerically with the CSV**: MAE $551.11, RMSE $690.04, direction 70%, range coverage 40%. Stored row error/percentage fields agree within their rounding precision. The calculations operate on incorrect target semantics, so numerical consistency does not make them valid forecasting evidence.

Diagnostic comparison on the same ten legacy rows: naive MAE $564.29859 versus TimesFM $551.10959, both measured against the wrong TWAP target. Across twelve rows, mean absolute predicted return is only 0.0407591%. These numbers explain why naive behavior must be tested, but cannot distinguish a useful signal from persistence. Seven adjacent prediction pairs overlap within 24h; treating outcomes as independent exaggerates sample size. ARIMA/ETS/etc. cannot be fairly replayed against the original live information set because no input snapshots exist.

## 7. Verification executed

| Check | Result | What it establishes / does not establish |
|---|---|---|
| Seven BTC test files | **128 passed, 1 failed** | Failure `TestLoadPredictions.test_loads_existing_data`: old positional fixture under new header maps 49,000 into initial_price. Separate copy probe proves real legacy append corruption. |
| Base/config/torch utility tests | **58 passed** | Existing 2.5 utility/layer behavior; not BTC or 3.0 checkpoint performance. |
| `PYTHONPATH=src:. pytest src/timesfm3` | **42 passed** | Small-model 3.0 decode, preprocessing and local-load tests; no pretrained forecasting validation. |
| Total distinct tests executed | **228 passed, 1 failed** | No complete green gate. |
| Broad collection attempt | **215 collected, 7 errors initially** | Five installed/source module mismatches resolved by source PYTHONPATH. Two remaining environment/API issues: absent `einshape` for Flax model-loading tests; archived `timesfm.data_loader` unavailable under 3.0. |
| Full 2.5 model tests | **Not run** | Large random models/checkpoints unnecessary for BTC audit with low available RAM. |
| Temporary characterization probes | **All completed** | Reproduced CSV failure, wrong target, invalid data acceptance, future/stale acceptance, adapter index mapping, future-equity sizing, duplicate exit mapping, timezone overlap, flat LONG, small-sample extremes, zero exit on verification failure, weekend calendar and Flask 404. |
| Syntax | **104 Python files passed** | Parsing only, not type-check/lint. |
| Local lint/type-check/build | **Not run** | ruff, mypy and build are not installed; no application type-check configuration/gate identified. No dependencies installed during read-only audit. |
| Existing remote package build | **Success at audited SHA** | Run 36228960146 builds package only; no BTC correctness tests. |
| Latest remote live collector | **Failure at audited SHA** | Run 36228960065, real CSV ParserError. |
| Latest successful collector inspected | **Pre-change source, TimesFM executed** | Run 36163764194; checkpoint/price and Binance 451/logging failure inspected. |
| Git/file integrity | **Original files unchanged** | Hashes above and final working-tree comparison. Only audit documents added. |

Commands:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -p no:cacheprovider tests/test_btc_metrics.py tests/test_btc_csv_manager.py tests/test_btc_simulation.py tests/test_binance.py tests/test_price_verifier.py tests/test_forecast_path.py tests/test_trading_engine.py -q
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -p no:cacheprovider tests/test_base_utils.py tests/test_configs.py tests/test_torch_layers.py tests/test_torch_utils.py -q
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python3 -m pytest -p no:cacheprovider src/timesfm3 -q
gh run view 36228960065 --repo artur-fedosiuk/btc-prediction-timesfm --log-failed
gh run view 36163764194 --repo artur-fedosiuk/btc-prediction-timesfm --log
```

The exact diagnostic Python code is preserved as documentation in [reproduce-diagnostics.md](reproduce-diagnostics.md), with outputs in [diagnostic-results.json](diagnostic-results.json). These assert that defects are reproducible, not that defects have been fixed. No regression tests were added to the application before approval.

The existing so-called end-to-end test (`tests/test_btc_simulation.py:79-162`) inserts handwritten predictions, directly updates verification, and renders Markdown. It does not run market fetching, the prediction entry point, TimesFM, verifier, trading, or dashboard. This is a useful partial integration test, not the requested full end-to-end proof.

CI evidence: [failed collector](https://github.com/artur-fedosiuk/btc-prediction-timesfm/actions/runs/36228960065), [successful earlier collector](https://github.com/artur-fedosiuk/btc-prediction-timesfm/actions/runs/36163764194), [package build](https://github.com/artur-fedosiuk/btc-prediction-timesfm/actions/runs/36228960146).

## 8. Storage decision and ordered corrective proposal

The dataset size (12 current rows) does **not** justify PostgreSQL or a large migration. Relational structure, immutability and crash safety do justify planning a small SQLite store once the immediate collector defect is addressed.

| Option | Advantages | Costs / risks | Recommendation |
|---|---|---|---|
| Versioned CSV + separate append-only records and snapshots | Small immediate patch; diffable in Git; preserves current operating model. | Implement locks, schema handling, atomic replacement, uniqueness and multi-file reconciliation manually. | Suitable bounded P0 repair or transitional ledger. |
| SQLite local MVP | Python standard library; transactions, foreign keys, unique keys, immutable-record triggers; natural forecast-points/results/trades relations. | Explicit backup/restore and runner persistence needed; GitHub runner disk is ephemeral; binary Git churn; SQLite alone is not tamper-proof. | Preferred next storage step after contract tests; keep CSV export and immutable legacy import. |
| PostgreSQL / hosted database | Concurrent service writers and queries. | Deployment, access control, operational work and potential recurring cost without present need. | Defer. No external paid infrastructure. |

### First proposed implementation batch: P0–P4 foundation

**Objective:** restore durable collection and establish a trustworthy new experiment before adding models, ensembles or dashboard features.

1. **P0 correctness:** reproduce F01 as a regression; add schema-aware, atomic compatibility handling or a separate versioned prediction ledger; preserve existing legacy forecasts and old actuals; store new endpoint actuals separately from TWAP. Remove false H24 assumptions from analytics inputs. Separate prediction failure from verification scheduling.
2. **P1 integrity:** validated common result/observation records, unique prediction IDs, append-only forecast snapshots, separate verification results, hashes and backup/recovery checks. Add provenance without rewriting historical forecasts into new experiments.
3. **P2 temporal safety:** regular UTC grids, explicit source/availability/candle-close times, data_cutoff, forecast_origin and target timestamps; reject future/stale/gapped data; closed candles only. Recommendation for a **new** same-quote experiment: Coinbase BTC/USD primary OHLCV and Kraken BTC/USD cross-check, subject to runner reachability verification. Preserve CoinGecko and Binance provenance as legacy/secondary observations, never silently convert USDT to USD. Alternative: retain CoinGecko samples with an explicit causal sampling policy, acknowledging they are not OHLC candles.
4. **P3 reproducibility:** persist exact input arrays/snapshots/configuration, pin checkpoint revision, record resolved model/package/Git/device/seed metadata and fallback reason. Test repeated runs and document numeric tolerance. Prevent scientific results from depending on ignored artifacts.
5. **P4 validation:** timestamp/index/quantile tests, constant/up/down/outlier/missing/duplicate fixtures, temporal leakage tests, immutable-persistence/crash/idempotence tests, and full mocked Market→Forecast→Save→Actual→Verify→Signal→Trade→Metrics→Report chain with hand-computed expected values. Fix F05/F13/F14 where needed to make that chain meaningful. Real-provider and real-checkpoint evidence remain explicitly separate from mocks.

**Risks and tradeoffs:** changing the source/grid/target changes the experiment, so it must start a new version; some invalid observations will be rejected and collection gaps will become visible. A provider can be reachable locally but unavailable in CI. A model reload can be memory intensive. Legacy target errors cannot be repaired by inventing old forecasts. No paid API, account change or real trade is required. No automatic threshold/weight tuning.

**Acceptance before deployment:** original forecast records preserved, frozen-input replay, schema/crash tests pass, all newly stored points have coherent aware UTC times and hashes, source values retained, backend arithmetic reconciles exactly, errors visible, and independent live verification succeeds in the actual runner environment. First make changes and local results reviewable; deployment/push is a separate explicitly authorized step.

### Later batches, in required priority order

- **P5 model comparison:** providers TimesFM, Naive, Drift, Moving Average, Momentum, ARIMA and ETS; identical cutoff/target, independent saved forecasts, no opportunistic retuning. Compare price, direction and magnitude separately by horizon.
- **P6 ensemble analysis:** mean, median, trimmed mean, equal-weight weighted interface; save constituents, weights, agreement votes excluding FLAT from directional denominator, dispersion and missing-provider policy. No learned weights on current live sample.
- **P7 simulation:** versioned threshold family 0/0.02/0.05/0.10/0.20/0.30/0.50%, independent agreement/dispersion experiments, explicit ONE_POSITION/INDEPENDENT_TRADES/REVERSE_ON_SIGNAL semantics, fill/cost/funding assumptions, capital/exposure accounting, meaningful baselines and ≥1,000 seeded random simulations with quantiles. Walk-forward splits and dependence-aware bootstrap only after sufficient data; no fake significance from overlapping trades.
- **P8 reporting:** one Python analytics source → `reports/data.json` → HTML rendering. Health, provenance, per-horizon metrics, sample counts and explicit insufficient-data status. Keep historical, live-forward, legacy and fallback metrics distinct. Do not spend time on styling before correctness is established.

## 9. Coverage of all 80 requested requirements

This matrix distinguishes current implementation from required future work. “Partial” never means a completion criterion passed.

| Request | State | Evidence / required action |
|---:|---|---|
| 1 Repository/pipeline audit | Investigated | Section 1 and inventory; chain is broken/disconnected. |
| 2 Audit report | Delivered | Section 2 has Component, Status, Evidence, Problems, Severity, Required Fix. |
| 3 TimesFM verification/tests | Partial | Section 3; index contract checked; temporal mapping and pretrained replay not certified. |
| 4 Compare naive/drift/MA/momentum/ARIMA/ETS | Missing | F19; only invalid-target diagnostic naive comparison performed. |
| 5 ForecastProvider | Missing | Two functions; no interchangeable provider interface. |
| 6 Standard ForecastResult | Partial | PredictionResult lacks timestamps, version/hash validation and rich provenance. |
| 7 Ensemble mean/median/trimmed | Missing | P6 proposal only. |
| 8 Weighted/equal ensemble | Missing | No implementation; do not fit weights. |
| 9 Model agreement | Missing | No independent concurrent forecasts. |
| 10 Model dispersion | Missing | Path volatility is not cross-model dispersion. |
| 11 Independent market sources | Partial | Two clients exist, single source in every saved verification; two alternatives probed locally. |
| 12 Symbol/quote identity | Failed | F07: USD/USDT combined without explicit persisted identity/conversion. |
| 13 Multi-source target actuals | Missing | Only TWAP sources saved; no T+1/4/8/12/24 observations. |
| 14 Cross-source vs temporal aggregate | Failed | F02; nested TWAP means used as target endpoint. |
| 15 Actual horizons/TWAP/extrema | Missing | Only mislabeled actual_price_24h persisted. |
| 16 OHLCV | Missing in evaluation | Binance fields discarded; separate chat OHLC not persisted. |
| 17 UTC/CET/CEST/DST | Partial/failed | Stored UTC Z strings; no enforcement; offset overlap probe fails. |
| 18 Closed-candle alignment | Failed | F04/F06; incomplete bars and irregular tail not excluded. |
| 19 Central look-ahead guard | Missing | Future history accepted; portfolio look-ahead reproduced. |
| 20 Prediction immutability | Partial | Observed six fields unchanged in Git, but enforcement/provenance absent. |
| 21 Predictions vs results separation | Missing | Same CSV rewritten by verification. |
| 22 Input hash | Missing | No hash field or snapshot. |
| 23 Reproducibility | Missing | Historical input unavailable; no manifest. |
| 24 API validation | Partial/failed | Timeout/retry exist; invalid prices accepted. |
| 25 MarketDataValidator | Missing | No central quality boundary. |
| 26 Source consensus | Partial | 1% TWAP discrepancy, LOW label; not persisted as discrepancy nor excluded from metrics. |
| 27 Forecast path analysis | Partial | Current code computes endpoints/extrema/slope/std; no saved path yet. |
| 28 Path shape | Missing | No deterministic classifier. |
| 29 Actual path vs forecast | Missing | Actual sequence discarded. |
| 30 Horizon performance table | Missing | HTML rows are placeholders, no hourly actuals. |
| 31 Pearson/Spearman | Failed/missing | JS Pearson on demo data only; Spearman absent. |
| 32 Price/direction/magnitude | Partial | Absolute errors/direction exist with invalid target; magnitude analysis absent. |
| 33 Magnitude calibration | Missing | No analysis; no live calibration should be added yet. |
| 34 Uncertainty/coverage | Partial | Correct q10/q90 indexing, nominal 80%; stored coverage uses wrong target. |
| 35 Separate strategy/risk/execution | Partial | Python modularity exists, but engine not integrated and timing wrong. |
| 36 Parallel threshold family | Partial | Python defaults omit 0/0.02/0.05; HTML omits 0.20/0.50 and uses demo data. |
| 37 Agreement strategies | Missing | P6/P7 experimental versions required. |
| 38 Dispersion filter | Missing | Must remain separate from primary strategy. |
| 39 Realistic costs | Partial | Fees/spread/slippage deducted; maker unused; no funding/borrow instrument contract. |
| 40 Execution prices | Missing | Recorded spot/TWAP passed through without fills. |
| 41 Overlap modes | Partial/failed | single and independent exist; string comparison/equity bias; reversal absent. |
| 42 Trading baselines | Partial | B&H/random/naive/momentum library only; always long/short absent; comparability issues. |
| 43 1,000+ Monte Carlo | Partial | 1,000 seeded default, mean only, biased engine, not reported. |
| 44 Statistical significance | Missing | No uncertainty/effective sample/explicit hypothesis criterion. |
| 45 Bootstrap | Missing/defer | No sufficient valid sample; avoid misleading implementation now. |
| 46 Walk-forward | Missing | No temporal train/validation/test/live protocol. |
| 47 Historical/live separation | Missing | No durable run-type/schema; legacy mixed in report. |
| 48 Preserve legacy horizon | Partial | Loader labels legacy; HTML falsely assigns H24; old forecasts unchanged during audit. |
| 49 System health | Missing | No health artifact/dashboard; CI failures need direct inspection. |
| 50 Data provenance | Missing | IDs/cutoff/model revision/input source/time/verification time absent. |
| 51 Fallback transparency | Partial | Method/experiment distinct; requested model and fallback reason absent. |
| 52 Visible failures | Partial/failed | Logging exists; verifier exits 0 on fetch failure; invalid format; signal parsing silently passes. |
| 53 Structured logging | Partial | Text timestamps/module names; no prediction ID/experiment context in structured records. |
| 54 Synthetic datasets | Partial | Existing arithmetic fixtures; audit adversarial probes; full pipeline fixture suite still required. |
| 55 Full E2E | Missing | Existing simulation skips actual entry points, provider, model, trade and HTML. |
| 56 Real reconciliation | Executed, failed | Section 6: Markdown consistent; HTML signal log differs ~10×. |
| 57 Single financial source | Failed | JS and Python duplicate formulas; disconnected inputs. |
| 58 reports/data.json | Missing | No backend analytics artifact. |
| 59 Types/validation | Partial | Dataclasses lack domain validation; prices/time/hash/finite checks absent. |
| 60 Automatic versioning | Partial | Experiment/horizon in new output; strategy config version; no model/schema/git persistence. |
| 61 CSV compatibility/atomicity | Failed | F01/F21; reproducible current live failure. |
| 62 Professional storage | Assessed only | Section 8: no large migration now; SQLite appropriate future bounded MVP. |
| 63 Metrics suite | Partial | Prediction basic metrics; trading metrics in disconnected library; missing correlation/sMAPE/valid coverage. |
| 64 Small-sample ratios | Failed | Sharpe 40.53 and Sortino infinity with two synthetic trades. |
| 65 Automated sanity checks | Missing | Length check only; NaN path accepted. |
| 66 Invalid prediction states | Missing | No central status/exclusion contract. |
| 67 Freshness gate | Missing | Old synthetic history reaches model and save. |
| 68 History length experiments | Missing | Fixed 512/168 samples; no 7/14/30/60-day versions. |
| 69 Price vs returns | Assessed | Raw price currently; no automatic transform change; future independent experiments. |
| 70 Scaling | Source-verified | Raw app input → internal RevIN/detrend → inverse/trend restore; no double app scaling found. |
| 71 Volume/volatility context | Missing in ledger | Source volume discarded; no context feature snapshots. |
| 72 Market regimes | Missing | No analytical labels/statistics; defer signal conditioning. |
| 73 Crypto 24/7 | Partial/failed | Evaluation schedule 7 days; separate chat skips weekends. |
| 74 Retry/failure/fallback | Partial | Timeout/backoff exist; 4xx retried despite comment; last sleep unnecessary; source/model fallback provenance incomplete. |
| 75 Exact and robust values | Failed | Saves provider TWAPs only; no exact endpoint timestamps/raw paths or same-time median. |
| 76 No premature tuning | Preserved during audit | No parameters selected/optimized. Future parallel experiment protocol needed. |
| 77 Live continuity | Failed before audit | Current SHA already fails; no interference introduced by audit. |
| 78 Completion criteria | **Not met** | Market alignment, immutability, target verification, integrated trading/dashboard and E2E remain open. |
| 79 No profitability claim | Partial | Markdown cautious; HTML ≥30-trade Reliable unsupported; no explicit edge criterion. Audit makes no profitability claim. |
| 80 Verification report | Delivered for investigation | Section 10; fixes and end-to-end completion explicitly outstanding. |

## 10. Requested final verification report

1. **Architecture discovered:** three separate pipelines plus a disconnected trading library; Section 1 shows the actual call chain and breakpoints.
2. **Problems found:** 26 evidence-backed findings in Section 2, including five CRITICAL findings.
3. **Critical bugs fixed:** **none in this pre-change audit**. Fixes require the approval requested in the supplied collaboration instructions.
4. **Files changed:** only new files under `reports/audit-2026-09-26/`: this report, repository inventory, diagnostic results, reproducibility appendix. Existing source/data/configuration unchanged.
5. **Tests added:** no committed regression tests yet; temporary audit characterization probes preserved in the appendix.
6. **Tests passed:** 228; one existing BTC test failed; full suite not green, with environment/legacy exclusions documented in Section 7.
7. **TimesFM configuration verified:** installed source matches repository; actual 3.0 checkpoint and representative execution confirmed in CI; configuration and normalization documented. No historical exact replay possible.
8. **Data sources verified:** live CoinGecko/Binance calls locally; all stored results only CoinGecko, Binance 451 in inspected CI run. Coinbase/Kraken candidate calls locally successful, not integrated or CI-verified.
9. **Horizon mapping verified:** array index semantics confirmed; application **wall-clock mapping not verified**. Current stored records are legacy one-step, no full paths.
10. **Look-ahead protection verified:** **failed/absent**; future-data acceptance and future-equity sizing reproduced.
11. **Forecast persistence verified:** legacy values observed unchanged, but live schema append broken, no immutable enforcement/snapshots/hashes.
12. **Dashboard reconciliation verified:** **failed**; demo data, synthetic paths and conflicting PnL. Markdown arithmetic agrees with CSV but evaluates the wrong target.
13. **Remaining risks:** unknown historical inputs; lost failed-run path; provider availability/quote differences; lack of temporal guards, transactionality, reproducibility, statistically valid samples and instrument execution model. All remain unresolved until corrective work is approved and verified.
14. **Recommended next experiment:** after P0–P4 pass, start a new frozen UTC hourly-close forward experiment with independent same-cutoff TimesFM/Naive/ARIMA/ETS/Momentum forecasts (plus Drift/MA), fixed settings, exact same-quote source observations, per-horizon verification and no tuning. Keep the original 12 rows and legacy metrics quarantined from new results. Current scientific status: **INSUFFICIENT VALID DATA / STATISTICALLY INCONCLUSIVE**.

Approval requested for the concrete first batch in Section 8. The reason is the supplied AGENTS.md collaboration instruction: “prima di applicare qualsiasi modifica a codice, dati, configurazioni, servizi esterni o produzione, chiedimi approvazione.” The audit and reviewable corrective proposal are now available; no skill-imposed approval requirement is being inferred.
