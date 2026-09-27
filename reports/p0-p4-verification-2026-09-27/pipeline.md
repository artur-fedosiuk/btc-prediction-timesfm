# Pipeline contract

1. Probe Coinbase BTC-USD and Kraken XBTUSD on the actual GitHub runner. Live generation refuses to start without recorded CI evidence for both. Their selection remains provisional until that probe succeeds.
2. Fetch 512 closed primary hourly OHLCV bars (Coinbase paginated below its 300-bar limit), and an independent Kraken origin cross-check. Normalize each provider's transport ordering. Never fill missing candles or substitute a different pair/provider. Preserve decimal strings; model input explicitly converts close prices to float32.
3. Validate identity, finite positive OHLC, nonnegative volume, hourly grid, completeness, closure, freshness and availability. Availability is the first retrieval time observed by this collector, a conservative bound, not a fabricated exchange publication timestamp.
4. Origin is the exclusive closing boundary of the last input bucket. Cutoff is taken after retrieval. Generation must complete after cutoff and before origin + 1h. A bucket starting at 15:00 ends at 16:00 and cannot be input at 15:43.
5. Pin TimesFM distribution 3.0.1 and `google/timesfm-3.0-pytorch` revision `43046b85ec22d584a13f8098c2ed39c889e129c2`. Use CPU, one series, 512 float32 hourly closes, 24 steps, symmetric averaging, seed 0, two CPU threads. Validate all nine quantiles and the saved q10/median/q90 paths. q10–q90 is a nominal 80% pointwise interval, uncalibrated on BTC. There is no automatic model fallback.
6. Atomically save experiment, snapshot and immutable prediction. Record Python, package versions, Git SHA, dirty-tree flag, device and configuration. Exact bitwise replay across hardware/library versions is not promised.
7. Independent verification fetches matured hourly buckets and appends a separate record for each horizon 1–24. The primary exact close remains the evaluation target. Raw cross-source values, mean, median and relative spread are separate. Spread >1% is explicitly flagged and excluded; two-source median is not robust outlier protection.
8. Complete windows additionally expose the equal-weight mean of 24 hourly closes, actual OHLC high and low. This mean is a discrete TWAP proxy, never substituted for the endpoint.
9. Python creates `reports/data.json`; HTML only renders it. The legacy archive is displayed separately and never enters new metrics. No live trade is manufactured from a pre-generation origin price. The mocked E2E supplies a separate post-generation entry quote and reconciles all costs.

## Operations

- Probe: `python -m evaluation.forward.provider_probe --output reports/provider-checks-ci.json` (the CLI records LOCAL outside Actions; local output cannot satisfy the CI adoption gate).
- Generate: `python -m evaluation.run_predict --use-timesfm`.
- Verify independently: `python -m evaluation.run_verify`.
- Export/report: `python -m evaluation.forward.cli report`.
- Serve locally: `python -m http.server 8000`, then open `/reports/index.html`.

The prepared workflow runs the BTC test gate before mutation, probes both providers, verifies even when generation fails, and commits data plus failure evidence. Manual modes are probe/verify/generate. Scheduled verification is hourly; scheduled generation is at 12 UTC within 2026-09-27 through 2026-10-26 inclusive. Verification continues after the generation window. GitHub schedules can be delayed or skipped; exact hourly execution is not promised.

Kraken's rolling history is limited to 720 entries. Long outages can leave targets unavailable. This is a visible verification failure, with no provider substitution. Existing deprecated CSV utilities only support isolated compatibility tests; the production legacy path rejects writes. The chat application and archived v1/v2 components are outside this forward experiment.

Provider contracts: [Coinbase candles](https://docs.cdp.coinbase.com/api-reference/exchange-api/rest-api/products/get-product-candles), [Kraken OHLC](https://docs.kraken.com/api-reference/market-data/get-ohlc-data).
