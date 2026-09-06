# Bitcoin (BTC/USD) Prediction Evaluation Report

> ⚠️ **Disclaimer**: This is a technical experiment, not financial advice. Past model performance does not predict future results.

## Experiment Configuration

- **Start date**: 2026-09-03
- **Duration**: 14 days
- **Prediction method(s)**: arima_fallback, timesfm
- **Report generated**: 2026-09-06 15:15 UTC
- **Timezone**: UTC

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total predictions | 5 |
| Verified predictions | 3 |
| MAE (Mean Absolute Error) | $727.92 |
| RMSE (Root Mean Square Error) | $957.68 |
| Mean Percentage Error | 0.92% |
| Direction Accuracy | 33.3% |
| Within Range Accuracy | 66.7% |

## Daily Results

| Date (UTC) | Initial ($) | Predicted ($) | Min ($) | Max ($) | Actual ($) | Error ($) | Error (%) | Dir. Pred. | Dir. Real | Dir. ✓ | In Range |
|------------|-------------|---------------|---------|---------|------------|-----------|-----------|------------|-----------|--------|----------|
| 2026-09-03 | 80,470.00 | 80,521.68 | 80,162.85 | 80,880.51 | 78,918.77 | 1,602.91 | 2.03% | 📈 up | 📉 down | ❌ | ❌ |
| 2026-09-04 | 79,390.00 | 79,430.98 | 79,032.35 | 79,829.60 | 79,803.39 | 372.41 | 0.47% | 📈 up | 📈 up | ✅ | ✅ |
| 2026-09-05 | 79,735.00 | 79,735.25 | 79,337.36 | 80,133.14 | 79,526.80 | 208.45 | 0.26% | 📈 up | 📉 down | ❌ | ✅ |
| 2026-09-05 | 79,783.00 | 79,789.30 | 79,691.75 | 79,911.03 | — | — | — | 📈 up | — | — | — |
| 2026-09-06 | 79,527.00 | 79,531.41 | 79,302.51 | 79,802.52 | — | — | — | 📈 up | — | — | — |

## Conclusion

With only 3 verified prediction(s), it is too early to draw meaningful conclusions about the model's forecasting ability. The experiment will continue to collect data over the full 14-day window.
