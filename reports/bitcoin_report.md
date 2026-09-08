# Bitcoin (BTC/USD) Prediction Evaluation Report

> ⚠️ **Disclaimer**: This is a technical experiment, not financial advice. Past model performance does not predict future results.

## Experiment Configuration

- **Start date**: 2026-09-03
- **Duration**: 14 days
- **Prediction method(s)**: arima_fallback, timesfm
- **Report generated**: 2026-09-08 16:24 UTC
- **Timezone**: UTC

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total predictions | 7 |
| Verified predictions | 5 |
| MAE (Mean Absolute Error) | $523.22 |
| RMSE (Root Mean Square Error) | $759.11 |
| Mean Percentage Error | 0.66% |
| Direction Accuracy | 20.0% |
| Within Range Accuracy | 60.0% |

## Daily Results

| Date (UTC) | Initial ($) | Predicted ($) | Min ($) | Max ($) | Actual ($) | Error ($) | Error (%) | Dir. Pred. | Dir. Real | Dir. ✓ | In Range |
|------------|-------------|---------------|---------|---------|------------|-----------|-----------|------------|-----------|--------|----------|
| 2026-09-03 | 80,470.00 | 80,521.68 | 80,162.85 | 80,880.51 | 78,918.77 | 1,602.91 | 2.03% | 📈 up | 📉 down | ❌ | ❌ |
| 2026-09-04 | 79,390.00 | 79,430.98 | 79,032.35 | 79,829.60 | 79,803.39 | 372.41 | 0.47% | 📈 up | 📈 up | ✅ | ✅ |
| 2026-09-05 | 79,735.00 | 79,735.25 | 79,337.36 | 80,133.14 | 79,526.80 | 208.45 | 0.26% | 📈 up | 📉 down | ❌ | ✅ |
| 2026-09-05 | 79,783.00 | 79,789.30 | 79,691.75 | 79,911.03 | 79,707.93 | 81.37 | 0.10% | 📈 up | 📉 down | ❌ | ✅ |
| 2026-09-06 | 79,527.00 | 79,531.41 | 79,302.51 | 79,802.52 | 79,180.44 | 350.97 | 0.44% | 📈 up | 📉 down | ❌ | ❌ |
| 2026-09-07 | 79,076.00 | 79,117.61 | 78,902.27 | 79,353.39 | — | — | — | 📈 up | — | — | — |
| 2026-09-08 | 78,643.00 | 78,637.05 | 78,360.05 | 78,901.78 | — | — | — | 📉 down | — | — | — |

## Conclusion

After 5 verified predictions, the model achieved a Mean Absolute Error of $523.22 (0.66% average percentage error). Direction was predicted correctly 20.0% of the time, and 60.0% of actual prices fell within the predicted range.

**Note**: These results are from a limited 14-day experiment. Cryptocurrency markets are highly volatile and unpredictable. This evaluation is for research purposes only and should not be used for trading decisions.
