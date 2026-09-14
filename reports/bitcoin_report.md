# Bitcoin (BTC/USD) Prediction Evaluation Report

> ⚠️ **Disclaimer**: This is a technical experiment, not financial advice. Past model performance does not predict future results.

## Experiment Configuration

- **Start date**: 2026-09-03
- **Duration**: 14 days
- **Prediction method(s)**: arima_fallback, timesfm
- **Report generated**: 2026-09-14 17:58 UTC
- **Timezone**: UTC

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total predictions | 13 |
| Verified predictions | 12 |
| MAE (Mean Absolute Error) | $604.93 |
| RMSE (Root Mean Square Error) | $797.17 |
| Mean Percentage Error | 0.77% |
| Direction Accuracy | 58.3% |
| Within Range Accuracy | 41.7% |

## Daily Results

| Date (UTC) | Initial ($) | Predicted ($) | Min ($) | Max ($) | Actual ($) | Error ($) | Error (%) | Dir. Pred. | Dir. Real | Dir. ✓ | In Range |
|------------|-------------|---------------|---------|---------|------------|-----------|-----------|------------|-----------|--------|----------|
| 2026-09-03 | 80,470.00 | 80,521.68 | 80,162.85 | 80,880.51 | 78,918.77 | 1,602.91 | 2.03% | 📈 up | 📉 down | ❌ | ❌ |
| 2026-09-04 | 79,390.00 | 79,430.98 | 79,032.35 | 79,829.60 | 79,803.39 | 372.41 | 0.47% | 📈 up | 📈 up | ✅ | ✅ |
| 2026-09-05 | 79,735.00 | 79,735.25 | 79,337.36 | 80,133.14 | 79,526.80 | 208.45 | 0.26% | 📈 up | 📉 down | ❌ | ✅ |
| 2026-09-05 | 79,783.00 | 79,789.30 | 79,691.75 | 79,911.03 | 79,707.93 | 81.37 | 0.10% | 📈 up | 📉 down | ❌ | ✅ |
| 2026-09-06 | 79,527.00 | 79,531.41 | 79,302.51 | 79,802.52 | 79,180.44 | 350.97 | 0.44% | 📈 up | 📉 down | ❌ | ❌ |
| 2026-09-07 | 79,076.00 | 79,117.61 | 78,902.27 | 79,353.39 | 78,511.13 | 606.48 | 0.77% | 📈 up | 📉 down | ❌ | ❌ |
| 2026-09-08 | 78,643.00 | 78,637.05 | 78,360.05 | 78,901.78 | 78,586.85 | 50.20 | 0.06% | 📉 down | 📉 down | ✅ | ✅ |
| 2026-09-09 | 78,524.00 | 78,516.34 | 78,193.12 | 78,861.67 | 77,209.59 | 1,306.75 | 1.69% | 📉 down | 📉 down | ✅ | ❌ |
| 2026-09-10 | 77,022.00 | 77,033.53 | 76,706.77 | 77,373.84 | 77,694.27 | 660.74 | 0.85% | 📈 up | 📈 up | ✅ | ❌ |
| 2026-09-11 | 77,876.00 | 77,705.84 | 77,275.70 | 78,236.39 | 77,366.92 | 338.92 | 0.44% | 📉 down | 📉 down | ✅ | ✅ |
| 2026-09-12 | 77,436.00 | 77,433.75 | 77,285.97 | 77,574.56 | 77,187.00 | 246.75 | 0.32% | 📉 down | 📉 down | ✅ | ❌ |
| 2026-09-13 | 77,122.00 | 77,128.54 | 76,923.19 | 77,337.22 | 78,561.79 | 1,433.25 | 1.82% | 📈 up | 📈 up | ✅ | ❌ |
| 2026-09-14 | 78,964.00 | 78,942.93 | 78,649.31 | 79,185.94 | — | — | — | 📉 down | — | — | — |

## Conclusion

After 12 verified predictions, the model achieved a Mean Absolute Error of $604.93 (0.77% average percentage error). Direction was predicted correctly 58.3% of the time, and 41.7% of actual prices fell within the predicted range.

**Note**: These results are from a limited 14-day experiment. Cryptocurrency markets are highly volatile and unpredictable. This evaluation is for research purposes only and should not be used for trading decisions.
