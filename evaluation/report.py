"""
Markdown report generator for the Bitcoin prediction experiment.

Generates `reports/bitcoin_report.md` with summary statistics,
a daily results table, per-method breakdowns, price verification
details, and a cautious conclusion.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import (
    EXPERIMENT_DURATION_DAYS,
    EXPERIMENT_START_DATE,
    REPORT_PATH,
)
from .metrics import compute_mae, compute_mean_percentage_error, compute_rmse

logger = logging.getLogger(__name__)


def generate_report(
    df: pd.DataFrame, output_path: Path | None = None
) -> None:
    """Generate the Markdown evaluation report.

    Args:
        df: DataFrame loaded from the predictions CSV.
        output_path: Override for the report file path.
    """
    output_path = output_path or REPORT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    total_predictions = len(df)

    # Verified rows have a non-empty actual_price_24h
    verified = df[
        df["actual_price_24h"].notna() & (df["actual_price_24h"] != "")
    ].copy()
    num_verified = len(verified)

    # Parse numeric columns for verified rows
    if num_verified > 0:
        abs_errors = verified["absolute_error"].astype(float).tolist()
        pct_errors = verified["percentage_error"].astype(float).tolist()
        direction_correct_count = (
            verified["direction_correct"] == "true"
        ).sum()
        within_range_count = (verified["within_range"] == "true").sum()

        mae = compute_mae(abs_errors)
        rmse = compute_rmse(abs_errors)
        mean_pct = compute_mean_percentage_error(pct_errors)
        direction_pct = (
            direction_correct_count / num_verified * 100
            if num_verified > 0
            else 0.0
        )
        range_pct = (
            within_range_count / num_verified * 100
            if num_verified > 0
            else 0.0
        )
    else:
        mae = rmse = mean_pct = direction_pct = range_pct = 0.0

    # Determine methods used
    methods_used = df["prediction_method"].dropna().unique().tolist()
    methods_str = ", ".join(methods_used) if methods_used else "N/A"

    # Build the report
    lines: list[str] = []
    lines.append("# Bitcoin (BTC/USD) Prediction Evaluation Report")
    lines.append("")
    lines.append(
        "> ⚠️ **Disclaimer**: This is a technical experiment, not financial "
        "advice. Past model performance does not predict future results."
    )
    lines.append("")
    lines.append("## Experiment Configuration")
    lines.append("")
    lines.append(f"- **Start date**: {EXPERIMENT_START_DATE}")
    lines.append(f"- **Duration**: {EXPERIMENT_DURATION_DAYS} days")
    lines.append(f"- **Prediction method(s)**: {methods_str}")
    lines.append(f"- **Report generated**: {now}")
    lines.append(f"- **Timezone**: UTC")
    lines.append(
        "- **Price verification**: TWAP 24h "
        "(CoinGecko + Binance cross-verified)"
    )
    lines.append(
        "- **Actual price method**: TWAP — simple average of all hourly "
        "prices in the 24h window following each prediction"
    )
    lines.append("")
    lines.append("## Summary Statistics")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total predictions | {total_predictions} |")
    lines.append(f"| Verified predictions | {num_verified} |")
    lines.append(f"| MAE (Mean Absolute Error) | ${mae:,.2f} |")
    lines.append(f"| RMSE (Root Mean Square Error) | ${rmse:,.2f} |")
    lines.append(f"| Mean Percentage Error | {mean_pct:.2f}% |")
    lines.append(f"| Direction Accuracy | {direction_pct:.1f}% |")
    lines.append(f"| Within Range Accuracy | {range_pct:.1f}% |")
    lines.append("")

    # --- Per-method breakdown ---
    if num_verified > 0 and len(methods_used) > 1:
        lines.append("## Performance by Method")
        lines.append("")
        lines.append(
            "| Method | Predictions | MAE ($) | RMSE ($) | "
            "Mean Error (%) | Direction Acc. | Range Acc. |"
        )
        lines.append(
            "|--------|-------------|---------|----------|"
            "----------------|---------------|------------|"
        )
        for method in sorted(methods_used):
            method_df = verified[
                verified["prediction_method"] == method
            ]
            n = len(method_df)
            if n == 0:
                continue
            m_abs = method_df["absolute_error"].astype(float).tolist()
            m_pct = method_df["percentage_error"].astype(float).tolist()
            m_mae = compute_mae(m_abs)
            m_rmse = compute_rmse(m_abs)
            m_mean_pct = compute_mean_percentage_error(m_pct)
            m_dir = (
                (method_df["direction_correct"] == "true").sum() / n * 100
            )
            m_range = (
                (method_df["within_range"] == "true").sum() / n * 100
            )
            lines.append(
                f"| {method} | {n} | ${m_mae:,.2f} | ${m_rmse:,.2f} | "
                f"{m_mean_pct:.2f}% | {m_dir:.1f}% | {m_range:.1f}% |"
            )
        lines.append("")

    # --- Daily table ---
    lines.append("## Daily Results")
    lines.append("")
    if total_predictions == 0:
        lines.append("*No predictions recorded yet.*")
    else:
        lines.append(
            "| Date (UTC) | Initial ($) | Predicted ($) | "
            "Min ($) | Max ($) | Actual ($) | Error ($) | Error (%) | "
            "Dir. Pred. | Dir. Real | Dir. ✓ | In Range |"
        )
        lines.append(
            "|------------|-------------|---------------|"
            "---------|---------|------------|-----------|-----------"
            "|------------|-----------|--------|----------|"
        )
        for _, row in df.iterrows():
            ts = str(row.get("timestamp_utc", ""))[:10]
            initial = _fmt_price(row.get("initial_price", ""))
            predicted = _fmt_price(row.get("predicted_price_24h", ""))
            pred_min = _fmt_price(row.get("predicted_min", ""))
            pred_max = _fmt_price(row.get("predicted_max", ""))
            actual = _fmt_price(row.get("actual_price_24h", ""))
            abs_err = _fmt_num(row.get("absolute_error", ""))
            pct_err = _fmt_pct(row.get("percentage_error", ""))
            pred_dir = _fmt_dir(row.get("predicted_direction", ""))
            act_dir = _fmt_dir(row.get("actual_direction", ""))
            dir_ok = _fmt_bool(row.get("direction_correct", ""))
            in_range = _fmt_bool(row.get("within_range", ""))

            lines.append(
                f"| {ts} | {initial} | {predicted} | "
                f"{pred_min} | {pred_max} | {actual} | {abs_err} | "
                f"{pct_err} | {pred_dir} | {act_dir} | {dir_ok} | "
                f"{in_range} |"
            )
    lines.append("")

    # --- Price Verification Details ---
    has_sources = (
        "source_coingecko" in df.columns
        and "source_binance" in df.columns
    )
    if has_sources and num_verified > 0:
        source_rows = verified[
            verified.get("source_coingecko", pd.Series(dtype=str)).notna()
            & (verified.get("source_coingecko", pd.Series(dtype=str)) != "")
        ]
        if len(source_rows) > 0:
            lines.append("## Price Source Verification (TWAP)")
            lines.append("")
            lines.append(
                "> Each \"Actual\" price is a **TWAP** (Time-Weighted Average "
                "Price): the simple average of all hourly BTC/USD prices "
                "in the 24h window after each prediction. Both CoinGecko "
                "and Binance TWAPs are computed independently; when they "
                "agree (≤1% discrepancy), the average of the two TWAPs "
                "is used."
            )
            lines.append("")
            lines.append(
                "| Date (UTC) | CoinGecko ($) | Binance ($) | "
                "Used ($) | Confidence |"
            )
            lines.append(
                "|------------|---------------|-------------|"
                "---------|------------|"
            )
            for _, row in source_rows.iterrows():
                ts = str(row.get("timestamp_utc", ""))[:10]
                cg = _fmt_price(row.get("source_coingecko", ""))
                bn = _fmt_price(row.get("source_binance", ""))
                used = _fmt_price(row.get("actual_price_24h", ""))
                conf = _fmt_confidence(
                    row.get("price_confidence", "")
                )
                lines.append(
                    f"| {ts} | {cg} | {bn} | {used} | {conf} |"
                )
            lines.append("")

    # --- Conclusion ---
    lines.append("## Conclusion")
    lines.append("")
    if num_verified == 0:
        lines.append(
            "The experiment has started but no predictions have been "
            "verified yet. Results will appear after the first 24-hour "
            "verification cycle completes."
        )
    elif num_verified < 5:
        lines.append(
            f"With only {num_verified} verified prediction(s), it is too "
            f"early to draw meaningful conclusions about the model's "
            f"forecasting ability. The experiment will continue to "
            f"collect data over the full {EXPERIMENT_DURATION_DAYS}-day "
            f"window."
        )
    else:
        lines.append(
            f"After {num_verified} verified predictions, the model "
            f"achieved a Mean Absolute Error of ${mae:,.2f} "
            f"({mean_pct:.2f}% average percentage error). "
            f"Direction was predicted correctly {direction_pct:.1f}% of "
            f"the time, and {range_pct:.1f}% of actual prices fell "
            f"within the predicted range."
        )
        lines.append("")
        lines.append(
            "**Note**: These results are from a limited {}-day experiment. "
            "Cryptocurrency markets are highly volatile and unpredictable. "
            "This evaluation is for research purposes only and should not "
            "be used for trading decisions.".format(
                EXPERIMENT_DURATION_DAYS
            )
        )
    lines.append("")

    report_text = "\n".join(lines)
    output_path.write_text(report_text, encoding="utf-8")
    logger.info("Report written to %s", output_path)


def _fmt_price(val: str) -> str:
    """Format a price string for the table."""
    if not val or val == "nan":
        return "—"
    try:
        return f"{float(val):,.2f}"
    except ValueError:
        return val


def _fmt_num(val: str) -> str:
    """Format a numeric string."""
    if not val or val == "nan":
        return "—"
    try:
        return f"{float(val):,.2f}"
    except ValueError:
        return val


def _fmt_pct(val: str) -> str:
    """Format a percentage string."""
    if not val or val == "nan":
        return "—"
    try:
        return f"{float(val):.2f}%"
    except ValueError:
        return val


def _fmt_dir(val: str) -> str:
    """Format a direction string with an emoji."""
    if not val or val == "nan":
        return "—"
    if val == "up":
        return "📈 up"
    elif val == "down":
        return "📉 down"
    return val


def _fmt_bool(val: str) -> str:
    """Format a boolean string."""
    if not val or val == "nan":
        return "—"
    return "✅" if val == "true" else "❌"


def _fmt_confidence(val: str) -> str:
    """Format a confidence level with emoji."""
    if not val or val == "nan":
        return "—"
    mapping = {
        "high": "🟢 high",
        "medium": "🟡 medium",
        "low": "🔴 low",
    }
    return mapping.get(val, val)
