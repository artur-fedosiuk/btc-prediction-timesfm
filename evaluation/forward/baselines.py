"""Same-origin baselines for fair comparison against TimesFM H24 forecasts.

All three baselines receive the IDENTICAL input context that TimesFM sees
(the float32 close-price series from the snapshot), so the comparison is
apples-to-apples on both information set and target horizon.

Metrics are computed on LOG RETURNS (not price levels) to avoid
scale contamination and to test whether the model beats a no-information
null hypothesis on relative moves.

Baselines
---------
persistence   Last observed value is repeated for all horizons (random-walk
              under IID log-returns).
drift         Linear trend estimated by OLS on the context window; extrapolated
              H steps forward.
arima         ARIMA(5,1,0) on the last 168 points (7 days), matching the
              legacy evaluation/predictor.py implementation so any future
              live backtest uses the same hyper-parameters.

Minimum-sample policy
---------------------
Statistical conclusions require >= 60 DAILY origins (where one origin = one
TimesFM prediction, i.e. one row in the predictions table, not one hourly
candle). Hourly verification targets within a single prediction are NOT
independent samples. report_baselines() enforces this gate and marks results
STATISTICALLY_INCONCLUSIVE until the threshold is reached.
"""

from __future__ import annotations

import math
from decimal import Decimal

# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

MIN_DAILY_ORIGINS = 60  # One per TimesFM prediction, NOT per hourly target.
ARIMA_CONTEXT = 168     # 7 days of hourly closes, matching predictor.py.
ARIMA_ORDER   = (5, 1, 0)


def _log_return(a: float, b: float) -> float:
    """log(b/a) — safe for positive prices."""
    if a <= 0 or b <= 0:
        return float("nan")
    return math.log(b / a)


def _persistence_path(values: list[float], horizon: int = 24) -> list[float]:
    """Repeat the last observed value for every horizon step."""
    last = values[-1]
    return [last] * horizon


def _drift_path(values: list[float], horizon: int = 24) -> list[float]:
    """Random walk with drift: last + h × mean(consecutive differences).

    Anchored to the last observed price so the forecast starts from the
    most recent value, not from a regression intercept.  With fewer than
    2 data points, falls back to persistence.
    """
    n = len(values)
    if n < 2:
        return _persistence_path(values, horizon)
    last = values[-1]
    mean_diff = sum(values[i] - values[i - 1] for i in range(1, n)) / (n - 1)
    return [max(last + h * mean_diff, 1e-6) for h in range(1, horizon + 1)]


def _arima_path(
    values: list[float], horizon: int = 24,
) -> tuple[list[float], bool]:
    """ARIMA(5,1,0) on the last ARIMA_CONTEXT points.

    Returns
    -------
    (path, fallback)
        *path* has length *horizon*.  *fallback* is True when ARIMA could
        not be fitted and the drift baseline was used instead.
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA  # type: ignore
    except ImportError:
        return _drift_path(values, horizon), True

    data = values[-ARIMA_CONTEXT:] if len(values) >= ARIMA_CONTEXT else values[:]
    try:
        fitted = ARIMA(data, order=ARIMA_ORDER).fit()
        return (
            [max(v, 1e-6) for v in fitted.get_forecast(steps=horizon).predicted_mean.tolist()],
            False,
        )
    except Exception:  # noqa: BLE001
        return _drift_path(values, horizon), True


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def baselines_for_prediction(
    input_values: list[float],
    forecast_path: list[float],
    verifications: list[dict],
    *,
    horizon: int = 24,
) -> dict:
    """Compute persistence, drift and ARIMA paths on the same origin as TimesFM.

    Parameters
    ----------
    input_values:
        float32 close-price series from the snapshot (already quantized to f32).
    forecast_path:
        TimesFM point-forecast path of length *horizon* (for direction comparison).
    verifications:
        List of verification dicts from the ledger.  May be a subset (some
        horizons are still pending).
    horizon:
        Forecast horizon length (always 24 for the current experiment).

    Returns
    -------
    dict with keys "persistence", "drift", "arima", each containing:
        "path":            list[float]  — forecast for H1…H24
        "origin_price":   float
        "metrics":        dict (only over verified horizons)
    """
    if not input_values:
        return {}

    origin_price = input_values[-1]

    arima_path, arima_fallback = _arima_path(input_values, horizon)
    paths = {
        "persistence": _persistence_path(input_values, horizon),
        "drift":       _drift_path(input_values, horizon),
        "arima":       arima_path,
        "timesfm":     forecast_path[:horizon],
    }

    # Build actuals map from verified horizons only.
    actuals: dict[int, float] = {}
    for v in verifications:
        if v.get("status") == "VERIFIED":
            try:
                actuals[int(v["horizon"])] = float(v["exact_price"])
            except (KeyError, ValueError):
                pass

    results: dict[str, dict] = {}
    for name, path in paths.items():
        metrics = _compute_metrics(origin_price, path, actuals)
        results[name] = {
            "path":         path,
            "origin_price": origin_price,
            "metrics":      metrics,
        }
    results["arima"]["fallback"] = arima_fallback

    return results


def _compute_metrics(
    origin: float,
    path: list[float],
    actuals: dict[int, float],
) -> dict:
    """Per-horizon and aggregate metrics over the verified subset.

    All errors are computed on LOG RETURNS so scale-independent comparison
    with any future multi-origin aggregation is correct.
    """
    if not actuals or origin <= 0:
        return {"note": "no verified horizons yet"}

    maes, dir_hits = [], []
    per_horizon = []
    for h, forecast in enumerate(path, 1):
        actual = actuals.get(h)
        if actual is None or forecast is None or forecast <= 0 or actual <= 0:
            per_horizon.append({"horizon": h, "verified": False})
            continue
        lr_forecast = _log_return(origin, forecast)
        lr_actual   = _log_return(origin, actual)
        ae = abs(lr_forecast - lr_actual)
        maes.append(ae)
        # Direction: sign of log-return vs origin (not vs prev hour).
        # When lr_forecast == 0 the model has no directional opinion
        # (persistence), so that horizon is excluded from direction_accuracy.
        if lr_forecast != 0:
            dir_hits.append((lr_forecast > 0) == (lr_actual > 0))
        per_horizon.append({
            "horizon":    h,
            "verified":   True,
            "lr_forecast": round(lr_forecast * 100, 6),  # in %
            "lr_actual":   round(lr_actual   * 100, 6),
            "log_ae":      round(ae * 100, 6),
        })

    n = len(maes)
    if n == 0:
        return {"note": "no verified horizons yet"}

    mae_lr  = sum(maes) / n
    rmse_lr = math.sqrt(sum(m ** 2 for m in maes) / n)
    dir_acc = sum(dir_hits) / len(dir_hits) if dir_hits else None

    return {
        "n_verified":      n,
        "mae_log_return":  round(mae_lr  * 100, 6),   # % units
        "rmse_log_return": round(rmse_lr * 100, 6),
        "direction_accuracy": round(dir_acc, 4) if dir_acc is not None else None,
        "per_horizon":     per_horizon,
        "note": (
            "Metrics on log returns relative to origin. "
            "Horizons are NOT independent — aggregate only across distinct daily origins."
        ),
    }


def report_baselines(predictions: list[dict], verifications: list[dict]) -> dict:
    """Aggregate baseline metrics across all predictions in the ledger.

    Enforces the minimum-sample policy: each row in *predictions* counts as
    ONE origin (one daily TimesFM run), regardless of how many hourly targets
    have been verified.  The 60-origin threshold must be met before any
    statistical conclusion is drawn.

    Parameters
    ----------
    predictions:
        All prediction records from the ledger (list of dicts).
    verifications:
        All verification records from the ledger (list of dicts).

    Returns
    -------
    dict suitable for embedding in reports/data.json under "baselines".
    """
    # Count distinct UTC dates, not raw rows.
    _dates: set[str] = set()
    for p in predictions:
        fo = p.get("forecast_origin", "")
        if fo:
            _dates.add(fo[:10])          # ISO-8601 date prefix
    n_daily_origins = len(_dates) if _dates else len(predictions)
    status = (
        "STATISTICALLY_INCONCLUSIVE"
        if n_daily_origins < MIN_DAILY_ORIGINS
        else "EVALUABLE"
    )
    note = (
        f"Need >= {MIN_DAILY_ORIGINS} daily origins (distinct UTC dates) for "
        f"statistical conclusions; current: {n_daily_origins}. Hourly verification "
        "targets within a single prediction are NOT independent samples."
    )

    # Map verifications by prediction_id.
    verif_map: dict[str, list[dict]] = {}
    for v in verifications:
        pid = v.get("prediction_id", "")
        verif_map.setdefault(pid, []).append(v)

    per_model: dict[str, dict] = {
        "persistence": {"mae_lr_sum": 0.0, "rmse_sq_sum": 0.0, "dir_sum": 0.0, "n": 0, "dir_n": 0},
        "drift":       {"mae_lr_sum": 0.0, "rmse_sq_sum": 0.0, "dir_sum": 0.0, "n": 0, "dir_n": 0},
        "arima":       {"mae_lr_sum": 0.0, "rmse_sq_sum": 0.0, "dir_sum": 0.0, "n": 0, "dir_n": 0},
        "timesfm":     {"mae_lr_sum": 0.0, "rmse_sq_sum": 0.0, "dir_sum": 0.0, "n": 0, "dir_n": 0},
    }

    arima_fallback_count = 0
    per_prediction = []
    for p in predictions:
        pid   = p.get("prediction_id", "")
        # input_values are stored in the snapshot, not the prediction itself.
        # The calling code in analytics.build_report has access to snapshots.
        # We accept them here pre-joined via a "_input_values" key if present.
        input_vals = p.get("_input_values", [])
        fp = p.get("forecast_path", [])
        vfs = verif_map.get(pid, [])

        if not input_vals or not fp:
            per_prediction.append({"prediction_id": pid, "note": "input_values not joined"})
            continue

        row = baselines_for_prediction(input_vals, fp, vfs)
        per_prediction.append({"prediction_id": pid, "baselines": row})
        if row.get("arima", {}).get("fallback"):
            arima_fallback_count += 1

        for name in per_model:
            m = row.get(name, {}).get("metrics", {})
            mae_lr = m.get("mae_log_return")
            rmse_lr = m.get("rmse_log_return")
            dir_acc = m.get("direction_accuracy")
            n = m.get("n_verified", 0)
            if n and mae_lr is not None and rmse_lr is not None:
                per_model[name]["mae_lr_sum"]   += mae_lr * n
                per_model[name]["rmse_sq_sum"]  += (rmse_lr ** 2) * n
                per_model[name]["n"]            += n
                if dir_acc is not None:
                    per_model[name]["dir_sum"] += dir_acc * n
                    per_model[name]["dir_n"]   += n

    summary = {}
    for name, acc in per_model.items():
        n = acc["n"]
        if n == 0:
            summary[name] = {"note": "no verified data yet"}
            continue
        summary[name] = {
            "mae_log_return_pct":  round(acc["mae_lr_sum"] / n, 6),
            "rmse_log_return_pct": round(math.sqrt(acc["rmse_sq_sum"] / n), 6),
            "direction_accuracy":  round(acc["dir_sum"] / acc["dir_n"], 4)
            if acc["dir_n"] else None,
            "n_verified_targets":  n,
        }

    return {
        "schema_version":    1,
        "status":            status,
        "n_daily_origins":   n_daily_origins,
        "min_required":      MIN_DAILY_ORIGINS,
        "arima_fallback_count": arima_fallback_count,
        "note":              note,
        "metric_definition": (
            "mae_log_return_pct and rmse_log_return_pct are 100 * |log(forecast/origin) "
            "- log(actual/origin)| averaged over verified horizons. Scale-free. "
            "direction_accuracy = fraction of horizons where sign(log-return forecast) "
            "== sign(log-return actual), relative to forecast origin."
        ),
        "summary":           summary,
        "per_prediction":    per_prediction,
    }
