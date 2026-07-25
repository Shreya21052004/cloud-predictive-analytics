"""prediction.py — Forecasting and prediction assembly.

Key changes from v2:
1. STL decomposition (statsmodels) replaces raw linear regression for series with
   >= LONG_SERIES_THRESHOLD points. Trend component is extracted cleanly without
   diurnal noise contaminating the slope.
2. Bootstrap confidence intervals on the trend slope. Every forecast now includes
   forecast_*_lower and forecast_*_upper bounds, not just a point estimate.
3. Trend detection uses the STL trend component slope, not raw signal range.
4. Diurnal override only fires when confidence < 0.5 (was unconditional).
5. forecast_reliability thresholds recalibrated for STL R² (tighter baseline).
6. Profiles built at prediction time and passed to score_resource.
7. cpu_idle → cpu_pct derivation in consolidated_row.
8. v7: profiles and the anomaly IsolationForest are pooled by
   canonical_resource_type (component), never fit per resource_id. A
   resource is only ever scored against its type's shared baseline/model —
   see profiler.build_all_profiles and anomaly.train_type_isolation_forests.
"""
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import pandas as pd

from .forecast_math import (
    HAS_STL, MIN_POINTS_FOR_FORECAST, LONG_SERIES_THRESHOLD, N_BOOTSTRAP,
    exp_smooth_forecast as _exp_smooth_forecast,
    bootstrap_slope_ci as _bootstrap_slope_ci,
    stl_decompose as _stl_decompose,
    seasonal_phase_forecast,
    long_run_drift_direction,
    safe_polyfit_slope,
)
from .backtest import run_holdout_backtest, backtest_quality_score, select_forecast_method as _select_forecast_method, choose_holdout_size as _choose_holdout_size

from .models import (
    MODEL_VERSION, risk_from_model, derive_missing_signals,
    predict_slope_prior, build_type_specific_slope_priors, compute_type_specific_slope_prior, 
    SLOPE_PRIOR_MIN_POINTS, SLOPE_PRIOR_SHRINK_K,
)
from .anomaly import score_resource, train_type_isolation_forests
from .metric_registry import (
    PRIMARY_FORECAST_FEATURE, canonical_features, metric_name_for_feature,
    forecast_feature_fallback_order,
)
from .profiler import build_all_profiles
from .resource_types import is_well_populated
from .config import TARGET_CATEGORIES


def now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def clean(value):
    if isinstance(value, (dict, list)):
        return value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return value
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def is_real(value):
    return value is not None and not pd.isna(value)


def is_true_flag(value):
    return value is True or value == 1


def positive(value):
    return is_real(value) and value > 0


def severity(score):
    if score >= 90: return "CRITICAL"
    if score >= 75: return "HIGH"
    if score >= 50: return "WARNING"
    return "INFO"


def badge(score):
    if score >= 75: return "red"
    if score >= 50: return "amber"
    if score >= 25: return "blue"
    return "green"


def channel_for(sev):
    return {"CRITICAL": "pagerduty", "HIGH": "pagerduty", "WARNING": "slack", "INFO": "suppressed"}[sev]


BURSTY_METRICS = {
    "canonical_storage_read_bytes", "canonical_storage_write_bytes",
    "canonical_storage_read_iops", "canonical_storage_write_iops",
    "canonical_storage_transactions",
    "canonical_net_in_bytes", "canonical_net_out_bytes",
    "canonical_active_connections",
    "canonical_disk_read_bytes", "canonical_disk_write_bytes",
    "canonical_disk_read_iops", "canonical_disk_write_iops",
    "lambda_throttles", "lambda_invocations", "lambda_errors",
    "lambda_concurrent_executions", "status_check_failed",
}

MIN_NONZERO_FRAC      = 0.2
MIN_STD_NORM          = 0.01
# BUG FIX / methodology upgrade (v9): a bursty metric needs at least two
# full diurnal cycles of history before phase-bucketed medians mean
# anything (each of the 24 hour-of-day buckets needs >= 2 points to be
# usable at all) -- see seasonal_phase_forecast() in forecast_math.py and
# the BURSTY_METRICS branch below. Below this, the old exponential-smoothing
# + slope-extrapolation approach is kept as an explicit short-series
# fallback, not silently applied to everything.
MIN_POINTS_FOR_SEASONAL_BASELINE = 48

# _exp_smooth_forecast, _bootstrap_slope_ci, _stl_decompose, HAS_STL,
# MIN_POINTS_FOR_FORECAST, LONG_SERIES_THRESHOLD, N_BOOTSTRAP now live in
# forecast_math.py (imported above) so the live forecaster and the holdout
# backtester in backtest.py share one copy of this math instead of two.


def _clean_series(series, feature):
    y = series[feature].values.astype(float)
    zero_frac = (y == 0).sum() / len(y)
    if zero_frac > 0.8:
        return series, "idle"
    if zero_frac > MIN_NONZERO_FRAC:
        nonzero_idx = np.where(y != 0)[0]
        if len(nonzero_idx) >= MIN_POINTS_FOR_FORECAST:
            series = series.iloc[:nonzero_idx[-1] + 1]
        return series, "cleaned"
    return series, "ok"


def _detect_mixed_stats(resource_history, feature):
    stat_col = f"{feature}_stat"
    if stat_col not in resource_history.columns:
        return False
    stats_used = resource_history[stat_col].dropna().unique()
    return len(stats_used) > 1


def slope_forecast(group, feature, thresholds):
    series = group[["timestamp", feature]].dropna().sort_values("timestamp")
    if len(series) < 2:
        return {}
    x = (series["timestamp"] - series["timestamp"].min()).dt.total_seconds() / 3600.0
    y = series[feature].astype(float)
    slope, intercept = safe_polyfit_slope(x, y)
    current_y = float(y.iloc[-1])
    out = {"current": current_y, "hourly_slope": float(slope)}
    MAX_FORECAST_HOURS = 87_600
    for threshold in thresholds:
        key = f"breach_{int(threshold)}"
        if current_y >= threshold:
            out[key] = series["timestamp"].iloc[-1].isoformat()
        elif slope > 0:
            hours = (threshold - current_y) / slope
            if hours > MAX_FORECAST_HOURS:
                out[key] = None
            else:
                try:
                    out[key] = (series["timestamp"].iloc[-1] + pd.Timedelta(hours=float(hours))).isoformat()
                except (OverflowError, pd.errors.OutOfBoundsDatetime):
                    out[key] = None
        else:
            out[key] = None
    return out


def _shrink_toward_prior(local_slope, n_points, prior_slope_norm, value_range):
    """Empirical-Bayes shrinkage: blend a noisy local slope estimate toward
    the pooled XGBoost prior when a resource/metric pair has thin history.

    weight_local = n / (n + K) — more local points => trust local slope more.
    Returns (blended_slope, weight_local, applied). `applied` is False when
    there's no prior available or the series already has enough points that
    shrinkage would be a no-op — callers use this to decide whether to
    relabel forecast_method.
    """
    if prior_slope_norm is None or n_points >= SLOPE_PRIOR_MIN_POINTS:
        return local_slope, 1.0, False
    weight_local = n_points / (n_points + SLOPE_PRIOR_SHRINK_K)
    prior_slope_raw = prior_slope_norm * value_range
    blended = weight_local * local_slope + (1 - weight_local) * prior_slope_raw
    return blended, weight_local, True


UNBOUNDED_RAW_FEATURES = {
    # Features with no natural ceiling/quota to validate a trend against
    # (raw byte counters, unresolved capacity counts). A forecast built on
    # one of these can look like a clean line yet still be meaningless as
    # a "capacity exhaustion" signal, so it is never allowed to claim
    # forecast_reliability == "high" — see _finalize_forecast().
    "canonical_storage_capacity_raw",
    "canonical_net_in_bytes",
    "canonical_net_out_bytes",
    "canonical_storage_read_bytes",
    "canonical_storage_write_bytes",
    "canonical_credit_balance_raw",
    # Same reasoning: GCP's flow-log count prediction and active-probing
    # count are volume/count metrics, not bounded percentages, even though
    # they now live under Network's forecast-feature priority list.
    "canonical_flow_log_count",
    "canonical_probe_failure_count",
}


def _finalize_forecast(r, resource_history, feature):
    """Single exit point for every behavioral_forecast() return path.

    BUG FIX (critical): previously only the main STL / linear-regression /
    exponential-smoothing branch called _compute_prediction_confidence_metrics().
    Every early-return path -- feature_not_available, no_data_points,
    insufficient_points (with or without a pooled prior), all_zeros_after_cleaning,
    and the entire flat_signal branch -- returned WITHOUT prediction_confidence_score,
    accuracy_proxy_score, data_completeness_score, or forecast_uncertainty_pct ever
    being computed. That silently left roughly 60% of real-world predictions
    (idle resources + no-data resources are common) with a missing/null
    confidence score, which downstream reporting then treated as untrustworthy
    by default. Routing every return through this function guarantees the
    full metrics block is always populated exactly once.

    BUG FIX: a feature with no natural ceiling (raw bytes/counts) is capped
    out of "high" reliability here, in one place, so every branch benefits
    consistently instead of only the main path.

    METHODOLOGY UPGRADE (v9): everything above (and the in-sample
    trend_fit_r2 computed inside behavioral_forecast) is goodness-of-fit —
    how well a model explains the SAME data it was fit on. That is not a
    measure of forecast accuracy. Here, whenever there's enough history
    (see backtest.choose_holdout_size), a real train/holdout backtest is
    run: fit on all but the most recent points, forecast forward, and
    compare against what actually happened. When available, its MASE +
    interval-coverage quality score REPLACES the in-sample R² as the basis
    for both forecast_reliability and the confidence composite, because a
    real out-of-sample result is a strictly better estimate of forecast
    quality than in-sample fit. `evaluation_basis` records honestly which
    one actually happened for this prediction — "out_of_sample_backtest" or
    "in_sample_fit_only" — so downstream consumers can filter to only trust
    the former for business-critical decisions.
    """
    model_r2_for_score = r.get("trend_fit_r2", 0.0) or 0.0

    try:
        backtest = run_holdout_backtest(resource_history, feature, is_bursty=feature in BURSTY_METRICS)
    except Exception:
        backtest = None

    if backtest:
        quality = backtest_quality_score(
            backtest.get("backtest_mase"), backtest.get("backtest_interval_coverage_pct")
        )
        backtest["backtest_quality_score"] = quality
        r.update(backtest)
        r["evaluation_basis"] = "out_of_sample_backtest"
        model_r2_for_score = quality / 100.0
        r["forecast_reliability"] = "high" if quality >= 80 else ("moderate" if quality >= 50 else "low")

        # v9.1: backtest_low_variance_training means the MASE denominator
        # and/or CI band had to be floored because the training window's
        # own variance was near-zero (see backtest.py). The resulting
        # quality score is the best honest estimate available, but it's
        # computed against an artificially widened denominator/band, so
        # don't let it alone claim "high" -- same treatment as
        # UNBOUNDED_RAW_FEATURES below, and for the same reason: a score
        # that's structurally propped up shouldn't outrank a score earned
        # on real variance.
        if backtest.get("backtest_low_variance_training") and r["forecast_reliability"] == "high":
            r["forecast_reliability"] = "moderate"
            note = r.get("data_note")
            r["data_note"] = f"{note}+low_variance_training_window" if note and note != "ok" else "low_variance_training_window"
    else:
        r["backtest_available"] = False
        r["evaluation_basis"] = "in_sample_fit_only"

    if feature in UNBOUNDED_RAW_FEATURES and r.get("forecast_reliability") == "high":
        r["forecast_reliability"] = "moderate"
        note = r.get("data_note")
        r["data_note"] = f"{note}+unbounded_raw_metric" if note and note != "ok" else "unbounded_raw_metric"

    if r.get("prediction_confidence_score") is None:
        r.update(_compute_prediction_confidence_metrics(
            r,
            model_r2_for_score,
            resource_history,
            feature,
            r.get("forecast_reliability", "low"),
            r.get("forecast_method", "none"),
        ))
    return r


def _pooling_tier(shrink_applied, resource_canonical_type):
    """Classify which tier a forecast actually drew on (handoff doc §5/§6).

    "individual"      — no shrinkage happened; the resource's own history
                         was trusted as-is (either it had enough points, or
                         no prior was available to shrink toward).
    "type_pooled"      — shrunk toward the pooled prior, and this resource's
                         canonical_resource_type is one of the well-populated
                         types with real pooled volume behind it.
    "category_pooled"  — shrunk toward the pooled prior, but this resource's
                         type is a thin/unmapped other_<category> bucket, so
                         the "pooling" it borrowed is really just
                         category-wide, not type-specific.
    """
    if not shrink_applied:
        return "individual"
    if resource_canonical_type and is_well_populated(resource_canonical_type):
        return "type_pooled"
    return "category_pooled"


def behavioral_forecast(category, resource_history, row, artifact=None, type_priors_cache=None, forced_feature=None):
    feature = forced_feature if forced_feature else choose_forecast_feature(category, resource_history)
    provider = row.get("service_name")
    metric_name = metric_name_for_feature(category, provider, feature)
    resource_canonical_type = row.get("canonical_resource_type")

    null_result = {
        "metric_name":          metric_name,
        "canonical_feature":    feature,
        "current_value":        None,
        # Additive fields (see resource_types.py / handoff doc §6).
        # pooling_tier reflects the tier this forecast actually drew on:
        # "individual"      — resource's own history was trusted as-is
        # "type_pooled"     — shrunk toward a well-populated canonical-type prior
        # "category_pooled" — shrunk toward the category-wide prior (thin/unmapped type)
        # Default here is "individual" since none of the pooling logic below
        # has run yet at this point in the function.
        "canonical_resource_type": resource_canonical_type,
        "pooling_tier":         "individual",
        "forecast_1h":          None, "forecast_1h_lower":  None, "forecast_1h_upper":  None,
        "forecast_6h":          None, "forecast_6h_lower":  None, "forecast_6h_upper":  None,
        "forecast_24h":         None, "forecast_24h_lower": None, "forecast_24h_upper": None,
        "forecast_7d":          None, "forecast_7d_lower":  None, "forecast_7d_upper":  None,
        "forecast_30d":         None, "forecast_30d_lower": None, "forecast_30d_upper": None,
        "trend_direction":      "stable",
        "trend_fit_r2":         0.0,
        "forecast_reliability": "low",
        "forecast_method":      "none",
        "data_note":            None,
    }

    if not feature or feature not in resource_history.columns:
        return _finalize_forecast({**null_result, "data_note": "feature_not_available"}, resource_history, feature)

    series = resource_history[["timestamp", feature]].dropna().sort_values("timestamp")
    if series.empty:
        return _finalize_forecast({**null_result, "data_note": "no_data_points"}, resource_history, feature)

    current = float(series[feature].iloc[-1])
    
    # Compute pooled prior early — needed for thin resources
    slope_prior_artifact = artifact.get("slope_prior") if artifact else None
    prior_slope_norm = None
    pool_size = 0
    if type_priors_cache is not None and resource_canonical_type:
        prior_slope_norm, pool_size = compute_type_specific_slope_prior(
            type_priors_cache, resource_canonical_type, feature
        )
    if prior_slope_norm is None:
        prior_slope_norm = predict_slope_prior(slope_prior_artifact, feature, resource_history, row)

    # For very thin series, use pooled prior to generate bounds
    if len(series) < MIN_POINTS_FOR_FORECAST:
        if prior_slope_norm is not None:
            # Use pooled prior to forecast
            y_vals = series[feature].astype(float)
            y_range = float(y_vals.max() - y_vals.min()) if len(y_vals) > 1 else abs(current) * 0.1
            if y_range == 0:
                y_range = abs(current) * 0.1 if current != 0 else 1.0
            
            # Generate forecast from prior slope
            prior_slope_raw = prior_slope_norm * y_range
            pooling_tier = _pooling_tier(True, resource_canonical_type)
            
            # Compute synthetic R² based on pool size (confidence in pooled data)
            # More resources in pool → higher confidence in the prior
            synthetic_r2 = min(0.85, 0.3 + (pool_size / 20.0))  # Cap at 0.85
            
            r = {**null_result, 
                 "current_value": clean(current),
                 "data_note": f"insufficient_points_{len(series)}_pooled",
                 "pooling_tier": pooling_tier,
                 "forecast_reliability": "high" if pool_size >= 10 else ("medium-high" if pool_size >= 5 else "medium"),
                 "forecast_method": "pooled_type_prior",
                 "trend_direction": "increasing" if prior_slope_raw > 0.01 else ("decreasing" if prior_slope_raw < -0.01 else "stable"),
                 "trend_fit_r2": synthetic_r2,
            }
            
            # Forecast using prior slope
            hours_map = {"1h": 1, "6h": 6, "24h": 24, "7d": 7*24, "30d": 30*24}
            for label, hrs in hours_map.items():
                forecast_val = current + prior_slope_raw * hrs
                margin = abs(prior_slope_raw * hrs * 0.25)  # ±25% CI
                r[f"forecast_{label}"] = clean(forecast_val)
                r[f"forecast_{label}_lower"] = clean(forecast_val - margin)
                r[f"forecast_{label}_upper"] = clean(forecast_val + margin)
            
            return _finalize_forecast(r, resource_history, feature)
        else:
            # No prior available, return point estimates
            r = {**null_result, "current_value": clean(current), "data_note": f"insufficient_points_{len(series)}"}
            for h in ("1h", "6h", "24h", "7d", "30d"):
                r[f"forecast_{h}"] = clean(current)
                r[f"forecast_{h}_lower"] = clean(current)
                r[f"forecast_{h}_upper"] = clean(current)
            return _finalize_forecast(r, resource_history, feature)

    mixed_stats = _detect_mixed_stats(resource_history, feature)
    data_note_suffix = None  # set by the adaptive-selection branch below when it runs; None elsewhere
    series, clean_status = _clean_series(series, feature)

    if len(series) < 2:
        r = {**null_result, "current_value": clean(current), "data_note": "all_zeros_after_cleaning"}
        for h in ("1h", "6h", "24h", "7d", "30d"):
            r[f"forecast_{h}"] = clean(current)
            r[f"forecast_{h}_lower"] = clean(current)
            r[f"forecast_{h}_upper"] = clean(current)
        return _finalize_forecast(r, resource_history, feature)

    x = (series["timestamp"] - series["timestamp"].min()).dt.total_seconds() / 3600.0
    y = series[feature].astype(float)
    is_pct = feature and (feature.endswith("_pct") or feature in {"canonical_cpu_pct", "canonical_storage_used_pct"})

    y_min, y_max = float(y.min()), float(y.max())
    y_range = y_max - y_min
    y_norm = (y - y_min) / y_range if y_range > 0 else pd.Series(np.zeros(len(y)), index=y.index)

    # Note: prior_slope_norm was already computed at the top of the function
    # for thin resources. It's reused here for thicker series too.

    if y_norm.std() < MIN_STD_NORM:
        # For flat signals, still check if we can apply pooling from well-populated types
        pooling_tier = "individual"
        forecast_reliability = "high" if clean_status == "idle" else "low"
        trend_fit_r2 = 1.0 if clean_status == "idle" else 0.0
        
        # If we have a pooled prior available, use it to widen confidence intervals
        if prior_slope_norm is not None and y_range > 0:
            # Use pooled prior to estimate uncertainty even though local signal is flat
            prior_slope_raw = prior_slope_norm * y_range
            # Confidence intervals based on pooled uncertainty
            margin = abs(prior_slope_raw) * 0.5  # Conservative margin for flat signal
            pooling_tier = _pooling_tier(True, resource_canonical_type)
            forecast_reliability = "high" if pool_size >= 10 else ("medium-high" if pool_size >= 5 else "medium")
            # Use pool size to set synthetic R²
            trend_fit_r2 = min(0.75, 0.2 + (pool_size / 25.0))  # Lower ceiling for flat signals
        
        r = {
            **null_result,
            "current_value":        clean(current),
            "trend_direction":      "stable",
            "trend_fit_r2":         trend_fit_r2,
            "forecast_reliability": forecast_reliability,
            "forecast_method":      "flat_signal",
            "data_note":            "mixed_stat_fields" if mixed_stats else clean_status,
            "pooling_tier":         pooling_tier,
        }
        for h in ("1h", "6h", "24h", "7d", "30d"):
            r[f"forecast_{h}"] = clean(current)
            if prior_slope_norm is not None and y_range > 0:
                margin = abs(prior_slope_norm * y_range * 0.5)
                r[f"forecast_{h}_lower"] = clean(current - margin)
                r[f"forecast_{h}_upper"] = clean(current + margin)
            else:
                r[f"forecast_{h}_lower"] = clean(current)
                r[f"forecast_{h}_upper"] = clean(current)
        return _finalize_forecast(r, resource_history, feature)

    x_arr = x.values
    y_arr = y.values
    current_y_val = float(y.iloc[-1])
    observed_min = float(y.min())
    observed_max = float(y.max())

    # ── Bursty metrics, enough history: seasonal-naive baseline ───────────
    # METHODOLOGY UPGRADE (v9): the old approach forced an exponential-
    # smoothing trend line onto metrics that are fundamentally diurnal and
    # non-trending (network traffic, IOPS, request counts) -- fitting a
    # slope to bursty, cyclical data produces a low-quality fit almost by
    # construction, no matter how the confidence formula is tuned. This is
    # the standard alternative real forecasting teams use for this kind of
    # metric: forecast "what usually happens at this point in the cycle"
    # (seasonal_phase_forecast in forecast_math.py) instead of "where the
    # trend line points." It makes no claim about long-run drift by design
    # -- long_run_drift_direction() below is a coarse, separate check so
    # genuine structural growth underneath the noise isn't silently lost.
    if feature in BURSTY_METRICS and not mixed_stats and len(series) >= MIN_POINTS_FOR_SEASONAL_BASELINE:
        hours_map = {"1h": 1, "6h": 6, "24h": 24, "7d": 7 * 24, "30d": 30 * 24}
        offsets = np.array(list(hours_map.values()), dtype=float)
        point, lower, upper = seasonal_phase_forecast(x_arr, y_arr, offsets, period=24)
        if is_pct:
            point = np.clip(point, 0.0, 100.0)
            lower = np.clip(lower, 0.0, 100.0)
            upper = np.clip(upper, 0.0, 100.0)
        else:
            point = np.clip(point, 0.0, None)
            lower = np.clip(lower, 0.0, None)
            upper = np.clip(upper, 0.0, None)

        forecasts = {}
        for label, p_val, l_val, u_val in zip(hours_map.keys(), point, lower, upper):
            forecasts[f"forecast_{label}"] = round(float(p_val), 3)
            forecasts[f"forecast_{label}_lower"] = round(float(l_val), 3)
            forecasts[f"forecast_{label}_upper"] = round(float(u_val), 3)

        # In-sample diagnostic only (how well the phase-bucket medians
        # explain the history they were built from) -- the authoritative
        # accuracy read for this branch is the out-of-sample backtest run
        # in _finalize_forecast (evaluation_basis == "out_of_sample_backtest"
        # whenever there's enough data, which there is here by construction).
        phase_hist = np.mod(np.round(x_arr).astype(int), 24)
        bucket_median = {p: float(np.median(y_arr[phase_hist == p])) for p in np.unique(phase_hist)}
        in_sample_pred = np.array([bucket_median[p] for p in phase_hist])
        ss_res = float(np.sum((y_arr - in_sample_pred) ** 2))
        ss_tot = float(np.sum((y_arr - y_arr.mean()) ** 2))
        confidence = 0.0 if ss_tot < 1e-9 else max(0.0, min(1.0, 1.0 - ss_res / ss_tot))

        trend = long_run_drift_direction(y_arr)
        method = "seasonal_naive_baseline"
        pooling_tier = "individual"

    # ── Bursty metrics, short series: exponential smoothing (fallback) ────
    elif feature in BURSTY_METRICS and not mixed_stats:
        alpha = 0.3
        smoothed_level = _exp_smooth_forecast(y_arr, alpha=alpha)
        recent_n = max(3, len(y) // 4)
        y_recent = y_arr[-recent_n:]
        x_recent = np.arange(len(y_recent), dtype=float)
        slope_recent, _ = safe_polyfit_slope(x_recent, y_recent)

        interval_h = float(x.iloc[-1] - x.iloc[-2]) if len(x) > 1 else 1.0

        # Bootstrap CI on recent slope
        _, slope_lo, slope_hi = _bootstrap_slope_ci(x_recent, y_recent)

        # Shrink toward the pooled prior when this series is thin — the raw
        # slope on <20 points of a bursty metric is mostly noise.
        slope_recent, shrink_weight, shrink_applied = _shrink_toward_prior(
            slope_recent, len(series), prior_slope_norm, y_range
        )
        if shrink_applied:
            slope_lo, _, _ = _shrink_toward_prior(slope_lo, len(series), prior_slope_norm, y_range)
            slope_hi, _, _ = _shrink_toward_prior(slope_hi, len(series), prior_slope_norm, y_range)

        pooling_tier = _pooling_tier(shrink_applied, resource_canonical_type)

        def _exp_forecast(hours, slope_used):
            steps = hours / max(interval_h, 0.001)
            val = smoothed_level + slope_used * steps
            return max(0.0, min(100.0, val)) if is_pct else max(0.0, val)

        # R² of exp-smoothed fit on normalised series
        y_norm_vals = y_norm.values
        smoothed_series = [y_norm_vals[0]]
        for v in y_norm_vals[1:]:
            smoothed_series.append(alpha * v + (1 - alpha) * smoothed_series[-1])
        smoothed_arr = np.array(smoothed_series)
        ss_res = float(np.sum((y_norm_vals - smoothed_arr) ** 2))
        ss_tot = float(np.sum((y_norm_vals - y_norm_vals.mean()) ** 2))
        confidence = 0.0 if ss_tot == 0 else max(0.0, min(1.0, 1.0 - ss_res / ss_tot))

        ref = smoothed_level
        if slope_recent > 0.05 * (ref if ref > 0 else 1):
            trend = "increasing"
        elif slope_recent < -0.05 * (ref if ref > 0 else 1):
            trend = "decreasing"
        else:
            trend = "stable"

        method = "shrunk_slope_xgboost_prior" if shrink_applied else "exponential_smoothing"

        hours_map = {"1h": 1, "6h": 6, "24h": 24, "7d": 7*24, "30d": 30*24}
        forecasts = {}
        for label, hrs in hours_map.items():
            forecasts[f"forecast_{label}"]       = round(_exp_forecast(hrs, slope_recent), 3)
            forecasts[f"forecast_{label}_lower"] = round(_exp_forecast(hrs, slope_lo), 3)
            forecasts[f"forecast_{label}_upper"] = round(_exp_forecast(hrs, slope_hi), 3)

    # ── Capacity/utilisation metrics: adaptive method selection ───────────
    else:
        hours_map = {"1h": 1, "6h": 6, "24h": 24, "7d": 7*24, "30d": 30*24}
        offsets = np.array(list(hours_map.values()), dtype=float)

        # v9.2: this branch used to unconditionally fit STL-trend-plus-
        # bootstrap-CI regardless of whether that model actually predicts
        # well for THIS resource. The backtest module (backtest.py) now
        # picks whichever of 5 candidate methods (naive persistence,
        # moving average, linear regression, STL trend, seasonal-naive)
        # wins a leakage-safe nested validation -- imported and reused
        # directly here (not reimplemented) so the live forecast and the
        # honesty check in _finalize_forecast are guaranteed to agree on
        # what "the model" means for this resource. See CHANGELOG_v9.2:
        # this was motivated by finding canonical_storage_capacity_raw's
        # median out-of-sample MASE was 1.42 (worse than doing nothing)
        # under the old always-fit-a-trend approach.
        #
        # Falls back to the original STL/bootstrap logic below when there
        # isn't enough history for select_forecast_method's own internal
        # nested holdout (same MIN_TRAIN_POINTS/MIN_HOLDOUT_POINTS guard
        # backtest.py uses for the real evaluation) -- this is the same
        # "don't pretend to validate on too little data" principle applied
        # consistently, not a silent quality regression for thin series.
        adaptive_used = False
        pooling_tier = "individual"
        shrink_weight = None
        try:
            _adaptive_k = _choose_holdout_size(len(x_arr))
            if _adaptive_k > 0:
                # select_forecast_method's candidates compute (test_x -
                # base_x) internally, where base_x is anchored at
                # train_x[-1] -- so test_x must be in the SAME absolute
                # hours-since-series-start frame as x_arr, not raw
                # horizon offsets. backtest.py gets this right for free
                # because its test_x already comes from the same
                # x_all array as train_x. Here we're constructing test_x
                # from scratch for the live forecast, so it must be built
                # the same way: last historical x-value + each offset.
                test_x_abs = x_arr[-1] + offsets
                point, lower, upper, resid_floored, selected_method, nested_skill, sel_diag = _select_forecast_method(
                    x_arr, y_arr, test_x_abs, period=24
                )

                def _clamp_adaptive(val):
                    if is_pct:
                        floor = max(0.0, observed_min * 0.9)
                        return max(floor, min(100.0, val))
                    return max(0.0, val)

                point = np.array([_clamp_adaptive(v) for v in point])
                lower = np.array([_clamp_adaptive(v) for v in lower])
                upper = np.array([_clamp_adaptive(v) for v in upper])
                lower, upper = np.minimum(lower, upper), np.maximum(lower, upper)

                forecasts = {}
                for label, p_val, l_val, u_val in zip(hours_map.keys(), point, lower, upper):
                    forecasts[f"forecast_{label}"] = round(float(p_val), 3)
                    forecasts[f"forecast_{label}_lower"] = round(float(l_val), 3)
                    forecasts[f"forecast_{label}_upper"] = round(float(u_val), 3)

                # Trend label from the actual selected method's own 24h
                # projection vs. current -- method-agnostic, doesn't assume
                # a slope exists (naive/moving-average have none).
                signal_range = max(y_arr.max() - y_arr.min(), 1.0)
                idx_24h = list(hours_map.keys()).index("24h")
                delta_24h = float(point[idx_24h]) - current_y_val
                trend_threshold = 0.02 if len(series) >= LONG_SERIES_THRESHOLD else 0.05
                if delta_24h / signal_range > trend_threshold:
                    trend = "increasing"
                elif delta_24h / signal_range < -trend_threshold:
                    trend = "decreasing"
                else:
                    trend = "stable"

                # nested_relative_skill is None when the winner tied with
                # naive on a naive_mae of ~0 (nothing to divide by) -- treat
                # that as "can't tell yet," not as zero confidence.
                confidence = 0.5 if nested_skill is None else nested_skill
                method = f"adaptive_{selected_method}"
                data_note_suffix = "adaptive_selection_no_pooling_shrink"
                if sel_diag.get("structural_break_detected"):
                    # v9.3: pre-break history was discarded for this fit --
                    # see detect_last_structural_break()'s docstring for the
                    # honesty caveat on how this threshold was chosen.
                    method = f"{method}_post_break"
                    data_note_suffix = f"{data_note_suffix}+structural_break_trimmed_{sel_diag['structural_break_trimmed_n']}pts"
                adaptive_used = True
        except Exception as _adaptive_err:
            adaptive_used = False
            # TEMPORARY DIAGNOSTIC (remove once root cause is found): the
            # except block below was silently swallowing whatever's
            # actually breaking adaptive selection on real data -- 0 of
            # 1179 real predictions used it despite passing on all
            # synthetic tests. Printing to stderr so pipeline logs capture
            # it without touching stdout/the actual prediction output.
            import sys, traceback
            print(
                f"[ADAPTIVE_SELECTION_FAILED] resource_id={row.get('resource_id')} "
                f"category={category} feature={feature} n={len(x_arr)}: {_adaptive_err!r}",
                file=sys.stderr,
            )
            traceback.print_exc(file=sys.stderr)

        if not adaptive_used:
            # ── Original STL + bootstrap CI logic (pre-v9.2), unmodified ──
            # Kept verbatim as the fallback for series too short for
            # select_forecast_method's own nested-holdout requirement, so
            # thin series don't silently lose quality relative to before.
            stl_trend, stl_seasonal, stl_residual, r2_stl = None, None, None, None
            slope_med = slope_lo = slope_hi = None
            confidence = 0.0
            stl_period = 24
            seasonal_last_cycle = None
            pooling_tier = "individual"

            if len(series) >= LONG_SERIES_THRESHOLD and HAS_STL:
                stl_trend, stl_seasonal, stl_residual, r2_stl = _stl_decompose(x_arr, y_arr, period=stl_period)

            if stl_trend is not None and r2_stl is not None:
                # Use STL trend component: fit a line through it + bootstrap CI
                trend_x = np.linspace(0, x_arr[-1], len(stl_trend))
                slope_med, slope_lo, slope_hi = _bootstrap_slope_ci(trend_x, stl_trend)
                confidence = r2_stl
                method = "stl_decomposition"
                if stl_seasonal is not None and len(stl_seasonal) >= stl_period:
                    # Most recent full cycle — used as the "typical" repeating
                    # pattern (e.g. daily dip/peak) when projecting forward.
                    seasonal_last_cycle = stl_seasonal[-stl_period:]
            else:
                # Short series or STL unavailable: bootstrap directly on raw series
                slope_med, slope_lo, slope_hi = _bootstrap_slope_ci(x_arr, y_arr)
                # R² of linear fit on normalised series
                slope_norm, intercept_norm = safe_polyfit_slope(x_arr, y_norm.values)
                yhat_norm = slope_norm * x_arr + intercept_norm
                ss_res = float(np.sum((y_norm.values - yhat_norm) ** 2))
                ss_tot = float(np.sum((y_norm.values - y_norm.values.mean()) ** 2))
                if ss_tot < 1e-9:
                    # Essentially flat series — maximally predictable, not
                    # unexplainable. See the matching fix in _stl_decompose.
                    confidence = 1.0
                else:
                    confidence = max(0.0, min(1.0, 1.0 - ss_res / ss_tot))
                method = "linear_regression_bootstrap"

                # Shrink toward the pooled prior for thin series — this is the
                # case that mattered most: STL isn't available yet (< 72 points)
                # and a handful of points give an OLS slope with huge variance.
                slope_med, shrink_weight, shrink_applied = _shrink_toward_prior(
                    slope_med, len(series), prior_slope_norm, y_range
                )
                if shrink_applied:
                    slope_lo, _, _ = _shrink_toward_prior(slope_lo, len(series), prior_slope_norm, y_range)
                    slope_hi, _, _ = _shrink_toward_prior(slope_hi, len(series), prior_slope_norm, y_range)
                    method = "shrunk_slope_xgboost_prior"

                pooling_tier = _pooling_tier(shrink_applied, resource_canonical_type)

            # Trend threshold: tighter for long STL-decomposed series
            signal_range = max(y_arr.max() - y_arr.min(), 1.0)
            slope_norm_equiv = slope_med / signal_range
            trend_threshold = 0.02 if len(series) >= LONG_SERIES_THRESHOLD else 0.05

            if slope_norm_equiv > trend_threshold:
                trend = "increasing"
            elif slope_norm_equiv < -trend_threshold:
                trend = "decreasing"
            else:
                trend = "stable"

            # Diurnal override: only when confidence is LOW and projection is extreme
            projected_24h = current_y_val + slope_med * 24
            if is_pct and projected_24h < observed_min * 0.75 and confidence < 0.5:
                trend = "stable"

            # A "stable" resource that clusters TIGHTLY around its own mean is
            # highly predictable ("it'll stay about where it is") even though a
            # sloped-line R² can't see that (there's no slope to fit). This is
            # deliberately a strict gate, not a general "flat = confident" rule
            # — a resource that's merely non-trending but genuinely noisy
            # (values scattered widely with no pattern) should NOT get a
            # confidence boost just for lacking a slope; that noise really is
            # hard to forecast and low confidence is the honest answer for it.
            if trend == "stable":
                std_y = float(np.std(y_arr))
                scale = max(abs(float(np.mean(y_arr))), 1.0)
                coeff_of_variation = std_y / scale
                CV_TIGHT_THRESHOLD = 0.05  # values stay within ~5% of their own mean
                if coeff_of_variation < CV_TIGHT_THRESHOLD:
                    stability_confidence = 1.0 - (coeff_of_variation / CV_TIGHT_THRESHOLD) * 0.2
                    confidence = max(confidence, stability_confidence)

            def _seasonal_delta(hours):
                """Difference between the seasonal effect `hours` ahead and the
                seasonal effect right now — added on top of the trend
                extrapolation so a resource with a real repeating pattern (e.g.
                a daily dip) actually forecasts that dip, instead of a flat
                trend line that ignores it. Zero when there's no seasonal
                component (short series, or STL unavailable).
                """
                if seasonal_last_cycle is None:
                    return 0.0
                k_now    = stl_period - 1
                k_future = (stl_period - 1 + int(round(hours))) % stl_period
                return float(seasonal_last_cycle[k_future] - seasonal_last_cycle[k_now])

            # BUG FIX (methodology, v9): forecast_*_lower/_upper used to come
            # entirely from the bootstrap CI on the SLOPE -- i.e. "how
            # uncertain is the trend line's angle." That leaves out the other,
            # usually larger source of uncertainty: how much any individual
            # future point actually wobbles around that trend line even when
            # the slope itself is estimated perfectly. Running a real
            # out-of-sample backtest (backtest.py) on this exact model exposed
            # it directly: interval coverage came back 0% on a clean synthetic
            # trend with ordinary noise -- the band was so narrow it missed
            # essentially every held-out point. Residual noise (the STL
            # residual's own spread, or the regression residual around the
            # slope line) is now added on top of the slope-uncertainty band.
            if stl_trend is not None and r2_stl is not None and stl_residual is not None and len(stl_residual):
                resid_std = float(np.std(stl_residual))
            else:
                resid_raw = y_arr - (current_y_val + slope_med * (x_arr - x_arr[-1]))
                resid_std = float(np.std(resid_raw))
            Z_NOISE = 1.28  # ~80% one-sided normal quantile -- matches the 20th/80th
                             # percentile band used by the seasonal-naive branch above,
                             # so both forecast families target a comparable interval width.
            noise_margin = Z_NOISE * resid_std

            def _lin_forecast(hours, slope_used):
                val = current_y_val + slope_used * hours + _seasonal_delta(hours)
                if is_pct:
                    floor = max(0.0, observed_min * 0.9)
                    return max(floor, min(100.0, val))
                return max(0.0, val)

            forecasts = {}

            if confidence < 0.25:
                # Clamp long-horizon forecasts to observed range
                obs_lo, obs_hi = observed_min, observed_max
                def _clamp(val):
                    return max(obs_lo * 0.9, min(obs_hi * 1.1, val)) if is_pct else max(0.0, val)
                for label, hrs in hours_map.items():
                    forecasts[f"forecast_{label}"]       = round(_clamp(_lin_forecast(hrs, slope_med)), 3)
                    forecasts[f"forecast_{label}_lower"] = round(_clamp(_lin_forecast(hrs, slope_lo) - noise_margin), 3)
                    forecasts[f"forecast_{label}_upper"] = round(_clamp(_lin_forecast(hrs, slope_hi) + noise_margin), 3)
            else:
                def _clamp_bound(val):
                    if is_pct:
                        floor = max(0.0, observed_min * 0.9)
                        return max(floor, min(100.0, val))
                    return max(0.0, val)
                for label, hrs in hours_map.items():
                    forecasts[f"forecast_{label}"]       = round(_lin_forecast(hrs, slope_med), 3)
                    forecasts[f"forecast_{label}_lower"] = round(_clamp_bound(_lin_forecast(hrs, slope_lo) - noise_margin), 3)
                    forecasts[f"forecast_{label}_upper"] = round(_clamp_bound(_lin_forecast(hrs, slope_hi) + noise_margin), 3)

    # ── Reliability label ──────────────────────────────────────────────────
    if confidence >= 0.65:
        forecast_reliability = "high"
    elif confidence >= 0.25:
        forecast_reliability = "moderate"
    else:
        forecast_reliability = "low"

    # Pooling tier is a downgrade-only signal on top of trend_fit_r2 (never
    # an upgrade): a forecast that's mostly borrowed from a thin,
    # single-provider type's pooled fit shouldn't get to claim "high"
    # reliability just because the blended fit looks clean, even though its
    # trend_fit_r2 alone can't see how much of the forecast was borrowed.
    # This mirrors the same never-upgrade-only-downgrade principle applied
    # to the AutoGluon backtest-based reliability path elsewhere in this
    # project.
    if pooling_tier == "category_pooled" and forecast_reliability == "high":
        forecast_reliability = "moderate"
    elif pooling_tier == "type_pooled" and forecast_reliability == "high" and shrink_weight < 0.5:
        # Even a well-populated type prior shouldn't carry a "high" label
        # when the individual resource itself contributed less than half
        # the blend.
        forecast_reliability = "moderate"

    base_data_note = "mixed_stat_fields" if mixed_stats else clean_status
    if data_note_suffix:
        data_note = f"{base_data_note}+{data_note_suffix}" if base_data_note and base_data_note != "ok" else data_note_suffix
    else:
        data_note = base_data_note

    r = {
        "metric_name":          metric_name,
        "canonical_feature":    feature,
        "current_value":        clean(current),
        **{k: v for k, v in forecasts.items()},
        "trend_direction":      trend,
        "trend_fit_r2":         round(confidence, 3),
        "forecast_reliability": forecast_reliability,
        "forecast_method":      method,
        "data_note":            data_note,
        "canonical_resource_type": resource_canonical_type,
        "pooling_tier":         pooling_tier,
    }
    return _finalize_forecast(r, resource_history, feature)


def _estimate_error_proxy_metrics(resource_history, feature):
    """Estimate lightweight MAE/RMSE/MAPE-style error proxies from the series fit.

    These are not historical backtest metrics, but they give each forecast an
    internal accuracy signal based on how tightly the recent series follows a
    simple linear trend. That makes MAE/RMSE/MAPE influence the confidence
    score even before future actuals are available.
    """
    try:
        if feature not in resource_history.columns:
            return {"mae_proxy": None, "rmse_proxy": None, "mape_proxy": None}

        values = pd.to_numeric(resource_history[feature], errors="coerce").dropna().astype(float)
        if len(values) < 5:
            return {"mae_proxy": None, "rmse_proxy": None, "mape_proxy": None}

        x = np.arange(len(values), dtype=float)
        slope, intercept = safe_polyfit_slope(x, values)
        fitted = slope * x + intercept
        residuals = values - fitted

        scale = max(float(np.mean(np.abs(values))), 1.0)
        mae = float(np.mean(np.abs(residuals)) / scale)
        rmse = float(np.sqrt(np.mean(residuals ** 2)) / scale)

        # BUG FIX: this used to divide each residual by that individual
        # point's own value (floored at 1.0) -- textbook MAPE. On raw
        # byte-scale features (canonical_storage_capacity_raw, net_in/out
        # bytes, etc.) individual observations near zero blow this up into
        # the billions/trillions, which then corrupted accuracy_proxy_score
        # and every downstream average. mae/rmse above already use a single
        # robust series-level `scale` instead of a per-point floor -- mape
        # now uses that same scale so all three proxies are on a consistent,
        # bounded basis. Still expressed as a percentage of the series'
        # typical magnitude, just no longer divided point-by-point.
        mape = float(np.mean(np.abs(residuals)) / scale * 100)
        mape = min(mape, 1000.0)  # defensive ceiling; quality score already clamps at 100

        return {
            "mae_proxy": round(mae, 4),
            "rmse_proxy": round(rmse, 4),
            "mape_proxy": round(mape, 2),
        }
    except Exception:
        return {"mae_proxy": None, "rmse_proxy": None, "mape_proxy": None}


def _compute_prediction_confidence_metrics(forecasts, model_r2, resource_history, feature, reliability, method):
    """Compute confidence and reliability metrics for the forecast.
    
    Returns dict with:
    - prediction_confidence_score: 0-100 composite score
    - forecast_uncertainty_pct: Width of confidence interval as % of current value
    - data_completeness_score: 0-100 based on completeness and consistency
    - mae_proxy / rmse_proxy / mape_proxy: error-based accuracy proxies
    - accuracy_proxy_score: combined error-based accuracy signal
    """
    metrics = {}
    error_metrics = _estimate_error_proxy_metrics(resource_history, feature)
    metrics.update(error_metrics)
    
    # Extract confidence interval width from 24h forecast
    if "forecast_24h" in forecasts and "forecast_24h_lower" in forecasts and "forecast_24h_upper" in forecasts:
        forecast_val = forecasts.get("forecast_24h")
        lower = forecasts.get("forecast_24h_lower")
        upper = forecasts.get("forecast_24h_upper")
        
        if all(v is not None for v in [forecast_val, lower, upper]):
            try:
                forecast_val = float(forecast_val)
                lower = float(lower)
                upper = float(upper)
                
                ci_width = upper - lower
                ref_val = abs(forecast_val) if forecast_val != 0 else 1.0
                ci_pct = (ci_width / ref_val * 100) if ref_val > 0 else 0
                metrics["forecast_uncertainty_pct"] = round(min(ci_pct, 200), 1)  # Cap at 200%
                
                # Narrower CI → higher confidence
                ci_confidence = max(0, 100 - ci_pct)
            except (TypeError, ValueError):
                ci_confidence = 50
        else:
            ci_confidence = 50
    else:
        ci_confidence = 50
    
    # Data quality score based on resource history
    try:
        if feature in resource_history.columns:
            data_series = resource_history[feature]
            non_null_count = data_series.notna().sum()
            total_count = len(data_series)
            completeness = (non_null_count / total_count) if total_count > 0 else 0
            
            # Non-null > 80% = good
            data_quality = min(100, max(0, completeness * 125))
        else:
            data_quality = 0
    except Exception:
        data_quality = 50
    
    metrics["data_completeness_score"] = round(data_quality, 1)
    
    reliability_weight = {
        "high": 0.9,
        "medium-high": 0.75,
        "medium": 0.6,
        # BUG FIX: the main STL/regression/exp-smoothing path in
        # behavioral_forecast() labels its mid-tier reliability "moderate",
        # not "medium" -- that string was never a key in this dict, so
        # every one of those predictions silently fell through to the 0.5
        # default instead of the intended weight. "moderate" is mapped to
        # the same weight as "medium" here since they represent the same
        # confidence band.
        "moderate": 0.6,
        "low": 0.3,
    }.get(reliability, 0.5)
    
    method_weight = {
        "stl_decomposition": 0.95,
        "linear_regression_bootstrap": 0.85,
        "exponential_smoothing": 0.8,
        "shrunk_slope_xgboost_prior": 0.7,
        "pooled_type_prior": 0.65,
        "flat_signal": 0.4,
    }.get(method, 0.5)

    mae_proxy = error_metrics.get("mae_proxy")
    rmse_proxy = error_metrics.get("rmse_proxy")
    mape_proxy = error_metrics.get("mape_proxy")

    if mae_proxy is None or rmse_proxy is None or mape_proxy is None:
        error_quality_score = 50.0
    else:
        mae_quality = max(0.0, 100.0 - min(100.0, mae_proxy * 100.0))
        rmse_quality = max(0.0, 100.0 - min(100.0, rmse_proxy * 100.0))
        mape_quality = max(0.0, 100.0 - min(100.0, mape_proxy))
        error_quality_score = round((mae_quality + rmse_quality + mape_quality) / 3.0, 1)

    metrics["accuracy_proxy_score"] = round(error_quality_score, 1)
    
    # Composite: weighted average of multiple factors
    # - Model fit (R²): 45%
    # - Reliability label: 25%
    # - Method quality: 15%
    # - Error-based accuracy proxies (MAE/RMSE/MAPE): 15%
    composite_score = (
        (model_r2 * 100 * 0.45) +
        (reliability_weight * 100 * 0.25) +
        (method_weight * 100 * 0.15) +
        (error_quality_score * 0.15)
    )
    metrics["prediction_confidence_score"] = round(min(100, max(0, composite_score)), 1)
    
    return metrics


def choose_forecast_feature(category, resource_history):
    preferred = PRIMARY_FORECAST_FEATURE.get(category)
    if preferred in resource_history.columns and resource_history[preferred].notna().any():
        return preferred
    # BUG FIX: previously iterated canonical_features(category), i.e. plain
    # dict-insertion order, which for Network meant raw unbounded byte
    # counters got picked ahead of bounded utilization/health percentages
    # for any resource lacking the primary feature (the vast majority of
    # vpc_network resources). forecast_feature_fallback_order() returns a
    # curated priority list for Network/Storage that prefers meaningful,
    # bounded signals first; other categories are unaffected.
    for feature in forecast_feature_fallback_order(category):
        if feature in resource_history.columns and resource_history[feature].notna().any():
            return feature
    return preferred


def latest_rows(df):
    if df.empty:
        return df
    idx = df.sort_values("timestamp").groupby(["resource_id", "category"], dropna=False).tail(1).index
    return df.loc[idx].copy()


def resource_metrics_present(category, history):
    """Every canonical metric this resource has EVER reported for this
    category (i.e. has at least one non-null value in its full history),
    in canonical_features() order. Used to explode one prediction per
    (resource_id, category, metric) instead of bundling every metric a
    resource reports into a single prediction.
    """
    return [
        f for f in canonical_features(category)
        if f in history.columns and history[f].notna().any()
    ]


def consolidated_row(category, resource_history, base_row):
    """Latest value of each canonical field. Derives cpu_pct from idle if missing."""
    row = base_row.copy()
    # Derive missing CPU signal before iterating
    if "canonical_cpu_pct" in resource_history.columns and "canonical_cpu_idle_pct" in resource_history.columns:
        cpu_null = resource_history["canonical_cpu_pct"].isna()
        idle_ok  = resource_history["canonical_cpu_idle_pct"].notna()
        resource_history = resource_history.copy()
        resource_history.loc[cpu_null & idle_ok, "canonical_cpu_pct"] = (
            100.0 - resource_history.loc[cpu_null & idle_ok, "canonical_cpu_idle_pct"]
        )
    for feature in canonical_features(category):
        if feature not in resource_history.columns:
            row[feature] = None
            row[f"{feature}_available"] = False
            continue
        s = resource_history[["timestamp", feature]].dropna().sort_values("timestamp")
        row[feature] = None if s.empty else s[feature].iloc[-1]
        row[f"{feature}_available"] = not s.empty
    return row


def category_payload(category, resource_history, row, risk_score):
    if category == "Compute":
        cpu = slope_forecast(resource_history, "canonical_cpu_pct", [80, 90]) if "canonical_cpu_pct" in resource_history else {}
        mem = row.get("canonical_mem_used_pct")
        mem_class = "UNKNOWN"
        if mem is not None and not pd.isna(mem):
            mem_class = "CRITICAL" if mem >= 90 else "WARNING" if mem >= 75 else "OK"
        return {
            "risk_score":                    round(risk_score, 2),
            "cpu_saturation_forecast":       cpu,
            "memory_pressure_class":         mem_class,
            "oom_probability":               round(max(0.0, min(1.0, ((mem or 0) - 70) / 30)), 3) if mem_class != "UNKNOWN" else None,
            "iowait_spike_flag":             is_real(row.get("canonical_iowait_pct")) and row.get("canonical_iowait_pct") >= 20,
            "health_failure_probability_24h": round(risk_score / 100.0, 3),
            "available_signals":             availability(category, row),
        }
    if category == "Network":
        return {
            "risk_score":                    round(risk_score, 2),
            "ddos_probability":              round(min(1.0, positive(row.get("canonical_ddos_signal")) * 0.8 + risk_score / 500), 3),
            "connection_saturation_forecast": slope_forecast(resource_history, "canonical_active_connections", [10000, 25000]) if "canonical_active_connections" in resource_history else {},
            "subnet_capacity_forecast":      slope_forecast(resource_history, "canonical_subnet_util_pct", [80, 90]) if "canonical_subnet_util_pct" in resource_history else {},
            "health_pct":                    clean(row.get("canonical_health_pct")),
            "available_signals":             availability(category, row),
        }
    if category == "Storage":
        burst = row.get("canonical_burst_balance_pct")
        return {
            "risk_score":                round(risk_score, 2),
            "capacity_exhaustion_forecast": slope_forecast(resource_history, "canonical_storage_used_pct", [80, 90, 95]) if "canonical_storage_used_pct" in resource_history else {},
            "io_anomaly_score":          round(risk_score / 100.0, 3),
            "burst_exhaustion_risk":     None if burst is None or pd.isna(burst) else round(max(0.0, (30 - burst) / 30), 3),
            "availability_score":        clean(row.get("availability_pct")),
            "backup_failure_risk":       positive(row.get("backup_health_event")),
            "available_signals":         availability(category, row),
        }
    if category == "Container":
        return {
            "risk_score":                    round(risk_score, 2),
            "node_cpu_pct":                  clean(row.get("canonical_node_cpu_pct")),
            "node_mem_pct":                  clean(row.get("canonical_node_mem_pct")),
            "node_disk_pct":                 clean(row.get("canonical_node_disk_pct")),
            "unschedulable_pods":            clean(row.get("canonical_unschedulable_pods")),
            "unneeded_nodes":                clean(row.get("canonical_unneeded_nodes")),
            "api_inflight_requests":         clean(row.get("canonical_api_inflight_requests")),
            "health_failure_probability_24h": round(risk_score / 100.0, 3),
            "available_signals":             availability(category, row),
        }
    if category == "Databases":
        return {
            "risk_score":            round(risk_score, 2),
            "db_cpu_pct":            clean(row.get("canonical_db_cpu_pct")),
            "db_mem_pct":            clean(row.get("canonical_db_mem_pct")),
            "db_storage_pct":        clean(row.get("canonical_db_storage_pct")),
            "db_connections":        clean(row.get("canonical_db_connections")),
            "db_dtu_used":           clean(row.get("canonical_db_dtu_used")),
            "available_signals":     availability(category, row),
        }
    if category == "Governance":
        return {
            "risk_score":            round(risk_score, 2),
            "runs_failed":           clean(row.get("canonical_runs_failed")),
            "run_failure_pct":       clean(row.get("canonical_run_failure_pct")),
            "group_instances":       clean(row.get("canonical_group_instances")),
            "throttled_requests":    clean(row.get("canonical_throttled_requests")),
            "available_signals":     availability(category, row),
        }
    if category == "Security":
        return {
            "risk_score":            round(risk_score, 2),
            "api_hit":               clean(row.get("canonical_api_hit")),
            "api_latency_ms":        clean(row.get("canonical_api_latency_ms")),
            "saturation":            clean(row.get("canonical_saturation")),
            "availability_pct":      clean(row.get("canonical_availability_pct")),
            "available_signals":     availability(category, row),
        }
    return {"risk_score": round(risk_score, 2)}


def availability(category, row):
    return {
        feature: is_true_flag(row.get(f"{feature}_available"))
        for feature in canonical_features(category)
    }


def data_completeness_score(category, row):
    features = canonical_features(category)
    if not features:
        return 0.0
    available = sum(1 for f in features if is_true_flag(row.get(f"{f}_available")))
    return round(available / len(features), 3)


def failure_predictions(category, forecast, row):
    failures = []

    def add(metric, threshold, window, value, failure_type):
        if value is not None and not pd.isna(value) and value >= threshold:
            failures.append({
                "failure_type":    failure_type,
                "metric_name":     metric,
                "threshold":       threshold,
                "window":          window,
                "predicted_value": round(float(value), 3),
            })

    metric = forecast.get("metric_name")
    if category == "Compute":
        if forecast.get("canonical_feature") == "canonical_cpu_pct":
            add(metric, 80, "1h", forecast.get("forecast_1h"), "CPU_80_PERCENT_BREACH")
            add(metric, 90, "6h", forecast.get("forecast_6h"), "CPU_90_PERCENT_BREACH")
        mem = row.get("canonical_mem_used_pct")
        if mem is not None and not pd.isna(mem) and mem >= 85:
            add("memory_used_percent", 85, "current", mem, "MEMORY_PRESSURE")
        if positive(row.get("status_check_failed")):
            add("StatusCheckFailed", 1, "current", row.get("status_check_failed"), "INSTANCE_HEALTH_CHECK_FAILURE")
    elif category == "Network":
        if forecast.get("canonical_feature") == "canonical_active_connections":
            add(metric, 10000, "1h", forecast.get("forecast_1h"), "CONNECTION_SATURATION")
        subnet = row.get("canonical_subnet_util_pct")
        if subnet is not None and not pd.isna(subnet) and subnet >= 80:
            add("subnet_utilization_percent", 80, "current", subnet, "SUBNET_CAPACITY_RISK")
        if positive(row.get("canonical_ddos_signal")):
            add("ddos_signal", 1, "current", row.get("canonical_ddos_signal"), "DDOS_SIGNAL_DETECTED")
    elif category == "Storage":
        if forecast.get("canonical_feature") == "canonical_storage_used_pct":
            add(metric, 80, "1h", forecast.get("forecast_1h"), "STORAGE_80_PERCENT_BREACH")
            add(metric, 90, "6h", forecast.get("forecast_6h"), "STORAGE_90_PERCENT_BREACH")
        burst = row.get("canonical_burst_balance_pct")
        if burst is not None and not pd.isna(burst) and burst < 20:
            failures.append({
                "failure_type":    "BURST_CREDIT_EXHAUSTION",
                "metric_name":     "burst_balance_percent",
                "threshold":       20,
                "window":          "current",
                "predicted_value": round(float(burst), 3),
            })
    return failures


def recommendations(category, row, forecast, failures, anomaly_score):
    recs = []
    trend   = forecast.get("trend_direction")
    current = forecast.get("current_value")
    reliability = forecast.get("forecast_reliability", "low")

    if category == "Compute":
        if forecast.get("canonical_feature") == "canonical_cpu_pct":
            if current is not None and current >= 80:
                recs.append("Scale up or add autoscaling capacity")
            elif current is not None and current < 25 and trend != "increasing":
                recs.append("Consider downsizing — CPU consistently underutilised")
        if row.get("canonical_mem_used_pct") is not None and not pd.isna(row.get("canonical_mem_used_pct")) and row.get("canonical_mem_used_pct") >= 85:
            recs.append("Increase memory or investigate memory-heavy processes")
        if row.get("cpu_credit_balance") is not None and not pd.isna(row.get("cpu_credit_balance")) and row.get("cpu_credit_balance") < 20:
            recs.append("Move burstable instance to larger class or non-burstable family")
        if trend == "increasing" and reliability in ("high", "moderate") and forecast.get("forecast_24h") is not None and forecast["forecast_24h"] > 80:
            recs.append(f"CPU trending up — forecast: {forecast['forecast_24h']:.0f}% in 24h (lower: {forecast.get('forecast_24h_lower', '?')}, upper: {forecast.get('forecast_24h_upper', '?')})")
    elif category == "Network":
        if any(item["failure_type"] == "DDOS_SIGNAL_DETECTED" for item in failures):
            recs.append("Enable or review DDoS protection and traffic filtering")
        if row.get("canonical_health_pct") is not None and not pd.isna(row.get("canonical_health_pct")) and row.get("canonical_health_pct") < 90:
            recs.append("Check unhealthy backend targets")
        if row.get("canonical_subnet_util_pct") is not None and not pd.isna(row.get("canonical_subnet_util_pct")) and row.get("canonical_subnet_util_pct") >= 80:
            recs.append("Expand subnet capacity or clean unused IP allocations")
    elif category == "Storage":
        if forecast.get("canonical_feature") == "canonical_storage_used_pct" and current is not None and current >= 80:
            recs.append("Increase storage capacity or clean old objects")
        if row.get("canonical_burst_balance_pct") is not None and not pd.isna(row.get("canonical_burst_balance_pct")) and row.get("canonical_burst_balance_pct") < 20:
            recs.append("Resize volume or move to a provisioned performance tier")
        if positive(row.get("backup_health_event")):
            recs.append("Investigate backup health event")
        if trend == "increasing" and reliability in ("high", "moderate"):
            f7 = forecast.get("forecast_7d")
            if f7 is not None and f7 >= 80:
                lo = forecast.get("forecast_7d_lower", "?")
                hi = forecast.get("forecast_7d_upper", "?")
                recs.append(f"Storage trending up — projected {f7:.0f}% in 7d (range: {lo}–{hi})")
    elif category == "Container":
        if row.get("canonical_node_cpu_pct") is not None and not pd.isna(row.get("canonical_node_cpu_pct")) and row["canonical_node_cpu_pct"] >= 85:
            recs.append("Node CPU pressure high — consider adding nodes or reviewing resource requests/limits")
        if row.get("canonical_node_mem_pct") is not None and not pd.isna(row.get("canonical_node_mem_pct")) and row["canonical_node_mem_pct"] >= 85:
            recs.append("Node memory pressure high — check for memory leaks or OOM-prone workloads")
        if row.get("canonical_unschedulable_pods") is not None and not pd.isna(row.get("canonical_unschedulable_pods")) and row["canonical_unschedulable_pods"] > 0:
            recs.append("Unschedulable pods detected — cluster may need scaling or resource requests tuned")
        if row.get("canonical_api_inflight_requests") is not None and not pd.isna(row.get("canonical_api_inflight_requests")) and row["canonical_api_inflight_requests"] > 400:
            recs.append("API server inflight requests high — risk of throttling; check controller churn")
    elif category == "Databases":
        if row.get("canonical_db_cpu_pct") is not None and not pd.isna(row.get("canonical_db_cpu_pct")) and row["canonical_db_cpu_pct"] >= 80:
            recs.append("DB CPU high — review long-running queries or scale compute tier")
        if row.get("canonical_db_storage_pct") is not None and not pd.isna(row.get("canonical_db_storage_pct")) and row["canonical_db_storage_pct"] >= 85:
            recs.append("DB storage approaching capacity — expand disk or purge old data")
        if row.get("canonical_db_connections") is not None and not pd.isna(row.get("canonical_db_connections")) and row["canonical_db_connections"] >= 90:
            recs.append("High connection count — consider connection pooling (PgBouncer/RDS Proxy)")
        if trend == "increasing" and reliability in ("high", "moderate") and forecast.get("forecast_24h") is not None and forecast["forecast_24h"] > 85:
            recs.append(f"DB CPU trending up — forecast {forecast['forecast_24h']:.0f}% in 24h")
    elif category == "Governance":
        if row.get("canonical_runs_failed") is not None and not pd.isna(row.get("canonical_runs_failed")) and row["canonical_runs_failed"] > 0:
            recs.append("Pipeline/activity runs failing — check ADF/SSM run logs for root cause")
        if row.get("canonical_run_failure_pct") is not None and not pd.isna(row.get("canonical_run_failure_pct")) and row["canonical_run_failure_pct"] >= 50:
            recs.append("Run failure rate ≥50% — investigate data source availability or pipeline config")
        if row.get("canonical_throttled_requests") is not None and not pd.isna(row.get("canonical_throttled_requests")) and row["canonical_throttled_requests"] > 0:
            recs.append("Throttled requests detected — review API quotas or add exponential backoff")
        if row.get("canonical_status_check_failed") is not None and not pd.isna(row.get("canonical_status_check_failed")) and row["canonical_status_check_failed"] > 0:
            recs.append("EC2 status check failed on governed instance — review Auto Scaling health checks")
    elif category == "Security":
        if row.get("canonical_saturation") is not None and not pd.isna(row.get("canonical_saturation")) and row["canonical_saturation"] >= 75:
            recs.append("Key Vault saturation high — review throttle limits or distribute across vaults")
        if row.get("canonical_availability_pct") is not None and not pd.isna(row.get("canonical_availability_pct")) and row["canonical_availability_pct"] < 99:
            recs.append("Key Vault availability degraded — check Azure service health and retry logic")
        if row.get("canonical_api_latency_ms") is not None and not pd.isna(row.get("canonical_api_latency_ms")) and row["canonical_api_latency_ms"] > 500:
            recs.append("Key Vault API latency elevated — investigate network path or request volume")
    if anomaly_score >= 0.6:
        recs.append("Investigate recent behaviour change — anomaly signals elevated vs this resource's own baseline")
    return recs


def summary_for(category, row, risk_score):
    parts = []
    if category == "Compute":
        if row.get("canonical_cpu_pct") is not None and not pd.isna(row.get("canonical_cpu_pct")):
            parts.append(f"CPU {row.get('canonical_cpu_pct'):.1f}%")
        if row.get("canonical_mem_used_pct") is not None and not pd.isna(row.get("canonical_mem_used_pct")):
            parts.append(f"memory {row.get('canonical_mem_used_pct'):.1f}%")
        if positive(row.get("status_check_failed")):
            parts.append("status check failed")
    elif category == "Network":
        if positive(row.get("canonical_ddos_signal")):
            parts.append("DDoS signal present")
        if row.get("canonical_health_pct") is not None and not pd.isna(row.get("canonical_health_pct")):
            parts.append(f"health {row.get('canonical_health_pct'):.1f}%")
    elif category == "Storage":
        if row.get("canonical_storage_used_pct") is not None and not pd.isna(row.get("canonical_storage_used_pct")):
            parts.append(f"storage {row.get('canonical_storage_used_pct'):.1f}%")
        if row.get("canonical_burst_balance_pct") is not None and not pd.isna(row.get("canonical_burst_balance_pct")):
            parts.append(f"burst {row.get('canonical_burst_balance_pct'):.1f}%")
    elif category == "Container":
        if row.get("canonical_node_cpu_pct") is not None and not pd.isna(row.get("canonical_node_cpu_pct")):
            parts.append(f"node CPU {row['canonical_node_cpu_pct']:.1f}%")
        if row.get("canonical_unschedulable_pods") is not None and not pd.isna(row.get("canonical_unschedulable_pods")) and row["canonical_unschedulable_pods"] > 0:
            parts.append(f"{row['canonical_unschedulable_pods']:.0f} unschedulable pods")
    elif category == "Databases":
        if row.get("canonical_db_cpu_pct") is not None and not pd.isna(row.get("canonical_db_cpu_pct")):
            parts.append(f"DB CPU {row['canonical_db_cpu_pct']:.1f}%")
        if row.get("canonical_db_storage_pct") is not None and not pd.isna(row.get("canonical_db_storage_pct")):
            parts.append(f"storage {row['canonical_db_storage_pct']:.1f}%")
    elif category == "Governance":
        if row.get("canonical_runs_failed") is not None and not pd.isna(row.get("canonical_runs_failed")):
            parts.append(f"{row['canonical_runs_failed']:.0f} failed runs")
        if row.get("canonical_run_failure_pct") is not None and not pd.isna(row.get("canonical_run_failure_pct")):
            parts.append(f"failure rate {row['canonical_run_failure_pct']:.1f}%")
    elif category == "Security":
        if row.get("canonical_availability_pct") is not None and not pd.isna(row.get("canonical_availability_pct")):
            parts.append(f"availability {row['canonical_availability_pct']:.1f}%")
        if row.get("canonical_saturation") is not None and not pd.isna(row.get("canonical_saturation")):
            parts.append(f"saturation {row['canonical_saturation']:.1f}")
    if not parts:
        parts.append(f"{category} risk score {risk_score:.0f}")
    return " + ".join(parts)


def prediction_type_for(category, payload):
    if category == "Compute":
        return "incident_probability"
    if category == "Network" and payload.get("ddos_probability", 0) >= 0.5:
        return "anomaly"
    if category == "Storage":
        return "utilisation_forecast"
    if category == "Container":
        return "cluster_health"
    if category == "Databases":
        return "utilisation_forecast"
    if category == "Governance":
        return "pipeline_health"
    if category == "Security":
        return "availability_risk"
    return "risk_score"


def generate_predictions(
    df,
    models,
    target_categories=TARGET_CATEGORIES,
    stale_resources=None,
    history_map=None,
    min_history_hours=24.0,
    pipeline_version="6.0.0",
):
    """Generate predictions. Builds pooled per-resource-type profiles and
    anomaly models once, reuses them for every resource of that type.

    New in v6:
    - stale_resources: dict {(resource_id, category): age_minutes} — predictions
      for stale resources are generated with suppressed alerts and a staleness note.
    - history_map: dict {(resource_id, category): hours} — resources below
      min_history_hours get a low-confidence prediction with a data_note.
    - pipeline_version: tagged on every prediction document for model versioning.

    v7: nothing here is trained on a single resource_id's own history. The
    diurnal profile (profiler.build_all_profiles) and the anomaly
    IsolationForest (anomaly.train_type_isolation_forests) are both pooled
    across every resource sharing a canonical_resource_type, then looked up
    per resource by that type — a resource's own `history` below is only
    ever used to know its current/recent values and to score against its
    type's shared baseline, never to fit a model of its own.
    """
    from .config import PIPELINE_VERSION as _PIPELINE_VERSION
    _pv = pipeline_version or _PIPELINE_VERSION

    stale_resources = stale_resources or {}
    history_map = history_map or {}

    # Build pooled per-(canonical_resource_type, category) profiles upfront
    # (one pass through df). No profile is ever built from a single
    # resource's own history alone.
    profiles = build_all_profiles(df)

    # Build type-specific slope prior caches for each category (one per category, reused for all resources)
    type_priors_by_category = {}
    for cat in target_categories:
        if canonical_features(cat):
            type_priors_by_category[cat] = build_type_specific_slope_priors(df, cat)

    # Build type-pooled anomaly IsolationForest models for each category
    # (one per canonical_resource_type within the category, reused for
    # every resource of that type — never fit per resource_id).
    type_if_by_category = {}
    for cat in target_categories:
        if canonical_features(cat):
            type_if_by_category[cat] = train_type_isolation_forests(df, cat, profiles)

    predictions = []
    for _, row in latest_rows(df).iterrows():
        category = row["category"]
        if category not in target_categories:
            continue
        if not canonical_features(category):
            continue

        history = df[(df["resource_id"] == row["resource_id"]) & (df["category"] == category)]
        signal_row = consolidated_row(category, history, row)
        artifact = models.get(category)
        type_priors_cache = type_priors_by_category.get(category)
        model_version = (
            artifact["model_version"]
            if artifact and "model_version" in artifact
            else MODEL_VERSION.get(category, f"unsupervised-v{_pv}")
        )

        # --- Freshness gate ---
        resource_key = (str(row.get("resource_id", "")), category)
        is_stale = resource_key in stale_resources
        stale_age = stale_resources.get(resource_key)

        # --- Minimum history guard ---
        history_hours = history_map.get(resource_key, None)
        insufficient_history = (history_hours is not None and history_hours < min_history_hours)

        completeness = data_completeness_score(category, signal_row)

        # This resource is never profiled/trained on its own — it looks up
        # its canonical_resource_type's pooled profile and pooled anomaly
        # model (built once per type above, shared by every resource of
        # that type).
        resource_canonical_type_key = str(row.get("canonical_resource_type") or "")
        profile = profiles.get((resource_canonical_type_key, category))
        type_if_artifact = type_if_by_category.get(category, {}).get(resource_canonical_type_key)

        quality_note = None
        if is_stale:
            quality_note = f"stale_resource_{stale_age:.0f}min"
        elif insufficient_history:
            quality_note = f"insufficient_history_{history_hours:.1f}h"

        if completeness == 0.0:
            risk_score = 0.0
            sev = "INFO"
            anomaly_score = 0.0
            is_anomalous = False
            alert_trigger = False
            anomaly_detail = {"error": "no_telemetry"}
            recs_override = ["No telemetry available — verify monitoring agent/collector is reporting"]
        elif is_stale:
            # Resource has gone silent — suppress alert, flag for investigation
            risk_score = 0.0
            sev = "INFO"
            anomaly_score = 0.0
            is_anomalous = False
            alert_trigger = False
            anomaly_detail = {"error": "stale_resource", "age_minutes": stale_age}
            recs_override = [f"Resource has not reported metrics for {stale_age:.0f}min — verify agent/collector"]
        elif insufficient_history:
            # Too little data for reliable prediction — run scoring but flag low confidence
            risk_score, anomaly_detail = score_resource(history, category, profile, type_if_artifact)
            anomaly_score = round(max(0.0, min(1.0, risk_score / 100.0)), 3)
            is_anomalous = False  # Don't alert on thin data
            sev = "INFO"
            alert_trigger = False
            anomaly_detail["warning"] = f"insufficient_history_{history_hours:.1f}h"
            recs_override = [f"Only {history_hours:.1f}h of history — predictions unreliable until 24h threshold is met"]
        else:
            risk_score, anomaly_detail = score_resource(history, category, profile, type_if_artifact)
            anomaly_score = round(max(0.0, min(1.0, risk_score / 100.0)), 3)
            any_dominant = anomaly_detail.get("any_signal_dominant", False)
            is_anomalous = anomaly_score >= 0.6 or any_dominant
            sev = severity(risk_score)
            alert_trigger = risk_score >= 40 or any_dominant
            recs_override = None

        payload  = category_payload(category, history, signal_row, risk_score)

        profile_info = {
            "profile_confidence":       profile.get("profile_confidence", 0.0) if profile else 0.0,
            "profile_age_days":         profile.get("n_days", 0) if profile else 0,
            "canonical_resource_type":  resource_canonical_type_key or None,
            "n_resources_pooled":       profile.get("n_resources_pooled") if profile else None,
            "regime_changed":     any(
                m.get("regime_changed", False)
                for m in (profile.get("metrics", {}).values() if profile else [])
            ),
        }

        # --- Explode into one prediction per individual metric this resource
        # has reported for this category, instead of a single prediction
        # bundling every metric together. A resourceId that reports IOPS,
        # burst-balance and throughput as separate log entries now yields
        # one prediction PER metric rather than one prediction total (the
        # earlier bug) or one prediction per joint set-of-metrics-present.
        metrics_present = resource_metrics_present(category, history)
        if not metrics_present:
            # No canonical metric has ever had data for this resource —
            # still emit exactly one no-telemetry prediction so the resource
            # isn't silently dropped from output entirely.
            metrics_present = [None]

        for feature in metrics_present:
            forecast = behavioral_forecast(
                category, history, row, artifact=artifact,
                type_priors_cache=type_priors_cache, forced_feature=feature,
            )
            failures = failure_predictions(category, forecast, signal_row)
            recs     = recs_override if recs_override is not None else recommendations(
                category, signal_row, forecast, failures, anomaly_score
            )

            # Regime-change aware recommendation: a sustained baseline shift means
            # the resource's "normal" has permanently changed (e.g. post-deploy
            # traffic doubled). Flag this explicitly instead of silently treating
            # the new normal as an ongoing anomaly indefinitely.
            if profile_info["regime_changed"] and recs_override is None:
                recs = recs + [
                    "Sustained shift from this resource's historical baseline detected "
                    "(regime change) — if this reflects an intentional change (deploy, "
                    "scale-up, traffic growth), the baseline will re-settle automatically "
                    "over the next profiling window; if unexpected, investigate the cause."
                ]

            prediction = {
                "prediction_id":           f"pred_{datetime.now(timezone.utc).strftime('%Y%m%d')}_{uuid4().hex[:6]}",
                "prediction_type":         prediction_type_for(category, payload),
                "model_version":           model_version,
                "pipeline_version":        _pv,
                "generated_at":            now_iso(),
                "data_quality_note":       quality_note,
                "resource_id":             clean(row.get("resource_id")),
                "resource_name":           clean(row.get("resource_name")),
                "account_id":              clean(row.get("account_id")),
                "project":                 clean(row.get("project")),
                "category":                clean(category),
                "service_family":          clean(category),
                "provider":                clean(row.get("service_name")),
                "component":               clean(row.get("component")),
                "canonical_resource_type": clean(row.get("canonical_resource_type")),
                # The specific resourceId + metric combination this
                # prediction was built from. A resource that reports several
                # metrics for this category gets one prediction document per
                # metric, each carrying its own forecast for that metric.
                "metric_combination":      clean(feature),
                "location":                clean(row.get("location")),
                "tags":                    clean(row.get("tags")),
                "summary_details":         clean(row.get("summary_details")),
                "prediction_timestamp":    now_iso(),
                "data_completeness_score": completeness,
                "behavioral_forecast":     forecast,
                "failure_predictions":     failures,
                "anomaly_score":           anomaly_score,
                "is_anomalous":            is_anomalous,
                "recommendations":         recs,
                "payload":                 payload,
                "anomaly_detail":          anomaly_detail,
                "profile_info":            profile_info,
                "alert": {
                    "trigger":    alert_trigger,
                    "severity":   sev,
                    "channel":    channel_for(sev),
                    "group_key":  f"{row.get('account_id')}::{category}",
                    "suppressed": sev == "INFO",
                    "ttl_minutes": 60,
                },
                "dashboard": {
                    "display_score": int(round(risk_score)),
                    "display_label": "Risk score",
                    "trend":         forecast.get("trend_direction", "STABLE").upper(),
                    "badge_color":   badge(risk_score),
                    "summary_text":  summary_for(category, signal_row, risk_score),
                },
                "feedback": {
                    "outcome_recorded": False,
                    "outcome_label":    None,
                    "outcome_at":       None,
                },
            }
            predictions.append(prediction)
    return predictions
