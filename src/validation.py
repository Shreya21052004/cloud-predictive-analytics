from .metric_registry import canonical_features

import numpy as np
import pandas as pd


PCT_FEATURES = {
    "canonical_cpu_pct",
    "canonical_cpu_idle_pct",
    "canonical_iowait_pct",
    "canonical_mem_used_pct",
    "canonical_swap_used_pct",
    "canonical_disk_used_pct",
    "canonical_health_pct",
    "canonical_subnet_util_pct",
    "canonical_storage_used_pct",
    "canonical_burst_balance_pct",
    "availability_pct",
}


def validate_normalized(df):
    issues = []
    if df.empty:
        return ["No normalized rows were produced. Check category, metric, and metric_value fields."]

    for category in sorted(df["category"].dropna().unique()):
        cat_df = df[df["category"] == category]
        for feature in canonical_features(category):
            if feature not in cat_df.columns:
                continue
            if feature not in PCT_FEATURES:
                continue
            for provider, provider_df in cat_df.groupby("service_name", dropna=False):
                series = provider_df[feature].dropna()
                if series.empty:
                    continue
                if series.max() > 150 or series.min() < -5:
                    issues.append(
                        f"{category}.{feature} ({provider}) has suspicious percent range "
                        f"min={series.min():.2f}, max={series.max():.2f}"
                    )
    return issues


def compute_forecast_accuracy(predictions_df, actuals_df, horizon_hours=24):
    """Compute forecast accuracy metrics by comparing predictions to actuals.
    
    Args:
        predictions_df: DataFrame with prediction documents
                       Must have: resource_id, canonical_feature, forecast_24h, 
                                  forecast_24h_lower, forecast_24h_upper, timestamp
        actuals_df: DataFrame with actual future values
                   Must have: resource_id, canonical_feature, value, timestamp
        horizon_hours: Which forecast horizon to evaluate (1, 6, 24, 168, 720)
    
    Returns: dict with accuracy metrics
        {
          "mae": Mean Absolute Error,
          "rmse": Root Mean Squared Error,
          "mape": Mean Absolute Percentage Error,
          "coverage": % of actuals within confidence interval,
          "directional_accuracy": % of trend predictions correct,
          "within_ci_pct": % of forecasts where actual fell in [lower, upper],
          "mean_forecast": Average forecast value,
          "mean_actual": Average actual value,
          "count": Number of valid forecast-actual pairs,
          "by_reliability": Accuracy stratified by forecast_reliability
        }
    """
    if predictions_df.empty or actuals_df.empty:
        return {}
    
    # Select the appropriate forecast horizon
    forecast_key = f"forecast_{horizon_hours}h"
    forecast_lower_key = f"forecast_{horizon_hours}h_lower"
    forecast_upper_key = f"forecast_{horizon_hours}h_upper"
    
    # Merge predictions with actuals on (resource_id, feature)
    pred_cols = ["resource_id", "canonical_feature", forecast_key, forecast_lower_key, 
                 forecast_upper_key, "trend_direction", "forecast_reliability", "timestamp"]
    actual_cols = ["resource_id", "canonical_feature", "value", "timestamp"]
    
    # Ensure all required columns exist
    pred_cols = [c for c in pred_cols if c in predictions_df.columns]
    actual_cols = [c for c in actual_cols if c in actuals_df.columns]
    
    if not all(c in predictions_df.columns for c in ["resource_id", "canonical_feature", forecast_key]):
        return {"error": "Missing required prediction columns"}
    if not all(c in actuals_df.columns for c in ["resource_id", "canonical_feature", "value"]):
        return {"error": "Missing required actual value columns"}
    
    pred = predictions_df[pred_cols].copy()
    actual = actuals_df[actual_cols].copy()
    
    # Align actuals to be ~horizon_hours after prediction timestamp
    # (allow ±6 hour tolerance)
    pred['pred_time'] = pd.to_datetime(pred['timestamp'], errors='coerce')
    actual['actual_time'] = pd.to_datetime(actual['timestamp'], errors='coerce')
    
    # Merge on resource and feature
    merged = pd.merge(
        pred, actual,
        on=['resource_id', 'canonical_feature'],
        how='inner'
    )
    
    if merged.empty:
        return {"error": "No matching predictions and actuals found", "count": 0}
    
    # Filter for actuals that occurred ~horizon_hours after prediction
    if 'pred_time' in merged.columns and 'actual_time' in merged.columns:
        time_diff_hours = (merged['actual_time'] - merged['pred_time']).dt.total_seconds() / 3600
        # Accept if within ±(horizon/2) of expected time
        tolerance = max(6, horizon_hours / 2)
        merged = merged[
            (time_diff_hours >= horizon_hours - tolerance) & 
            (time_diff_hours <= horizon_hours + tolerance)
        ]
    
    if merged.empty:
        return {"error": f"No actuals found within {horizon_hours}h ±6h window", "count": 0}
    
    forecast_vals = pd.to_numeric(merged[forecast_key], errors='coerce')
    actual_vals = pd.to_numeric(merged['value'], errors='coerce')
    
    # Remove NaN values
    valid_mask = forecast_vals.notna() & actual_vals.notna()
    forecast_vals = forecast_vals[valid_mask]
    actual_vals = actual_vals[valid_mask]
    merged = merged[valid_mask]
    
    if len(forecast_vals) == 0:
        return {"error": "No valid numeric forecast-actual pairs", "count": 0}
    
    # ── Accuracy Metrics ──
    errors = forecast_vals.values - actual_vals.values
    abs_errors = np.abs(errors)
    
    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    
    # MAPE (avoid division by zero)
    nonzero_actual = actual_vals[actual_vals != 0]
    if len(nonzero_actual) > 0:
        mape = float(np.mean(np.abs((forecast_vals[actual_vals != 0] - nonzero_actual) / nonzero_actual)) * 100)
    else:
        mape = None
    
    # Confidence Interval Coverage
    ci_lower = pd.to_numeric(merged[forecast_lower_key], errors='coerce')
    ci_upper = pd.to_numeric(merged[forecast_upper_key], errors='coerce')
    
    ci_valid = ci_lower.notna() & ci_upper.notna()
    if ci_valid.any():
        within_ci = (
            (actual_vals[ci_valid].values >= ci_lower[ci_valid].values) &
            (actual_vals[ci_valid].values <= ci_upper[ci_valid].values)
        )
        coverage_pct = float(within_ci.mean() * 100)
    else:
        coverage_pct = None
    
    # Directional Accuracy
    actual_direction = np.where(
        actual_vals.values > forecast_vals.values, "increasing",
        np.where(actual_vals.values < forecast_vals.values, "decreasing", "stable")
    )
    predicted_direction = merged['trend_direction'].values
    
    if len(predicted_direction) > 0:
        # Simplistic: consider "stable" correct if actual was within ±2% of forecast
        directional_correct = (predicted_direction == actual_direction)
        directional_accuracy = float(directional_correct.mean() * 100)
    else:
        directional_accuracy = None
    
    # Stratify by forecast reliability
    reliability_groups = {}
    for reliability in ['high', 'medium-high', 'medium', 'low']:
        if 'forecast_reliability' in merged.columns:
            group = merged[merged['forecast_reliability'] == reliability]
            if len(group) > 0:
                group_forecast = pd.to_numeric(group[forecast_key], errors='coerce')
                group_actual = pd.to_numeric(group['value'], errors='coerce')
                valid = group_forecast.notna() & group_actual.notna()
                if valid.any():
                    group_errors = group_forecast[valid].values - group_actual[valid].values
                    reliability_groups[reliability] = {
                        "count": int(valid.sum()),
                        "mae": float(np.mean(np.abs(group_errors))),
                        "rmse": float(np.sqrt(np.mean(group_errors ** 2))),
                        "bias": float(np.mean(group_errors)),  # Positive = over-forecasting
                    }
    
    return {
        "horizon_hours": horizon_hours,
        "count": len(forecast_vals),
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "mape": round(mape, 2) if mape is not None else None,
        "coverage_pct": round(coverage_pct, 1) if coverage_pct is not None else None,
        "directional_accuracy_pct": round(directional_accuracy, 1) if directional_accuracy is not None else None,
        "mean_forecast": round(float(forecast_vals.mean()), 3),
        "mean_actual": round(float(actual_vals.mean()), 3),
        "forecast_bias": round(float(forecast_vals.mean() - actual_vals.mean()), 3),
        "by_reliability": reliability_groups,
    }
