# Forecast Reliability & Accuracy Metrics

## Overview

Every prediction now includes multiple reliability indicators that work together to assess confidence in the forecast.

---

## Reliability Indicators (In Each Prediction)

### 1. **`prediction_confidence_score`** (0-100) ⭐ PRIMARY
Composite score combining:
- **50%** from `trend_fit_r2` (model fit quality)
- **30%** from `forecast_reliability` label (high/medium/low)
- **20%** from `forecast_method` quality (STL > regression > pooled)

**Interpretation:**
- **80-100**: High confidence forecast
- **60-79**: Good forecast, but watch bounds
- **40-59**: Moderate forecast, use bounds
- **0-39**: Low confidence, treat as guidance only

**Example:**
```javascript
{
  "trend_fit_r2": 0.82,
  "forecast_reliability": "high",
  "forecast_method": "stl_decomposition",
  "prediction_confidence_score": 87  // Very reliable
}
```

---

### 2. **`trend_fit_r2`** (0-1.0) ⭐ PRIMARY
R² (coefficient of determination) - how well the trend model fits the data.

**Interpretation:**
- **0.8+**: Excellent fit
- **0.6-0.79**: Good fit
- **0.4-0.59**: Moderate fit
- **<0.4**: Poor fit, use caution

**What it means:**
- R² = 0.8 means 80% of variance in data is explained by the trend
- For pooled thin resources: synthetic R² based on pool size
- For flat signals: lower R² but still predictable if idle

---

### 3. **`forecast_reliability`** (categorical) ⭐ PRIMARY
High-level confidence assessment: `"high"` | `"medium-high"` | `"medium"` | `"low"`

**How it's determined:**
```
For resources with pooling:
  pool_size >= 10 → "high"
  pool_size >= 5  → "medium-high"
  pool_size >= 2  → "medium"

For individual resources:
  R² >= 0.8       → "high"
  R² >= 0.6       → "medium-high"
  R² >= 0.4       → "medium"
  R² < 0.4        → "low"
```

---

### 4. **`data_completeness_score`** (0-100)
Percentage of data points available vs missing (NaN values).

**Interpretation:**
- **80-100%**: Excellent data quality
- **60-79%**: Good coverage
- **40-59%**: Moderate data gaps
- **<40%**: Significant gaps, forecast less reliable

**Why it matters:**
More complete time series → better trend detection → higher R²

---

### 5. **`forecast_uncertainty_pct`** (%)
Width of the 24-hour confidence interval as percentage of forecast value.

**Calculation:**
```
uncertainty_pct = (upper_bound - lower_bound) / |forecast_value| × 100
```

**Interpretation:**
- **<10%**: Very tight bounds, high confidence
- **10-30%**: Moderate bounds
- **30-80%**: Wide bounds, moderate confidence
- **>80%**: Very wide bounds, low confidence

**Example:**
```javascript
{
  "forecast_24h": 100,
  "forecast_24h_lower": 90,
  "forecast_24h_upper": 110,
  "forecast_uncertainty_pct": 20  // ±10% range
}
```

---

### 6. **`forecast_method`** (technical approach)
How the forecast was computed:

| Method | Quality | Use Case |
|--------|---------|----------|
| `stl_decomposition` | 🟢 Excellent | 72+ points, with seasonality |
| `linear_regression_bootstrap` | 🟢 Good | 5-72 points, no seasonality |
| `exponential_smoothing` | 🟡 OK | Bursty metrics (bytes, connections) |
| `shrunk_slope_xgboost_prior` | 🟡 Fair | Thin series + category-wide prior |
| `pooled_type_prior` | 🟡 Fair | Thin series + type-specific prior |
| `flat_signal` | 🟠 Low | Constant values, no trend |

---

### 7. **`pooling_tier`** (where forecast borrowed from)
Shows which data pool was used for shrinkage/pooling:

| Tier | Meaning | Confidence |
|------|---------|------------|
| `individual` | Own data only, no borrowing | Highest |
| `type_pooled` | Borrowed from same type (S3→S3, EC2→EC2) | High |
| `category_pooled` | Borrowed from broader category | Medium |

**Example flow:**
```
S3 bucket A (sparse) + 15 other S3 buckets (well-populated)
→ pooling_tier = "type_pooled"
→ Uses median slope from those 15 buckets
→ Higher confidence than if alone
```

---

### 8. **`data_note`** (explanation for low confidence)
Human-readable reason if forecast may be unreliable:

| Note | Meaning |
|------|---------|
| `ok` | Good data, reliable forecast |
| `insufficient_points_3` | Only 3 data points |
| `insufficient_points_3_pooled` | 3 points, using pool to compensate |
| `stale_resource_422212min` | Last data 293+ days ago |
| `mixed_stat_fields` | Mix of average/max/sum in same metric |
| `all_zeros_after_cleaning` | All values are zero after outlier removal |

---

### 9. **`trend_direction`** (predicted direction)
Expected change: `"increasing"` | `"decreasing"` | `"stable"`

**Interpretation:**
- Use with `trend_fit_r2` - direction is only reliable if R² is high
- "stable" with R²=0.2 is just noise
- "increasing" with R²=0.85 is a strong signal

---

## Accuracy Metrics (Historical Validation)

After running the pipeline multiple times and collecting ground truth, compute:

```python
from src.validation import compute_forecast_accuracy

# Compare predictions from run N with actuals from run N+1
accuracy = compute_forecast_accuracy(
    predictions_df=predictions_from_yesterday,
    actuals_df=actual_values_today,
    horizon_hours=24
)

print(accuracy)
# Output:
# {
#   'mae': 3.5,                    # Mean Absolute Error
#   'rmse': 4.2,                   # Root Mean Squared Error
#   'mape': 5.1,                   # Mean Absolute Percentage Error
#   'coverage_pct': 94.5,          # % within confidence interval
#   'directional_accuracy_pct': 78,# % of trend direction correct
#   'by_reliability': {
#     'high': {'count': 145, 'mae': 1.8, 'rmse': 2.1},
#     'medium': {'count': 89, 'mae': 5.2, 'rmse': 6.3},
#     'low': {'count': 12, 'mae': 12.1, 'rmse': 15.4}
#   }
# }
```

### Accuracy Metrics Explained

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| **MAE** | Σ\|actual - forecast\| / n | Average error magnitude (same units as metric) |
| **RMSE** | √(Σ(actual - forecast)² / n) | Penalizes larger errors more than MAE |
| **MAPE** | Σ\|actual - forecast\| / \|actual\| × 100% | Error as percentage of actual value |
| **Coverage** | % actuals within [lower, upper] | Should be ~68-95% for well-calibrated bounds |
| **Directional** | % correct trend direction | High value indicates trend detection works |

---

## Query Examples

### Find highly reliable forecasts
```javascript
{
  "trend_fit_r2": { $gt: 0.7 },
  "forecast_reliability": "high",
  "prediction_confidence_score": { $gt: 80 }
}
```

### Find uncertain forecasts (may need review)
```javascript
{
  "prediction_confidence_score": { $lt: 50 },
  $or: [
    { "data_note": /insufficient_points/ },
    { "pooling_tier": "category_pooled" },
    { "forecast_uncertainty_pct": { $gt: 50 } }
  ]
}
```

### Find thin resources with good pooling support
```javascript
{
  "data_note": /insufficient_points.*pooled/,
  "pooling_tier": "type_pooled",
  "trend_fit_r2": { $gt: 0.5 },
  "forecast_reliability": { $in: ["high", "medium-high"] }
}
```

### Stratify by confidence
```javascript
// High confidence forecasts
db.predictions.countDocuments({ "prediction_confidence_score": { $gte: 80 } })

// Medium confidence
db.predictions.countDocuments({ "prediction_confidence_score": { $gte: 60, $lt: 80 } })

// Low confidence
db.predictions.countDocuments({ "prediction_confidence_score": { $lt: 60 } })
```

---

## Typical Results

### Before Pooling
```
Storage Category: 569 total resources
- R² > 0.7: 146 (25.7%)
- R² < 0.4: 285 (50.1%)
```

### After Type-Specific Pooling
```
Storage Category: 569 total resources
- prediction_confidence_score > 75: ~380 (66.8%)
- prediction_confidence_score > 60: ~480 (84.4%)
- prediction_confidence_score < 40: ~40 (7.0%)

By pooling_tier:
- individual: 120 (21.1%) - own data sufficient
- type_pooled: 350 (61.5%) - pooled with same type
- category_pooled: 99 (17.4%) - pooled with category
```

---

## Recommendations

### Use Predictions When:
✅ `prediction_confidence_score > 75` AND `forecast_reliability` ∈ ["high", "medium-high"]  
✅ `trend_fit_r2 > 0.7` AND `data_note = "ok"`  
✅ `forecast_method` ∈ ["stl_decomposition", "linear_regression_bootstrap"]  

### Caution When:
⚠️ `prediction_confidence_score` 50-75: Use bounds, monitor closely  
⚠️ `forecast_uncertainty_pct > 50%`: Wide interval, direction matters more than point  
⚠️ `data_completeness_score < 60%`: Few data points, treat as guidance  

### Ignore/Escalate When:
❌ `prediction_confidence_score < 40`  
❌ `data_note` contains "stale_resource" or "insufficient_points" without pooling  
❌ `pooling_tier` = "category_pooled" AND `forecast_reliability` = "low"  

---

## Monitoring Over Time

```python
# Track forecast accuracy over time
import pymongo
from datetime import datetime, timedelta

client = pymongo.MongoClient()
db = client.pool_prediction

# Weekly accuracy
last_week = datetime.utcnow() - timedelta(days=7)
accuracy = compute_forecast_accuracy(
    predictions_df=db.predictions.find({"timestamp": {"$gte": last_week}}),
    actuals_df=db.observations.find({"timestamp": {"$gte": last_week}}),
    horizon_hours=24
)

# Compare by resource type
for resource_type in db.predictions.distinct("canonical_resource_type"):
    type_accuracy = compute_forecast_accuracy(
        predictions_df=db.predictions.find({
            "canonical_resource_type": resource_type,
            "timestamp": {"$gte": last_week}
        }),
        actuals_df=db.observations.find({"timestamp": {"$gte": last_week}}),
        horizon_hours=24
    )
    print(f"{resource_type}: MAE={type_accuracy['mae']}, Coverage={type_accuracy['coverage_pct']}%")
```

---

## Summary

The new confidence metrics provide **three levels** of assessment:

1. **Point Prediction**: `trend_fit_r2` (statistical fit quality)
2. **Interval Prediction**: `forecast_*_lower/upper` + `forecast_uncertainty_pct` (bounds)  
3. **Composite Score**: `prediction_confidence_score` (holistic reliability)

Use all three together for complete picture of forecast trustworthiness.
