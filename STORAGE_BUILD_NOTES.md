# Storage-only build — notes

This package is a Storage-category-scoped variant of the predictive analytics
pipeline, pointed at the `22predictive_analytics_dataset_sorted` collection.

## What's different from the original package

1. **`src/config.py`**
   - `source_collection` defaults to `22predictive_analytics_dataset_sorted`.
   - `output_collection` defaults to `22prediction_logs_storage`.
   - `limit` defaults to `0` (load all documents — the sorted collection is
     expected to hold the full dataset).
   - New `TARGET_CATEGORIES = ("Storage",)` constant. The normalizer's
     `CATEGORY_ALIASES` still maps `"networking" -> "Network"` etc., but since
     only `"Storage"` is requested, Network/Networking/Compute documents are
     never loaded or trained on.

2. **`src/mongo_io.py`**
   - The find cursor now sorts by `from_date`, then `created_at`
     (`.sort([("from_date", 1), ("created_at", 1)])`) before batching, so rows
     arrive in chronological order even before the normalizer's own
     `sort_values(["category", "resource_id", "timestamp"])`.

3. **`src/models.py`**
   - Fixed `feature_columns` / `build_preprocessor`: canonical feature columns
     that are entirely `None` for the loaded data come back as pandas
     `object` dtype, which previously caused them to be misclassified as
     *categorical* and routed into `OneHotEncoder` — and pushed
     `service_name`/`component`/`location` into the *numeric* branch instead,
     crashing `SimpleImputer` with `"could not convert string to float: 'GCP'"`.
     Columns are now classified explicitly: all `canonical_*` /
     `*_available` columns are numeric (coerced via `pd.to_numeric`), and
     `service_name` / `component` / `location` are always categorical.
   - `train_models` / `load_models` now only handle the `"Storage"` category.

4. **`src/metric_registry.py`** — `canonical_storage_used_pct` previously mixed
   true percentages (AWS `PercentIOLimit`, OCI `FileSystemUsage`) with raw
   byte/byte-second counters (AWS `BucketSizeBytes`, Azure `UsedCapacity`, GCP
   `storage/total_byte_seconds`). Mixing units in one column corrupts both the
   model and the `validate_normalized` percent-range check. Split into:
   - `canonical_storage_used_pct`: AWS `PercentIOLimit`, OCI `FileSystemUsage`
     only (genuine 0-100 / 0-1 percentages).
   - `canonical_storage_capacity_raw` (new): AWS `BucketSizeBytes`, Azure
     `UsedCapacity`, GCP `storage/total_byte_seconds`. These need a per-resource
     quota/size lookup to become a true percent — same class of problem as the
     Azure memory-baseline gap. Until that lookup table exists, this stays a
     raw growth-trend signal; `behavioral_forecast`'s fallback logic
     (`choose_forecast_feature`) will use it for Azure/GCP resources that have
     no `canonical_storage_used_pct` signal.

5. **`src/validation.py`** — percent-range checks (`PCT_FEATURES`) are now
   evaluated **per provider** (`groupby("service_name")`) instead of across
   the whole column. A unit mismatch in one provider's feed no longer masks
   (or falsely flags) the same canonical feature for other providers.

6. **`src/prediction.py`**
   - `generate_predictions(df, models, target_categories=("Storage",))` now
     filters to the requested categories explicitly (defensive; previously
     relied on whatever was in `df`).
   - **No-data guard**: if `data_completeness_score == 0.0` (every canonical
     Storage signal is unavailable for a resource at its latest timestamp),
     the prediction is forced to `risk_score=0`, `severity="INFO"`,
     `is_anomalous=False`, `alert.trigger=False`, with a recommendation to
     check the monitoring agent/collector. Previously an all-missing feature
     row could be scored as ~90%+ "anomalous" by IsolationForest/RandomForest
     purely because "no data" is itself out-of-distribution — producing a
     false CRITICAL PagerDuty page for a resource that simply stopped
     reporting telemetry.

## Stale model artifact

`models/storage_model.joblib` from the original package was trained under the
old registry (mixed-unit `canonical_storage_used_pct`, broken
categorical/numeric split) and has been **removed**. Run `train` or `run` to
regenerate it under the corrected registry/preprocessor — the feature set and
column ordering have changed, so the old artifact is not compatible.

## Running it

```bash
pip install -r requirements.txt

# Inspect raw -> canonical mapping coverage for Storage docs
python -m src.pipeline inspect --mongo-uri "<uri>" --db "<db>"

# Validate normalized rows (per-provider percent range checks)
python -m src.pipeline validate --mongo-uri "<uri>" --db "<db>"

# Train the Storage model on 22predictive_analytics_dataset_sorted
python -m src.pipeline train --mongo-uri "<uri>" --db "<db>"

# Train + predict, writing to 22prediction_logs_storage
python -m src.pipeline run --mongo-uri "<uri>" --db "<db>"

# Predict only (loads existing models/storage_model.joblib if present)
python -m src.pipeline predict --mongo-uri "<uri>" --db "<db>" --dry-run
```

All commands default to:
- `--source 22predictive_analytics_dataset_sorted`
- `--out 22prediction_logs_storage`
- `--limit 0` (all documents)

Override with `--source` / `--out` / `--limit` / `--mongo-uri` / `--db` as needed.

## Additional fix: case-insensitive metric_value keys

`src/normalizer.py` — fixed a critical extraction bug: GCP (and possibly other)
documents store `metric_value` array items with capitalized CloudWatch-style
keys (`Timestamp`, `Average`, `Maximum`, `Minimum`, `SampleCount`, `Sum`)
instead of the lowercase keys (`timeStamp`, `average`, `maximum`, `minimum`,
`count`) the original code looked for. Every lookup silently missed, so these
rows got `value=None` and `*_available=False` for every canonical feature —
producing `data_completeness_score: 0.0` predictions for resources that
actually had full telemetry (e.g. `migratebucket222` /
`storage.googleapis.com/storage/total_byte_seconds`).

Added `get_ci()` — a case-insensitive key lookup — used in
`extract_metric_value`, `extract_timestamp`, and `metric_points`. The value-key
priority order is `average -> maximum -> total -> sum -> minimum -> count ->
samplecount` (checked case-insensitively); `average`/`Average` is checked
first so a `Sum` field (which for hourly cumulative metrics equals
`Average * SampleCount`, not a per-sample reading) is never mistaken for the
metric value when `average` is present.

## Additional fix: split EBS burst balance % from EC2 CPU credit balance count

`src/metric_registry.py` — `canonical_burst_balance_pct` previously mapped both
AWS `BurstBalance` (EBS gp2 volume burst bucket, true 0-100%) and
`BurstCreditBalance` (EC2 T-class instance CPU credit count, unbounded -
observed values in the trillions) through `pct()` into the same column. Since
`pct()` only rescales values already in `[0,1]`, the large `BurstCreditBalance`
counts passed through unchanged and corrupted the "percent" column (validation
flagged `max=2308974418330.00`).

Split into:
- `canonical_burst_balance_pct`: AWS `BurstBalance` only (true 0-100% EBS burst
  bucket). Used unchanged by `burst_exhaustion_risk`,
  `BURST_CREDIT_EXHAUSTION` failure prediction, and the "resize volume"
  recommendation.
- `canonical_credit_balance_raw` (new): AWS `BurstCreditBalance` (raw,
  unbounded count). Currently carried as a model feature with its
  `_available` flag; not yet wired into payload/recommendations logic.

Re-train (`train` or `run`) after this fix — the Storage model's feature set
changed again.
