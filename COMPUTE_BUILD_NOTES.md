# Compute extension — build notes

This package extends the Storage-only pipeline to run **Storage and Compute in
parallel**. Storage is untouched — every existing behaviour, model artifact,
metric mapping, output collection, and prediction schema is preserved exactly
as described in `STORAGE_BUILD_NOTES.md`.

---

## What changed to add Compute

### 1. `src/config.py`

- `TARGET_CATEGORIES` changed from `("Storage",)` to `("Storage", "Compute")`.
- New field `compute_output_collection` (default `"22prediction_logs_compute"`).
  Storage continues writing to `output_collection` (`"22prediction_logs_storage"`);
  Compute writes to a separate collection so the two datasets never intermingle.

### 2. `src/models.py`

- `train_models` and `load_models` now iterate `("Storage", "Compute")`.
  - Storage: writes/reads `models/storage_model.joblib` — **unchanged**.
  - Compute: writes/reads `models/compute_model.joblib` — new.
- The `weak_risk_label` function already contained full Compute scoring logic
  (CPU ≥ 80 → +35, mem ≥ 85 → +30, iowait ≥ 20 → +20, status check failed →
  +30, lambda throttles → +20, cpu credit balance < 20 → +15); it was just
  never reached because Compute documents were filtered out at load time.
- Model selection for Compute follows the same XGBoost-first / IsolationForest
  fallback path as Storage. XGBoost is preferred because Compute has more
  binary event signals (status_check_failed, lambda_throttles) that a gradient
  boosted classifier handles better than isolation-based anomaly detection.

### 3. `src/prediction.py`

- `generate_predictions` default `target_categories` updated to
  `("Storage", "Compute")`.
- `BURSTY_METRICS` extended with Compute-specific bursty signals:
  `canonical_disk_read_iops`, `canonical_disk_write_iops`,
  `lambda_throttles`, `lambda_invocations`, `lambda_errors`,
  `lambda_concurrent_executions`, `status_check_failed`.
  These use exponential smoothing (α=0.3) rather than linear regression —
  appropriate because Lambda invocation counts and status-check events are
  episodic, not capacity trends.
- `PRIMARY_FORECAST_FEATURE["Compute"]` (already in `metric_registry.py`) is
  `canonical_cpu_pct`, so the primary behavioral forecast for Compute resources
  is CPU utilisation.
- `category_payload`, `failure_predictions`, `recommendations`, and
  `summary_for` all had complete Compute branches already; no changes needed.

### 4. `src/pipeline.py`

- `predict_from_df` splits the returned predictions list by category and routes:
  - `Storage` predictions → `config.output_collection`
  - `Compute` predictions → `config.compute_output_collection`
  Each write is independent; a failure in one does not affect the other.
- `load_normalized` print message simplified (no longer mentions "Network alias").
- New `--compute-out` CLI flag (default `"22prediction_logs_compute"`).

### 5. `src/metric_registry.py`, `src/normalizer.py`, `src/validation.py`

No changes. The Compute metric registry (`REGISTRY["Compute"]`) was already
fully defined with 18 canonical features across AWS, Azure, GCP, and OCI.

---

## Compute model: design decisions

| Aspect | Choice | Rationale |
|---|---|---|
| Primary forecast metric | `canonical_cpu_pct` | Most universally available; directly actionable for scale-up decisions |
| Bursty metric handling | Exponential smoothing | Lambda counts, IOPS, and status checks are episodic, not monotone trends |
| Risk labelling | `weak_risk_label("Compute")` already in codebase | CPU ≥ 80 + mem ≥ 85 + iowait ≥ 20 + status_check + lambda_throttles + cpu_credit |
| Classifier vs anomaly | XGBoost if data allows, IsolationForest fallback | Compute has more labellable binary events than Storage |
| Output collection | `22prediction_logs_compute` | Separate from Storage to allow independent querying and alerting |

### Forecast horizon for Compute

Compute resources are more volatile than block storage (CPU spikes in seconds,
not hours). The behavioral forecast still emits `forecast_1h`, `forecast_6h`,
`forecast_24h`, `forecast_7d`, `forecast_30d` for schema consistency — but
downstream consumers should treat `forecast_1h` and `forecast_6h` as the
actionable signals and weight longer horizons accordingly.

### Failure predictions emitted for Compute

| failure_type | Trigger |
|---|---|
| `CPU_80_PERCENT_BREACH` | forecast_1h ≥ 80 on `canonical_cpu_pct` |
| `CPU_90_PERCENT_BREACH` | forecast_6h ≥ 90 on `canonical_cpu_pct` |
| `MEMORY_PRESSURE` | current `canonical_mem_used_pct` ≥ 85 |
| `INSTANCE_HEALTH_CHECK_FAILURE` | `status_check_failed` > 0 |

---

## Running it

```bash
pip install -r requirements.txt

# Inspect raw -> canonical mapping for Storage + Compute docs
python -m src.pipeline inspect --mongo-uri "<uri>" --db "<db>"

# Validate
python -m src.pipeline validate --mongo-uri "<uri>" --db "<db>"

# Train both Storage and Compute models
python -m src.pipeline train --mongo-uri "<uri>" --db "<db>"

# Full run: train + predict, writing to both output collections
python -m src.pipeline run --mongo-uri "<uri>" --db "<db>"

# Dry run — prints first 10 predictions (mix of Storage and Compute)
python -m src.pipeline predict --mongo-uri "<uri>" --db "<db>" --dry-run

# Override output collections
python -m src.pipeline run \
  --out 22prediction_logs_storage \
  --compute-out 22prediction_logs_compute \
  --mongo-uri "<uri>" --db "<db>"
```

### Model artifacts after training

```
models/
  storage_model.joblib   ← unchanged from Storage-only build
  compute_model.joblib   ← new
```

Existing `storage_model.joblib` (if present) is loaded as-is. Compute model
is trained fresh. If neither exists, both are trained automatically on first
`predict` or `run`.
