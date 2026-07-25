# Predictive Analytics Pipeline

This folder contains a complete local MongoDB pipeline for cloud predictive analytics.

It reads from:

- MongoDB URI: `mongodb://localhost:27017`
- Database: `mydb`
- Source collection: `22predictive_analytics_dataset_sorted`

It only processes these source categories:

- `Compute`
- `Network`
- `Networking` (normalized to `Network`)
- `Storage`

Network resources may be stored under either the `Network` or `Networking`
raw category spelling in the source collection — both are queried from
MongoDB and normalized to the single canonical `Network` category, so no
Network data is silently dropped regardless of which spelling was used
when the document was written.

It writes prediction logs to separate per-category collections:

- `22prediction_logs_storage`
- `22prediction_logs_compute`
- `22prediction_logs_network`

## What this can predict

### Compute

- CPU saturation forecast: estimates if CPU will cross 80% or 90% in the next 1h, 6h, 24h, or 7d.
- Memory pressure: classifies memory state as `OK`, `WARNING`, or `CRITICAL`.
- Disk I/O bottleneck or anomaly: uses disk bytes, IOPS, queue, iowait, and load signals.
- Network throughput anomaly: detects unusual inbound/outbound traffic for compute resources.
- Instance health risk: uses AWS status checks, GCP boot integrity, OCI load, and related signals.
- AWS-specific Lambda and CPU credit risk when those metrics exist.

### Network

- DDoS probability: strongest for Azure, proxy-only for AWS.
- Load balancer health and connection saturation.
- Subnet/VNet IP capacity exhaustion.
- DNS/query or throughput anomaly.

### Storage

- Storage capacity exhaustion forecast.
- I/O performance degradation.
- AWS burst credit exhaustion.
- Availability, backup, and access anomaly risk.

## Run the whole thing live, one command

The batch commands below (`inspect` / `validate` / `train` / `predict`) are
for one-off runs. For the full real-time flow shown in the architecture
diagram — MongoDB change streams -> enrichment -> prediction -> local Ollama LLM
project summaries, all running continuously — use `run_pipeline.py`
instead. See [INTEGRATION.md](INTEGRATION.md#one-command-real-time-mode)
for details. Short version:

```bash
ollama serve &                          # start the local model server
ollama pull qwen3:8b                    # one-time model download
python run_pipeline.py
```

**Prerequisite:** change streams only work against a MongoDB replica set,
not a standalone `mongod`. If you're running MongoDB locally (not Atlas),
do this once:

```bash
mongosh --eval 'rs.initiate()'
```

If you skip this, `run_pipeline.py` still starts, but the ingestion
threads will print a one-time explanation and keep retrying every 30s
until you fix it — everything else (predict/LLM loops) keeps running.

That single command keeps running (Ctrl+C to stop) and:
1. On first startup, runs a one-time batch merge of whatever account +
   inventory documents already exist into `enriched_data_lake` (so you're
   not stuck waiting for something to *change* before there's any data to
   predict on), then watches both collections via change streams to keep
   it current. Pass `--skip-backfill` to skip this, or `--force-backfill`
   to redo it even if the collection already has documents.
2. Re-trains models periodically and runs predictions on a timer.
3. Runs the local Ollama-powered per-project LLM summary pass on its own timer.

## Install

From this folder:

```bash
cd predictive_analytics_pipeline
/usr/bin/python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Use Python 3.12 for this project. On this machine, plain `python3` points to Anaconda Python 3.13, which can force packages such as pandas to compile from source.

Make sure MongoDB is running locally and your data exists:

```bash
mongosh mydb --eval 'db["22predictive_analytics_dataset_sorted"].countDocuments()'
```

## Run the whole pipeline

```bash
python -m src.pipeline run --dry-run
```

This will:

1. Load data from MongoDB.
2. Normalize raw cloud-provider metrics into canonical features.
3. Validate normalized ranges.
4. Train separate model families for `Compute`, `Network`, and `Storage`.
5. Generate prediction logs in your requested format.
6. Print the first predictions without writing because `--dry-run` is enabled.

When the dry run looks correct, write to MongoDB:

```bash
python -m src.pipeline run
```

## Run steps separately

```bash
python -m src.pipeline inspect
python -m src.pipeline validate
python -m src.pipeline train
python -m src.pipeline predict
```

## Useful options

```bash
python -m src.pipeline run --mongo-uri mongodb://localhost:27017 --db mydb --source 22predictive_analytics_dataset_sorted --out 22prediction_logs_storage --compute-out 22prediction_logs_compute --network-out 22prediction_logs_network
python -m src.pipeline predict --dry-run
python -m src.pipeline train --models-dir models
python -m src.pipeline run --limit 10000 --dry-run
python -m src.pipeline inspect --limit 10000
python -m src.pipeline run --limit 0
```

By default the pipeline loads at most `50000` Mongo documents to avoid large RAM spikes. Use `--limit 0` only when you are ready to process the full collection.

## Output format

Each prediction document follows this shape:

```json
{
  "prediction_id": "pred_20250614_a3f9c2",
  "prediction_type": "risk_score",
  "model_version": "compute-anomaly-v1.0",
  "generated_at": "2025-06-14T10:00:00Z",
  "resource_id": "<resourceId>",
  "resource_name": "vm-prod-eastus-01",
  "account_id": "sub-xxxx",
  "category": "Compute",
  "service_family": "Compute",
  "provider": "Azure",
  "component": "Virtual_Machines",
  "location": "eastus",
  "prediction_timestamp": "2025-06-14T10:00:00Z",
  "data_completeness_score": 0.33,
  "behavioral_forecast": {
    "metric_name": "Percentage CPU",
    "current_value": 62.4,
    "forecast_1h": 65.1,
    "forecast_6h": 70.2,
    "forecast_24h": 58.3,
    "trend_direction": "increasing",
    "confidence": 0.85
  },
  "failure_predictions": [],
  "anomaly_score": 0.0,
  "is_anomalous": false,
  "recommendations": [],
  "payload": {},
  "alert": {},
  "dashboard": {},
  "feedback": {
    "outcome_recorded": false,
    "outcome_label": null,
    "outcome_at": null
  }
}
```

## Notes

- **Baselines and anomaly detection are pooled by resource type, not per
  resource.** `canonical_resource_type` (see `src/resource_types.py`) maps
  raw components like `EC2`, `Compute_Engine`, and `Virtual_Machines` to one
  shared type (`vm_instance`). The diurnal baseline profile and the anomaly
  IsolationForest are each fit ONCE per `(canonical_resource_type,
  category)`, pooling data across every resource of that type — never per
  individual `resource_id`. Each resource is only ever scored against its
  type's shared model. See `CHANGELOG_v7_type_pooling.md` for details.
- The pipeline does not fill missing cloud metrics with `0`. It creates `*_available` flags so models can learn from missingness safely.
- `metric_value` can be either a single object or a list of timestamped points. List values are expanded into one normalized row per timestamp.
- Predictions are generated only for `Compute`, `Network`, and `Storage`. Mongo reads are filtered to those raw categories — including both `Network` and `Networking` spellings — so unsupported categories such as `Governance` and `Web` are not loaded.
- `Network` and `Networking` are two raw spellings for the same logical category in the source data. Both are matched at the MongoDB query layer and normalized to canonical `Network` before any modeling, profiling, or anomaly scoring happens — Network predictions cover the full Network dataset regardless of which spelling individual documents use.
- If there are no incident labels yet, training uses anomaly detection plus threshold-derived weak labels. Once you add real incident outcomes, the same structure can be upgraded to supervised models.
- Azure memory percentage can be derived only when total VM memory is available. If it is not available, memory pressure uses available direct signals and availability flags.
