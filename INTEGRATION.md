# End-to-end pipeline: Inventory/Account merge -> Predictions -> LLM project summaries -> Project UI

This ties together three previously separate pieces into one flow, matching
the architecture diagram (Service Resource Inventory / Account Metrics ->
Inventory Lookup Cache -> Metadata Enrichment -> Existing Prediction
Pipeline -> Prediction + Alerts + Dashboard + MongoDB), extended with a
project-level LLM summarization stage and a project-scoped read API.

```
MongoDB (Account Metrics Collection, Service Resource Inventory)
        |
        v
ingestion/resource_metadata_enrichment_pipeline.py      <- (1)
        |  writes enriched_data_lake (resource+account merged,
        |   `project` field inventory-authoritative)
        v
src/pipeline.py  predict --source enriched_data_lake     <- (2)
        |  unchanged: still writes to ONE shared collection
        |  per category (22prediction_logs_compute, _storage, ...),
        |  `project` + `account_id` stamped on every prediction doc
        v
scripts/llm_project_summary.py   (cron, every 1-2h)       <- (3)
        |  reads all 5 category collections for the window,
        |  groups by `project`, one map-reduce LLM call per project,
        |  writes to ONE shared 22llm_project_summaries collection
        v
scripts/project_dashboard.py                              <- (4)
        |  read-only query layer for the per-project UI:
        |  filters existing collections by `project`,
        |  enforces account -> project authorization
        v
Project UI (one UI, project-scoped by query, no per-project collections)
```

## One-command real-time mode

Steps (1)-(3) below can each be run by hand, but for the always-on flow
matching the flowchart there's a single entrypoint that runs all of them
together as background threads in one process: `run_pipeline.py` (repo
root). It does NOT replace steps (1)-(3) — it just runs the same code
continuously instead of you invoking each script yourself.

```bash
ollama serve &                          # local model server, once
ollama pull qwen3:8b                    # one-time model download
python run_pipeline.py \
  --mongo-uri mongodb://localhost:27017 --db mydb \
  --predict-interval 300 --train-interval 3600 --summary-interval 5400
```

What it does, concurrently, until Ctrl+C:

- **Ingestion thread(s)** — same change-stream watchers as
  `ingestion/realtime_stream_consumer.py`: Account Metrics + Service
  Resource Inventory -> Inventory Lookup Cache -> Metadata Enrichment ->
  `enriched_data_lake`, live, with resume tokens so a restart doesn't miss
  or replay changes.
- **Predict thread** — every `--predict-interval` seconds, runs
  `src.pipeline` predict against `enriched_data_lake`; every
  `--train-interval` seconds it retrains first (`cmd_train`) so models
  don't go stale, without retraining on every single predict cycle.
- **LLM summary thread** — every `--summary-interval` seconds, pulls the
  last `--summary-window-minutes` of predictions across all category
  collections, groups by project, and writes one summary per project to
  `22llm_project_summaries` — the same logic as
  `scripts/llm_project_summary.py`, using **local Ollama** by default (no API key, `OLLAMA_HOST`, default `qwen3:8b`). Pass `--llm-provider anthropic` with `ANTHROPIC_API_KEY` set to use a cloud model instead.

Project indexes (`ensure_project_indexes`) are created once on startup
unless `--skip-indexes` is passed. Run `python run_pipeline.py --help`
for the full flag list (Mongo URIs, collection names, field names, timers).

## (1) Ingestion / enrichment

`ingestion/resource_metadata_enrichment_pipeline.py` — your original merge
pipeline, with one fix: `project` is now **inventory-authoritative** in
`merge_documents()` (see `INVENTORY_AUTHORITATIVE_FIELDS`). A resource
belongs to exactly one project by definition; if an account doc ever
carries a stale `project` value it can no longer silently win the merge
and misroute a resource's predictions to the wrong project.

Output: `enriched_data_lake` (batch) / same collection kept current via
`watch_account_changes` + `watch_inventory_changes` (real-time, Section 5
of the script).

## (1b) Optional: Kafka as a second ingestion front door

`ingestion/kafka_stream_consumer.py` lets Kafka feed `enriched_data_lake`
**alongside** the Mongo change streams — it doesn't replace them. Both
front doors share the same in-memory Inventory Lookup Cache and write to
the same `enriched_data_lake` collection, so anything downstream (predict
loop, LLM summaries, dashboard) is unaware of which door a row came in
through.

```
Kafka <account-topic>    --> lookup(resourceId, account_id) in cache --> merge --> enriched_data_lake
Kafka <inventory-topic>  --> cache update + append to data-lake collection
```

Enable it with one flag on the existing one-command entrypoint:

```bash
pip install kafka-python   # already in requirements.txt
export KAFKA_BOOTSTRAP_SERVERS=localhost:9092
python run_pipeline.py --enable-kafka \
  --kafka-account-topic cloud-account-metrics \
  --kafka-inventory-topic service-resource-inventory
```

Message schema on each topic mirrors the corresponding Mongo document
(same field names as `cloud_account_data_collection_daily_updated` /
`service_resource_inventory_v2_updated`); the inventory topic accepts an
optional `operation_type` field (`insert` / `update` / `delete`, defaults
to `upsert`).

Offsets are committed manually, one message at a time, only after the
Mongo write succeeds (at-least-once delivery); the resulting Mongo upserts
are idempotent so safe reprocessing on restart is guaranteed the same way
the Mongo change-stream resume tokens guarantee it for that path.

It can also run as its own standalone process instead of inside
`run_pipeline.py`:

```bash
python -m ingestion.kafka_stream_consumer --db mydb \
  --kafka-account-topic cloud-account-metrics \
  --kafka-inventory-topic service-resource-inventory
```

See `--help` on either command for broker/security flags (SASL/SSL for
Confluent Cloud, MSK, etc.) and `.env.example` for the equivalent env vars.

## (2) Prediction pipeline

No code changes needed. `src/normalizer.py::base_context()` already reads
every field the merge produces (`project`, `account_id`, `component`,
`location`, `tags`), so pointing `--source` at `enriched_data_lake` is a
config change, not a code change:

```bash
python -m src.pipeline predict \
  --mongo-uri mongodb://localhost:27017 \
  --db mydb \
  --source enriched_data_lake
```

`_collection_for()` in `pipeline.py` deliberately keeps one shared
collection per category regardless of project — that's unchanged and
correct. Project routing never happens at write time, only at read time
(step 4).

## (3) Per-project LLM summarization

`src/llm_summary.py` + `scripts/llm_project_summary.py`. Runs as its own
scheduled job (not inline in `predict`), because the 5 category
collections are populated independently — a project-level view has to be
assembled after the fact across all of them for a given window.

```bash
ollama serve &            # local model server, once
ollama pull qwen3:8b      # one-time model download
python scripts/llm_project_summary.py \
  --db mydb --window-minutes 90 --provider ollama --model qwen3:8b
```

Pass `--provider anthropic` (with `ANTHROPIC_API_KEY` set) to use a cloud
model instead — Ollama is just the local default.

Per project per window: build a compact digest per prediction, pack by
token budget (not resource count), map-reduce if a project's resource
count blows one context window, write ONE summary doc to the shared
`22llm_project_summaries` collection (never a per-project collection).

## (4) Project-scoped UI read path

`scripts/project_dashboard.py`. No new collections anywhere — every
query here is a `{"project": project_id}` filter against the same shared
collections written in steps 2 and 3. This is the *only* place project
routing logic should live; don't duplicate the filter/auth logic in the
frontend.

```bash
# once, and after any collection rename:
python scripts/project_dashboard.py create-indexes --db mydb

# per UI page load:
python scripts/project_dashboard.py fetch \
  --db mydb --account-id acct_123 --project-id proj_a
```

`get_project_dashboard()` enforces account->project authorization by
resolving the account's real project from the data itself
(`account_can_access_project`) before returning anything — it never
trusts a `project_id` passed directly from the client.

## Required indexes (run once)

```python
from scripts.project_dashboard import ensure_project_indexes
ensure_project_indexes(mongo_uri, db_name)
```

Creates `(project, prediction_timestamp)` on all 5 category collections
and `(project, generated_at)` on `22llm_project_summaries`. Without these,
every UI dashboard load and every cron summarization run collection-scans.

## Files added/changed in this pass

- `ingestion/resource_metadata_enrichment_pipeline.py` — `project` made inventory-authoritative in merge
- `src/llm_summary.py` — new
- `scripts/llm_project_summary.py` — new
- `scripts/project_dashboard.py` — new
- unchanged: `src/pipeline.py`, `src/normalizer.py`, `src/config.py`, `src/mongo_io.py`, everything else in `src/`
