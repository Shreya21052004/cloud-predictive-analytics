# What changed

## 1. Predict cycles are now incremental, not a full re-scan

- `src/mongo_io.py`: `load_documents()` takes an optional `extra_query` dict,
  merged into the Mongo filter (used to filter on `merged_at` or a set of
  `resourceId`s without touching the existing `categories`/`limit` logic).
- `src/pipeline.py`: new `cmd_predict_incremental(config, since_iso)`.
  - Finds which resources got a doc with `merged_at > since_iso` (cheap —
    only reads what's new, not the whole collection).
  - Pulls full history for *just those resources* (still needed so
    forecasting/pooling has enough data points per resource).
  - Runs prediction only on that slice.
  - Returns the new cursor (max `merged_at` seen) for next cycle.
- `run_pipeline.py`: `predict_loop` now persists this cursor to
  `<state-dir>/predict_cursor.json` and calls `cmd_predict_incremental`
  instead of `cmd_predict` every cycle. `--predict-interval` can now
  safely be lowered (e.g. 30-60s) since each cycle is cheap.
- Training (`cmd_train`) is untouched — it still does a full/limited sweep
  on `--train-interval` (default 1h), which is correct: retraining should
  see all the data, only *predicting* should be incremental.
- First-ever run (no cursor yet) still predicts everything once, same as
  before, so there's history to work from.

No new CLI flags are required — this is a drop-in behavior change.

## 2. Per-project live dashboard (`ui/`)

- `ui/backend.py` — FastAPI read-only API. Reuses the existing
  `scripts/project_dashboard.py` query functions (so project-routing logic
  stays in one place). One endpoint:
  `GET /api/project/{project_id}` → latest LLM summary/recommendation +
  latest forecast per resource (with tags) for that project.
- `ui/index.html` — single static page. Enter a project id, it polls the
  API every 5 seconds and renders:
  - The project's LLM narrative, risk level badge, priority alerts, and
    recommended action order.
  - One card per resource: name/id, tags, provider/location, the
    1h/6h/24h/7d/30d forecast, trend + reliability, anomaly flag, and any
    per-resource recommendations.
- No new database writes, no new collections — purely reads what
  `run_pipeline.py` is already producing.

### Running it

```bash
pip install -r ui/requirements.txt
export MONGO_URI="mongodb://localhost:27017"
export MONGO_DB="mydb"
uvicorn ui.backend:app --reload --port 8010
```

Then open `http://localhost:8010/` (backend serves `index.html` directly),
type a project id, click View. To run the frontend separately (e.g. from a
different host), open `ui/index.html` on its own and set `API_BASE` at the
top of its `<script>` to the backend's URL.

### Known simplification

This UI skips the account-level authorization check that
`scripts/project_dashboard.py::get_project_dashboard` enforces (it needs an
`account_id` to verify the caller may see that project). This version is
"anyone who knows the project id can view it" — fine for an internal/ops
dashboard, but add an auth layer before exposing it externally.
