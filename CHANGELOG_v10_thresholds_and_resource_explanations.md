# v10 — manual thresholds + per-resource LLM explanations

Addresses three requirements that project-level LLM summaries didn't cover:

## 1. Manual per-resource thresholds + breach tracking (`src/thresholds.py`)

- `set_threshold(uri, db, resource_id, metric_name, operator, value, ...)` —
  one active rule per (resource_id, metric_name); setting again replaces it.
- Every prediction cycle, `evaluate_resource()` checks the resource's current
  value against each of its rules, logs an idempotent breach event (so
  re-processing the same prediction cycle never double-counts), and refreshes
  a rolling breach-state doc with `breach_count`, `last_breached_at`.
- `predict_next_breach()` interpolates across the existing
  current/1h/6h/24h/7d/30d forecast points (no new modeling) to project the
  earliest time the metric crosses the threshold — or reports "already
  breaching" / "not projected to cross within 30d".
- CLI: `python -m scripts.resource_dashboard set-threshold --resource-id ...
  --metric-name ... --operator gt --value 90`, plus `list-thresholds`,
  `delete-threshold`, `breach-state`.
- New collections: `22resource_thresholds`, `22threshold_breach_log`,
  `22resource_breach_state`.

## 2. Per-resource LLM explanation (`src/llm_resource_explain.py`)

- One LLM call per resource (not per project) — full metric set (all
  forecast horizons + CI bands, anomaly signals, alert severity, failure
  predictions, tags, summary_details) plus threshold/breach state if any
  rule is set. Explicitly prompted as a senior multi-cloud engineer (AWS,
  Azure, GCP, OpenStack) so recommendations use provider-correct
  terminology instead of generic advice.
- Written to `22llm_resource_explanations`, one doc per `resource_id`
  (latest wins), with `resource_id` and `tags` as top-level indexed fields —
  queryable either way (`scripts/resource_dashboard.py get` /
  `by-tag`, or `GET /api/resource/{id}` / `GET /api/resources?tag_key=...`
  on `ui/backend.py`).
- Uses the same incremental-write-per-item fix already applied to the
  project-summary loop: each resource's explanation is written the moment
  it's ready, not buffered until every resource in the cycle is done.

## 3. Wiring into `run_pipeline.py`

- New background thread, `resource_explain_loop`, on its own timer
  (`--resource-explain-interval`, default 5 min).
- `--resource-explain-mode priority` (default): only spends an LLM call on
  resources that are anomalous, alerting, or breaching a threshold this
  cycle — explaining every resource every cycle multiplies call volume by
  (resource count / project count) and re-creates the rate-limit stall the
  project-summary loop already hit once. `--resource-explain-mode all`
  explains everything, no filtering — budget your provider's rate limits
  (or use `--llm-provider anthropic` for this step) before turning it on.
- Threshold evaluation itself always runs over every resource in the
  window regardless of mode, so breach counts/ETAs stay accurate even for
  resources that don't get an LLM explanation that cycle.
- `--skip-resource-explain` to disable the thread entirely.
