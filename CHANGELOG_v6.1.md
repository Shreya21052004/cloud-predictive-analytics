# Pipeline v6.1 — Patch Notes

Built on top of your existing v6 pipeline. **No cost prediction was added**, per request.

## What was already in v6 (confirmed, not rebuilt)
Your pipeline already had Option 3 (per-resource baseline profiling) fully implemented:
- `profiler.py` — diurnal (day-of-week × hour) baselines, regime-change detection, reporting-cadence tracking
- `anomaly.py` — residual-based scoring (z-score + burst ratio + IsolationForest on residuals, not raw values)
- `data_quality.py` — dedup, freshness gate, min-history guard, drift trigger, feedback staleness check, alert dedup
- `feedback.py` — incident matching, true/false positive labelling, precision/recall, training-label export
- `pipeline.py` — all of the above wired together end-to-end via `cmd_run` / `cmd_feedback`

This was a more mature pipeline than the original message assumed — most of the "critical gaps" were
already closed. The real gaps were narrower than first described.

## What's new in this patch

### 1. `src/explainability.py` (new)
Every prediction now gets an `explanation` field with a plain-English narrative built from signals
that already existed (`anomaly_detail`, `behavioral_forecast`, `profile_info`) — no new ML, just a
templating layer. Example output:

> "Risk score 82/100 for this Compute resource. Current value is far outside this resource's own
> historical pattern for this time of day/week. cpu pct has been trending up ~2.10/hour (forecast
> reliability: high). Profile confidence: 0.86 (24d of history)."

Wired into `pipeline.py::predict_from_df` via `attach_explanations(predictions)`.

### 2. `src/maintenance_windows.py` (new)
Reads a `22maintenance_windows` MongoDB collection (resource_id/category/tag-filter + start/end +
reason). Any prediction falling inside an active window gets its severity downgraded to INFO,
`alert.trigger=False`, `alert.suppressed=True`, and a `maintenance_context` block — visible for
audit, but won't page anyone. Fails open (skips silently) if the collection doesn't exist yet, so it
doesn't break existing deployments.

### 3. `src/cross_domain.py` (new)
Groups same-run predictions by `(account_id, component/location)` and checks how many distinct
categories are simultaneously anomalous. 2+ categories anomalous on the same logical unit emits a
`composite_incident` (written to `22composite_incidents`) with a `likely_cause` hint
(e.g. Compute+Network → "possible traffic surge or DDoS-adjacent pattern"). Contributing predictions
get a `composite_incident_id` so they group in one incident view instead of N separate pages.

### 4. `src/anomaly_clustering.py` (new)
Groups same-run predictions by `(provider, location, category)`. If ≥30% of a group (min 5
resources) is anomalous simultaneously, emits a `fleet_anomaly` (written to `22fleet_anomalies`) —
"likely a shared/platform-level cause, not independent incidents." Points the on-call team at
provider status pages / fleet deploys before they triage 50 individual tickets.

### 5. Regime-change recommendation (prediction.py, small addition)
`profiler.py` already detected `regime_changed` and surfaced it in `profile_info`, but
`recommendations()` never acted on it. Added one conditional: when a sustained baseline shift is
detected, the prediction now explicitly recommends checking whether it's an intentional change
(deploy/scale-up) vs. unexpected drift, instead of silently flagging the new normal as an ongoing
anomaly indefinitely.

## Wiring (pipeline.py)
All four new modules are called in `predict_from_df`, after `generate_predictions()` and before
writing to Mongo:
```
predictions = generate_predictions(...)
attach_explanations(predictions)
apply_maintenance_suppression(predictions, active_windows)   # fails open if collection missing
composite_incidents = detect_composite_incidents(predictions)
fleet_anomalies = cluster_anomalies(predictions)
# ... existing drift check, dry-run, per-category writes ...
write_predictions(..., COMPOSITE_INCIDENTS_COLLECTION, composite_incidents)
write_predictions(..., FLEET_ANOMALIES_COLLECTION, fleet_anomalies)
```

## New collections introduced
| Collection | Purpose |
|---|---|
| `22maintenance_windows` | Input — you populate this from your change calendar/CI pipeline |
| `22composite_incidents` | Output — cross-domain correlated incidents |
| `22fleet_anomalies` | Output — blast-radius/platform-wide anomaly clusters |

## Explicitly NOT done (per your instruction)
- No cost forecasting model
- No FinOps spend prediction layer

## Still recommended but not built this round (needs product/infra decisions, not just code)
- Cascading failure / dependency-graph prediction — needs a topology data source first
- SHAP-based feature attribution — `explainability.py` gives template-based explanations now; SHAP
  would require retraining hooks into `models.py` and is a bigger lift
- LLM-generated incident narratives — `explanation.narrative` is the deterministic foundation this
  would sit on top of
- Autoremediation hooks — needs a remediation API contract with the enterprise client first

## v6.1.1 (unreleased) — project-aware correlation & per-project output routing

- `mongo_io.py`: `load_documents` now fetches the `project` field; added
  `list_collections_with_prefix` for discovering per-project output
  collections at runtime.
- `normalizer.py`: `base_context` now carries `project` through to the
  working DataFrame.
- `prediction.py`: every prediction now includes a `project` field.
- `cross_domain.py`: composite-incident correlation key changed from
  `account_id` to `project` (falling back to `account_id` for legacy
  records with no project). Rationale: a single project can span multiple
  cloud accounts/subscriptions across providers, so account_id alone
  under-grouped cross-provider composite incidents (e.g. Compute in AWS +
  Database in Azure for the same project were previously never correlated).
- `pipeline.py`: predictions and composite incidents are now written to
  per-project collections (`<base>__<project>`, e.g.
  `22prediction_logs_compute__proj_a`), instead of one shared collection
  per category. Records with no project route to a `__unassigned`
  collection rather than being dropped. Fleet anomaly clusters
  (`anomaly_clustering.py`) remain on a single shared collection —
  intentionally not split per project, since they exist to catch
  cross-project/platform-wide blast radius. `cmd_feedback` updated to
  discover and iterate per-project collections instead of assuming one
  fixed collection per category.
- `scripts/backfill_project_collections.py` (new): one-time migration of
  documents in the old shared collections into the new per-project
  collections. Old collections are left intact by default
  (`--delete-source` to remove migrated docs after verifying).
