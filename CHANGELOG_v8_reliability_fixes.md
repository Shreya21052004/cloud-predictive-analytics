# v8 — Confidence/reliability bug-fix pass (Network + Storage)

This pass fixes six concrete bugs found while diagnosing why Network and
Storage predictions scored so poorly (avg `trend_fit_r2` 0.12 for Network;
68.4% of all predictions category-wide flagged low-confidence). None of
these are "retune the model" issues — they were logic bugs corrupting the
confidence pipeline itself.

## 1. [CRITICAL] ~60% of predictions never had a confidence score computed at all

**File:** `src/prediction.py` — `behavioral_forecast()`

Only the main STL/linear-regression/exponential-smoothing branch called
`_compute_prediction_confidence_metrics()`. Every early-return path —
`feature_not_available`, `no_data_points`, `insufficient_points_*` (with or
without a pooled prior), `all_zeros_after_cleaning`, and the entire
`flat_signal` branch — returned without `prediction_confidence_score`,
`accuracy_proxy_score`, `data_completeness_score`, or
`forecast_uncertainty_pct` ever being set. In the sample dataset this was
`flat_signal` (n=366) + `none` (n=336) = 702/1179 predictions (~60%) with a
null confidence score, which downstream reporting evidently treated as
untrustworthy/flagged by default.

**Fix:** added a single exit-point helper, `_finalize_forecast()`, that
every return path now routes through. It computes the full confidence
metrics block exactly once, only if not already present.

## 2. `"moderate"` vs `"medium"` label mismatch silently zeroed out 25% of the score

**File:** `src/prediction.py` — `_compute_prediction_confidence_metrics()`

The main forecast path labels its mid-tier reliability `"moderate"`, but
`reliability_weight` only had keys `"high" / "medium-high" / "medium" /
"low"`. `"moderate"` fell through to the `.get(reliability, 0.5)` default
every time — silently using the wrong weight for a large share of
predictions (anything that wasn't `"high"` or `"low"` in the main path).

**Fix:** added `"moderate": 0.6` to the dict (same weight as `"medium"`,
since they're the same confidence band under different names).

## 3. Network's primary forecast feature doesn't exist for 96% of Network resources

**Files:** `src/metric_registry.py`, `src/prediction.py`

`PRIMARY_FORECAST_FEATURE["Network"] = "canonical_active_connections"` only
applies to AWS/Azure load balancers. `vpc_network` resources (96% of Network
in the sample data) never emit it, so `choose_forecast_feature()` fell back
to plain dict-insertion order — which picked raw, unbounded byte counters
(`canonical_net_in_bytes` / `canonical_net_out_bytes`) ahead of
`canonical_subnet_util_pct`, the bounded 0–100% signal the README itself
advertises for "Subnet/VNet IP capacity exhaustion." Forecasting a metric
with no natural ceiling produces poor trend fits almost by construction.

**Fix:** added `FORECAST_FEATURE_PRIORITY` with a curated fallback order for
Network (and Storage) that prefers bounded, semantically meaningful signals
over raw counters. `choose_forecast_feature()` now uses
`forecast_feature_fallback_order(category)` instead of raw dict order.
Other categories are unaffected (they still use plain registry order).

## 4. OCI `SubnetIpUtilization` was double-mapped with incompatible units

**File:** `src/metric_registry.py`

The same raw OCI metric was registered under both
`canonical_net_in_bytes` (identity/raw passthrough) and
`canonical_subnet_util_pct` (percent transform) — the same value read two
different, incompatible ways depending on which canonical column won the
fallback race.

**Fix:** removed the `net_in_bytes` mapping; it now lives only under
`canonical_subnet_util_pct`, which is what it actually is.

## 5. MAPE-proxy exploded into the billions/trillions on byte-scale metrics

**File:** `src/prediction.py` — `_estimate_error_proxy_metrics()`

`mae_proxy`/`rmse_proxy` divided by a single robust series-level `scale`,
but `mape_proxy` divided each residual by that individual point's own value
(floored at 1.0) — textbook MAPE. On byte-scale features
(`canonical_storage_capacity_raw`, `net_in/out_bytes`, etc.) observations
near zero blew this up into astronomical numbers (matches the observed
`avg_mape_proxy: 4.6 trillion` for Storage), which fed straight into
`accuracy_proxy_score` and corrupted 15% of the composite confidence score.

**Fix:** `mape_proxy` now uses the same robust aggregate `scale` as
`mae_proxy`/`rmse_proxy`, with a defensive 1000% ceiling.

## 6. Unbounded raw features could still claim "high" reliability

**File:** `src/prediction.py` — new `UNBOUNDED_RAW_FEATURES` set + `_finalize_forecast()`

Raw byte/count features with no natural ceiling
(`canonical_storage_capacity_raw`, `net_in/out_bytes`,
`storage_read/write_bytes`, `credit_balance_raw`) could still be labeled
`forecast_reliability = "high"` purely because the trend line looked clean
— even though there's no quota/capacity ceiling to validate the forecast
against (see `STORAGE_BUILD_NOTES.md`, which already flagged this as an open
gap pending a real per-resource quota lookup).

**Fix:** any forecast on one of these features is now capped at `"moderate"`
reliability and tagged with `data_note` containing
`unbounded_raw_metric`, so it's honestly represented as "less validated"
rather than falsely confident. This is a stopgap until a real
quota/size lookup table exists (still an open item, same as noted in
`STORAGE_BUILD_NOTES.md`).

---

## What to expect after this fix

- Every prediction document will now have a real, non-null
  `prediction_confidence_score` — the biggest single driver of the previous
  68.4% "flagged" rate should drop substantially.
- Network predictions should show meaningfully higher `trend_fit_r2` on
  average, since `vpc_network` resources are now forecast on
  `canonical_subnet_util_pct` (or `health_pct`) instead of raw bytes.
- `avg_mape_proxy` for Storage/Network should fall from trillions/thousands
  to a normal 0–a few hundred percent range.
- Fewer predictions will falsely claim "high" reliability on raw/unbounded
  metrics.

**No forced retrain needed.** Fixes 1, 2, 3, 5, and 6 are entirely inside
`behavioral_forecast()` — the separate risk/anomaly `models/*.joblib`
artifacts (trained on the full canonical feature set, not the single chosen
forecast feature) are unaffected and don't need retraining. Fix 4 removes
one OCI provider mapping for `canonical_net_in_bytes`; if you have OCI
Network data, it's worth running `python -m src.pipeline train` again for
`Network` so that column reflects the corrected registry, but it's not
required for the confidence-score fixes to take effect. Either way, run
`predict` (or `run`) fresh — the fixes only apply to newly generated
prediction documents, not ones already written to Mongo.

---

# v8.1 — Real-data findings (Delta-key extraction + GCP mislabeling)

Two more bugs surfaced once this was run against real production data and
diagnosed with actual raw documents (not guessed at). Both were root causes
that no amount of confidence-formula tuning could have fixed, because the
underlying data was never reaching the forecaster at all.

## 7. [CRITICAL] GCP's "Delta" value key wasn't recognized — ~84% of raw Network docs were silently dropped

**File:** `src/normalizer.py` — `STAT_PRIORITY`

Real GCP documents report their metric reading as
`{"Timestamp": ..., "Delta": "10516"}` — a cumulative/interval-delta counter
convention. `STAT_PRIORITY` only recognized
`average/maximum/sum/total/minimum/count/samplecount/value`. `"delta"`
wasn't in that list at all (in any casing), so `extract_metric_value()`
silently returned `(None, None)` for **every** GCP document using this shape.
A live query against the source collection showed GCP is 236,575 of ~283k
raw Network-category documents (84%) — three of its four dominant metrics
(`vm_flow/ingress_bytes_count`, `vm_flow/egress_bytes_count`,
`vpc_flow/predicted_max_vpc_flow_logs_count`) all use this convention. This
is what was actually producing the wall of `forecast_method: "none"` /
`data_note: "no_data_points"` vpc_network predictions — not a
registry-mapping gap, a raw-value-extraction gap. No amount of reordering
`FORECAST_FEATURE_PRIORITY` (fix #3) could help when the value was never
being read out of the document in the first place.

**Fix:** added `"delta"` to `STAT_PRIORITY` (case-insensitive match already
existed via the surrounding `lowered` dict lookup, so this was a one-line
fix). Verified against the exact document shapes from production — GCP VPC
resources now populate `canonical_net_in_bytes`, `canonical_net_out_bytes`,
and the new `canonical_flow_log_count` (see #9) correctly.

Also added a mapping for `networking.googleapis.com/cloud_netslo/active_probing/probe_count`
(56,339 occurrences, the 3rd most common GCP Network metric) as
`canonical_probe_failure_count` — it had no canonical feature registered
at all before, independent of the Delta-key bug.

## 8. GCP had zero real Network documents flowing through at all until #7 — re-verify the numbers after this fix

Because of bug #7, every number reported in the v8 changelog above for
Network (avg confidence 31.4, `vpc_network` r2 0.095) was still computed
almost entirely on the ~16% of Network data that came from AWS/Azure/OCI.
Expect Network's numbers to move substantially once this fix is deployed
and predictions are regenerated — this wasn't visible from confidence-score
math alone, only from inspecting raw documents.

## 9. GCP's "subnet utilization" was never a utilization metric — mislabeling bug independent of #7

**File:** `src/metric_registry.py`

`canonical_subnet_util_pct["GCP"]` was mapped from
`predicted_max_vpc_flow_logs_count` — a flow-log **count prediction**
(observed values 40–400+), not a 0–100% utilization reading, and it wasn't
even passed through the `pct()` transform (just `identity`). This is the
same class of bug as #4 (OCI double-mapping): a raw, unbounded metric
dressed up as a bounded percentage. It directly undermined fix #3, since
that fix specifically prioritizes `subnet_util_pct` *because* it's supposed
to be a real, validated bounded signal — for GCP it was actually just
another unbounded count in disguise, and worse, it was silently competing
with the Delta-key bug so its true (mis)behavior was never visible until
now. GCP has no genuine subnet/VPC IP-utilization metric in this dataset —
Azure (`VirtualNetworkLinkCapacityUtilization`) and OCI
(`SubnetIpUtilization`) do, and those mappings are untouched.

**Fix:** moved the GCP metric to its own honestly-named
`canonical_flow_log_count` feature. Added it to `FORECAST_FEATURE_PRIORITY["Network"]`
(after the true bounded signals, before raw byte counters) and to
`UNBOUNDED_RAW_FEATURES` (alongside the new `canonical_probe_failure_count`)
so neither can claim `"high"` reliability — same honesty principle as fix #6.

---

## What to expect after v8.1

- The wall of `vpc_network` / `forecast_method: "none"` / `no_data_points`
  predictions should mostly disappear — GCP is 84% of your Network volume
  and was previously contributing almost nothing to the forecaster.
- `canonical_subnet_util_pct` should now only ever appear for Azure/OCI
  resources that have a genuine utilization metric; GCP resources will
  correctly show `canonical_flow_log_count` instead, capped at "moderate"
  reliability rather than falsely "high".
- Because this fix touches raw-value extraction (not just the confidence
  formula), it's worth re-running `train` for `Network` at minimum before
  trusting the anomaly/risk model outputs for GCP resources specifically —
  their feature columns were essentially all-null before this fix and the
  existing `models/*.joblib` may have been trained on a Network dataset that
  was ~84% missing.
