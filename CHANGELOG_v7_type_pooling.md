# v7 — Baseline profiling and anomaly detection pooled by resource type

## What changed

Previously, two things in the pipeline were trained/fit **per individual
resource_id**:

1. **`profiler.py`** built a baseline (mean/std/p5/p95 + a 168-bucket
   diurnal day-of-week × hour-of-day pattern) from that one resource's own
   history only.
2. **`anomaly.py`** Signal 3 fit a fresh `IsolationForest` on that one
   resource's own residual history, every time it was scored.

That meant a resource needed real history of its own before it got a
trustworthy baseline or anomaly score, and two resources of the exact same
type (e.g. two `EC2` instances) never shared any statistical strength.

**Now, nothing is fit to a single resource's own history.** Instead:

1. `profiler.build_all_profiles(df)` groups by
   `(canonical_resource_type, category)` — see `resource_types.py` for the
   raw-component → canonical-type mapping (e.g. `EC2`, `Compute_Engine`,
   `Virtual_Machines` all map to `vm_instance`). One profile is built per
   type, pooling every resource that maps to it.
2. `anomaly.train_type_isolation_forests(df, category, profiles)` fits one
   `IsolationForest` per canonical_resource_type, pooling residuals across
   every resource of that type. It's called once per category, up front
   (in `prediction.generate_predictions`), not once per resource.
3. At prediction time, each resource looks up its type's profile and its
   type's IsolationForest and is **scored** against them
   (`anomaly.score_resource`). A resource's own timeseries (`history`) is
   only ever used to know its current/recent values — never to fit a model.

This means:
- A brand-new resource with only a few hours of data still gets a fully
  formed baseline and anomaly score on day one, because it borrows its
  type's pooled model instead of needing to build its own.
- Two resources of the same type genuinely share statistical strength —
  what's "normal" for a `vm_instance` at 3pm on a Tuesday is learned from
  every `vm_instance`, not re-derived per resource.
- Types with too little pooled data (`< MIN_TYPE_POOL_FOR_IF` = 100 rows)
  simply have Signal 3 switched off (weight 0) rather than falling back to
  a resource-specific fit.

## What did NOT change

- `models.py`'s risk classifier (`train_category`) was already trained
  pooled at the **category** level (Compute/Network/Storage), not per
  resource — that's unchanged.
- The slope-prior pooling for forecast trend shrinkage
  (`build_type_specific_slope_priors`) was already pooled by
  `canonical_resource_type` — that's unchanged, and the new profile/anomaly
  pooling now uses the same granularity for consistency.

## Storage

- The profiles collection is renamed from `22resource_profiles` to
  `22resource_type_profiles` (documents are now keyed by
  `(canonical_resource_type, category)` instead of `(resource_id,
  category)`), reflecting that a profile document no longer represents one
  resource.

## Files touched

- `src/profiler.py` — profiles pooled by canonical_resource_type.
- `src/anomaly.py` — Signal 3 IsolationForest pooled by
  canonical_resource_type (`train_type_isolation_forests` +
  `score_resource(..., type_if_artifact=...)`).
- `src/prediction.py` — builds the pooled profiles/IF models once per
  category, looks each resource up by its `canonical_resource_type`.
- `src/pipeline.py` — renamed collection, updated log messages.
- `src/explainability.py` — updated narrative wording ("this resource's
  baseline" → "this resource type's pooled baseline").
