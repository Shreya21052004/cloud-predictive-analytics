# v9.1 — Backtest denominator/band robustness fix

## Background

Auditing real `evaluation_basis: "out_of_sample_backtest"` predictions
turned up two features with pathological aggregate stats:
`canonical_storage_capacity_raw` (mase_mean 23.7, coverage_mean 36.8%
across 326 predictions) and `canonical_subnet_util_pct` (mase_mean 10.9,
coverage_mean 9.7% across 62 predictions) -- individual predictions with
MASE in the hundreds to low thousands and 0% interval coverage.

We traced the mechanism, not just the symptom: in both `naive_scale()`
(the MASE denominator) and `_linear_backtest_forecast()`'s CI band width,
a training window with near-zero variance produces a near-zero
denominator/band. Any real deviation in the holdout window -- for ANY
reason -- then gets divided by/measured against something close to zero,
producing an inflated MASE and a missed interval regardless of whether
that holdout deviation was itself large or small.

**What this fix does NOT claim:** we have not confirmed via raw telemetry
inspection whether `canonical_storage_capacity_raw` / `canonical_subnet_util_pct`
are genuinely staircase/step-change series, continuously-varying-but-noisy
series with a short/sparse training window, or something else. That
investigation is still open. This fix does not special-case those two
features and does not assume a step-change model is the right one --
it fixes the general mechanical failure mode (near-zero denominator /
near-zero band width against real-world variation) that would produce
exactly these symptoms under any of those root causes.

## Changes -- `src/backtest.py`

- `naive_scale()` now returns `(scale, floored)` instead of just `scale`.
  When the raw computed scale falls below `1% of the training series'
  mean absolute value`, it's floored to that 1% value rather than passed
  through as-is or forced to `None`. Genuinely all-zero series (no
  magnitude to floor against) are unaffected -- they still correctly
  return `(None, False)`, same as before.
- `_linear_backtest_forecast()` applies the same relative floor to
  `resid_std` before it's used to build the noise margin, and now returns
  a 4th value, `resid_floored`, alongside `(point, lower, upper)`.
- `run_holdout_backtest()` surfaces a new field,
  `backtest_low_variance_training` (bool), true whenever either floor was
  applied for that prediction.

## Changes -- `src/prediction.py`

- `_finalize_forecast()`: when `backtest_low_variance_training` is true,
  `forecast_reliability` is capped at `"moderate"` even if
  `backtest_quality_score >= 80`, with `data_note` set to
  `"low_variance_training_window"` (or appended if a note already exists).
  Same treatment `UNBOUNDED_RAW_FEATURES` already gets, and for the same
  reason: a score propped up by a floored denominator/band shouldn't
  outrank one earned against real variance.

## What downstream consumers should do differently

- Anything already filtering on `evaluation_basis: "out_of_sample_backtest"`
  gets this transparently -- MASE values in the hundreds/thousands should
  no longer appear (verify against your own data; the floor bounds the
  *denominator*, not the numerator, so a genuinely huge point-forecast
  error can still produce an elevated but bounded MASE).
- New field: filter or group by `backtest_low_variance_training` to see
  which predictions relied on the floor. This is the honest signal to use
  if you want to exclude potentially-propped-up scores entirely rather
  than accept the `"moderate"` downgrade.


---

# v9.2 — Adaptive method selection ported to the live forecaster

## What changed

v9.1 fixed the backtest module's scoring (MASE denominator / CI band
floors). It did NOT change what the live forecaster actually predicts —
`forecast_1h`/`forecast_24h`/etc. still came from an unconditional
STL-trend-plus-bootstrap fit, regardless of whether that model was
actually the best available for a given resource. This entry closes that
gap: `select_forecast_method()` (from `backtest.py`, imported directly —
not reimplemented) is now the primary path in `behavioral_forecast()`'s
capacity/utilisation branch (the branch `canonical_storage_capacity_raw`
and `canonical_subnet_util_pct` go through, since neither is in
`BURSTY_METRICS`).

**Why this is a real improvement, not another rescaling trick:** the
pre-existing v9.1 fix changed how error was *measured*. This changes what
gets *predicted*. `backtest_mae` (the model's real absolute error against
held-out actuals) is directly affected, because the point forecast itself
is now whichever of 5 methods (naive persistence, moving average, linear
regression, STL trend, seasonal-naive) won a leakage-safe nested
validation on that resource's own history — see backtest.py's
`select_forecast_method()` docstring for the leakage-safety argument.

## Motivating evidence

Median out-of-sample MASE for `canonical_storage_capacity_raw` was 1.42
under the old always-fit-a-trend approach — meaning the median prediction
was already worse than doing nothing (naive persistence). That's a real
finding about the wrong method being forced onto the wrong kind of
series, not a measurement artifact.

## Scope / what's NOT changed

- Only the capacity/utilisation ("else") branch of `behavioral_forecast()`
  was touched. `BURSTY_METRICS` features (network/disk bytes, IOPS,
  Lambda metrics, etc.) still use the original seasonal-naive /
  exponential-smoothing logic, untouched, since that logic already
  includes shrink-toward-prior pooling for thin series that adaptive
  selection does not replicate.
- The original STL/bootstrap logic in the capacity/utilisation branch is
  KEPT, verbatim, as the fallback for series too short for
  `select_forecast_method`'s own nested-holdout requirement (same
  `MIN_TRAIN_POINTS`/`MIN_HOLDOUT_POINTS` guard used everywhere else in
  this pipeline) -- thin series see no behavior change.
- New `data_note` value `adaptive_selection_no_pooling_shrink` marks
  predictions that went through the new path, since it does not apply the
  type-prior shrinkage the old branch's short-series fallback does.
- `forecast_method` is prefixed `adaptive_` (e.g. `adaptive_trend`,
  `adaptive_naive_persistence`) so it's distinguishable in output from the
  old fixed method names.

## Bug found and fixed during testing (before this shipped)

Initial wiring passed raw horizon offsets (`[1, 6, 24, 168, 720]`)
directly as `test_x` to `select_forecast_method`. The candidate
forecasters compute `point = base + slope * (test_x - base_x)` with
`base_x` anchored at the end of the training window in ABSOLUTE
hours-since-series-start -- so raw offsets (small numbers) minus a large
`base_x` produced forecasts effectively projected backward in time. Caught
by a regression test asserting forecasts increase monotonically for a
known increasing series -- they didn't (`forecast_1h` came out below
`forecast_30d` in a way inconsistent with the fitted trend). Fixed by
building `test_x` the same way `backtest.py` does: last historical
x-value + each offset, not the offset alone.

## Verification performed (see test transcript in conversation, not just
## asserted here)

- Near-flat-then-jump series -> selects `naive_persistence`,
  `forecast_1h` correctly anchors at `current_value`.
- Genuinely flat noisy series -> `moving_average`.
- Real linear trend -> `trend`, forecasts increase monotonically,
  `trend_direction` correctly "increasing".
- Diurnal seasonal pattern IN A FEATURE NOT LISTED IN `BURSTY_METRICS`
  (`canonical_storage_capacity_raw` given synthetic diurnal data) ->
  correctly selects `seasonal_naive` anyway, something the old
  feature-name-based branching could never do since seasonality detection
  was previously gated entirely on static membership in
  `BURSTY_METRICS`, not on the data itself.
- Bursty-metric branch and thin-series fallback confirmed byte-for-byte
  unchanged in behavior.



---

# v9.3 — Rolling-origin validation, ETS/Theta/Croston, structural break detection

Four changes, in priority order requested: rolling-origin validation for
model selection, an ETS model, Theta/Croston for intermittent series, and
structural break detection with retraining after a break.

## 1. Rolling-origin validation (replaces the single nested split)

**Motivation, from real data, not theory:** the v9.2 accuracy summary
showed `adaptive_moving_average` (n=80, mase 5.35) and `adaptive_trend`
(n=76, mase 3.17) both selected on Network resources, yet scoring far
worse than `adaptive_naive_persistence` (mase 1.66) on the real held-out
backtest -- the signature of a method winning a single small validation
split by luck, not by generalizing.

**Fix:** `select_forecast_method()` now evaluates every candidate across
up to `MAX_ROLLING_FOLDS` (4) independent expanding-window folds carved
out of the training data, and averages each method's validation MAE
across the folds it was eligible for, before picking a winner. Still
leakage-safe by construction -- every fold stays inside `train_x`/
`train_y`, never touching the real backtest holdout or (in live
production) the actual future. Degrades gracefully to a single fold on
short series (same behavior as v9.2), so short series see no regression.

## 2. ETS (Holt's linear trend / double exponential smoothing)

New candidate `ets_trend`: exponentially-weighted level and trend,
updated per-observation with `alpha`/`beta` decay. Genuinely
complementary to `linear_regression`/`trend` (both OLS-family, weight all
training points equally) -- ETS tracks a resource whose recent regime
differs from its older history, rather than being dragged toward an
average across the whole window.

## 3. Theta method + Croston's method

- `theta`: classic Assimakopoulos & Nikolopoulos (2000) Theta method --
  blends a linear-trend extrapolation with an SES forecast of the
  trend-amplified line. Self-tuning blend of "follow the long-run trend"
  and "follow the recent level."
- `croston`: for intermittent series (mostly zero, occasional nonzero
  spikes -- e.g. read/write bytes on a mostly-idle volume). Separately
  smooths event size and inter-arrival gap, forecasts their ratio as a
  constant rate. Only eligible when the training window's zero-fraction
  indicates real intermittency (`INTERMITTENT_ZERO_FRAC_MIN`=0.3 to
  `_MAX`=0.9) -- verified it does NOT engage on dense series, and
  verified it can lose fairly to `seasonal_naive` when spikes are
  actually periodic rather than irregular (a period-4 spike pattern
  correctly went to `seasonal_naive`, not `croston` -- periodic beats
  rate-based when the exact position is predictable).

## 4. Structural break detection + retrain after break

`detect_last_structural_break()` scans for a recent large level shift
(capacity resize, config change, workload migration) and, when found with
enough post-break data to validate honestly, discards everything before
it for both fold generation and the final fit.

**This went through SEVEN design iterations before shipping, each
rejected for a concrete, tested reason -- recorded in full in the
function's docstring, not just asserted here:**
1. Full-segment means, raw values -> false-positived on any smooth trend.
2. Global OLS detrend, full-segment residual means -> fixed that, but a
   large real break distorts the global trend fit itself, mislocating
   the detected split (a break at index 40 was detected at index 8).
3. Local windowed means, fresh per-window std -> fixed localization, but
   17/30 pure-trend test series false-positived from small-sample
   variance instability.
4. Local windowed means, global noise scale, short window, no seasonal
   handling -> fixed 1-3, but 20/20 seasonal (no-break) test series were
   flagged -- a daily peak/trough looks like a level shift to a short
   window. Not a corner case: CPU/network metrics in this pipeline ARE
   seasonal.
5. Same as 4 but window widened to a full period to average out
   seasonality -> fixed the seasonal case, broke the trend case again
   (30/30 false positives) -- trend and seasonality need different
   window sizes to fool a raw comparison; no single window fixes both.
6. STL-based deseasonalize+detrend, small window on the STL residual ->
   seemed principled, but STL with only ~3-4 seasonal cycles (this
   pipeline's typical short-series regime) overfits noise INTO the
   seasonal component, leaving a residual that's near-zero almost
   everywhere except scattered LOESS-boundary artifacts -- caused false
   positives on PURE NOISE and PURE TREND (30/30 each), worse than
   several earlier attempts. Also uncovered and fixed a real bug during
   this iteration: a line-range code edit accidentally deleted the
   `_naive_persistence_forecast` function definition itself, leaving its
   body as dead code silently merged into the previous function -- valid
   Python syntax (unreachable code after a `return`), so it passed
   `ast.parse` and was only caught by explicitly testing that
   `naive_persistence` was still reachable, not by syntax checking alone.
7. **Shipped version:** single global OLS detrend (cheap, not STL, avoids
   STL's data-hunger) + ACF-based seasonality gate (abstain when the
   detrended series still shows autocorrelation > 0.3 at the given
   period) + small local window (6) on the detrended series when the gate
   doesn't fire. Passed every synthetic scenario used during development:
   pure trend (0/30 false positives), pure noise (0/30), seasonal-no-break
   (0/20, correctly abstained), real breaks alone/with trend (30/30
   correctly localized within +/-6), and real breaks on top of genuine
   seasonality (17/17 correctly localized among cases the gate didn't
   abstain on -- a real break lowers a series' own autocorrelation, so
   the gate naturally opens for many of exactly the cases it should).

**One more real regression found and fixed after that:** a break
positioned very close to the end of a series can only be detected a
little early (the scanner requires `min_segment` points on both sides),
leaving a post-break slice that cleared the bare `MIN_TRAIN_POINTS` floor
but was too thin for `select_forecast_method`'s own rolling-origin
validation to run on properly -- trimming to it anyway forced a crude
single-fit fallback that scored WORSE than letting the full untrimmed
history compete normally (where `naive_persistence` correctly won on
merit in the same scenario without break detection active). Fixed by
tightening the trim guard to require `choose_holdout_size(post_break_n) >
0` (real validation capacity), not just the bare minimum-points floor.

**HONESTY ABOUT WHAT THIS IS AND ISN'T**, same caveat as in the function's
own docstring: this is a heuristic scan verified against a real but
limited set of synthetic scenarios, not a validated change-point test
with a corrected significance level or a guarantee against every shape
real telemetry could take (weekly-on-daily seasonality, multiple breaks
in one window, gradual/ramped changes rather than sharp steps are all
untested). `backtest_structural_break_detected` / `backtest_structural_break_trimmed_n`
/ `backtest_rolling_origin_folds_used` are new fields on every backtest
result -- use them to audit which predictions this affected rather than
trusting it blindly.

## New fields (all backtests, and the live forecaster's `data_note`/`forecast_method`)

- `backtest_structural_break_detected` (bool)
- `backtest_structural_break_trimmed_n` (int -- how many older points were discarded)
- `backtest_rolling_origin_folds_used` (int)
- Live forecaster: `forecast_method` gets a `_post_break` suffix, `data_note`
  gets `structural_break_trimmed_{n}pts` appended, when a break was used.

## Still open

- Rolling-origin CV, ETS, Theta, and Croston are all wired into BOTH the
  backtest module and the live forecaster (same code, imported once) --
  but as with v9.2, this only affects the capacity/utilisation branch.
  `BURSTY_METRICS` features still use the original seasonal-naive/
  exponential-smoothing + pooling logic.
- Structural break detection's ACF gate and t-threshold are tuned against
  synthetic scenarios, not calibrated against labeled real breaks. Watch
  `backtest_structural_break_detected` rates in real data before trusting
  it heavily.
- Everything from v9.1/v9.2's "still open" sections remains open: no
  automated test suite (everything in this entry was verified via ad hoc
  scripts in a sandbox, the same way v9.1/v9.2 were -- this needs to
  become a committed pytest suite before production deployment, not stay
  a conversation transcript), no CI, no shadow-mode run against real
  outcomes yet.

