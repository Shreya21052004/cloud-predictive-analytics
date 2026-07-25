# v9 — Methodology upgrade: real out-of-sample validation

Everything in v8 fixed logic bugs in a confidence-scoring formula. This pass
fixes something more fundamental: the formula itself was scoring
goodness-of-fit (how well a model explains its own training data), not
forecast accuracy (how well it predicts data it hasn't seen). No amount of
bug-fixing in the composite-score weights could have closed that gap — it
required an actual holdout-validation harness. This is that harness, plus
two follow-on fixes it immediately surfaced.

## 1. [STRUCTURAL] Real train/holdout backtesting — `src/backtest.py`

Every prediction that has enough history (see `choose_holdout_size`: needs
roughly 8+ training points left over after holding out the most recent
3–14) now gets a genuine backtest: fit on all data up to a cutoff, forecast
forward across the held-out window using the *same* model logic the live
forecaster would use, and compare against what actually happened.

Two things are checked, not just the point forecast:
- **MASE** (Mean Absolute Scaled Error) — error divided by a naive
  "no-change" (or seasonal-naive) baseline computed on training data only.
  MASE < 1 means the model beat naive; MASE ≥ 1 means it didn't. Chosen
  over MAPE deliberately — MASE is the metric the M4/M5 forecasting
  competitions standardized on because it doesn't blow up near zero the way
  MAPE does, and it gives a meaningful reference point rather than an
  unanchored percentage.
- **Interval coverage** — the fraction of held-out actual values that fell
  inside the model's own predicted [lower, upper] band. This is a
  calibration check: a model whose "confidence interval" only contains the
  truth a fraction of the time its own math implies has miscalibrated
  uncertainty, independent of how good the point forecast looks.

When a backtest is available, its `backtest_quality_score` (0–100,
weighted 70% MASE / 30% coverage) **replaces** the in-sample R² as the
basis for `forecast_reliability` and the confidence composite — a real
out-of-sample result is a strictly better estimate of forecast quality than
in-sample fit. New field `evaluation_basis` records which one actually
happened for each prediction: `"out_of_sample_backtest"` or
`"in_sample_fit_only"`. **Anything business-critical should filter to the
former** — that's the whole point of adding this.

New fields on every prediction: `backtest_available`, `backtest_holdout_n`,
`backtest_mae`, `backtest_rmse`, `backtest_mase`,
`backtest_interval_coverage_pct`, `backtest_quality_score`,
`evaluation_basis`.

## 2. [FOUND BY #1] Prediction intervals only reflected slope uncertainty, not residual noise

Running the very first backtest against a clean synthetic trend exposed
this immediately: **interval coverage came back 0%**. The old
`forecast_*_lower`/`forecast_*_upper` bounds were built entirely from the
bootstrap CI on the trend line's *slope* — i.e., "how uncertain is the
angle of the line" — and never included how much any individual future
point actually wobbles around that line even when the slope is estimated
perfectly. That's the dominant source of real forecast uncertainty for
noisy data, and it was completely absent from the bounds.

**Fix:** both the live forecaster (`prediction.py`) and the backtest's own
model (`backtest.py`, so it evaluates the *corrected* model) now add a
residual-noise margin (±1.28 × residual standard deviation, an ~80%
one-sided normal quantile) on top of the slope-uncertainty band — the STL
residual's own spread when STL ran, otherwise the regression residual
around the fitted line. Coverage on the same synthetic trend went from 0%
to 64.3% after this fix, landing in the intended 60–95% calibrated range.
**This was a real, previously-invisible bug that no in-sample metric could
ever have revealed** — it only shows up once you check a model's stated
uncertainty against data it didn't get to see.

## 3. [METHODOLOGY] Seasonal-naive baseline replaces forced trend-fitting for bursty metrics

Raised directly in conversation: does forcing a linear/exponential-smoothing
trend line onto genuinely bursty, diurnal metrics (network traffic, IOPS,
request counts) even make sense? No — companies that forecast this kind of
data generally use a seasonal-naive or anomaly-baseline approach ("what
usually happens at this point in the cycle"), not a trend extrapolation,
precisely because the data isn't trending, it's cyclical.

**Fix:** `BURSTY_METRICS` with at least 2 full diurnal cycles of history
(≥48 hourly points) now use `seasonal_phase_forecast()`
(`forecast_math.py`) — bucket historical observations by hour-of-day, and
forecast each horizon as that bucket's median with a 10th/90th-percentile
band. This makes no claim about long-run drift by design, which is
deliberate: point-in-cycle expectation and structural trend are different
questions and conflating them is exactly what produced poor R² for Network
before. A separate, coarse check — `long_run_drift_direction()` — flags
genuine first-half-vs-second-half growth so real structural change (e.g.
egress creeping up over months) isn't silently thrown away.

Short bursty series (<48 points) still use the old
exponential-smoothing-plus-shrinkage approach as an explicit fallback —
there isn't enough history yet to build meaningful hour-of-day buckets.
New forecast_method value: `"seasonal_naive_baseline"`.

## 4. Shared math, not duplicated math — `src/forecast_math.py`

The bootstrap-slope-CI, STL-decomposition, and exponential-smoothing
primitives used to live only in `prediction.py`. They're now in
`forecast_math.py`, imported by both `prediction.py` (the live forecaster)
and `backtest.py` (the holdout evaluator), so the model being backtested is
provably the same model making live predictions — not a hand-maintained
second copy that can silently drift out of sync.

---

## What this means in practice

- Every prediction now honestly states whether its reliability label is
  backed by real out-of-sample validation (`evaluation_basis:
  "out_of_sample_backtest"`) or is still just an in-sample heuristic
  (`"in_sample_fit_only"`, when there isn't enough history yet). Build any
  downstream trust/alerting logic on top of `evaluation_basis` and
  `backtest_quality_score` where they're available — they're strictly more
  trustworthy than `trend_fit_r2` alone.
- Network's forecasts on bursty byte/count metrics should look
  qualitatively different now: a "what's typical for 2pm on a weekday"
  band instead of a linear ramp, which is both more honest and more useful
  for anomaly-style alerting than a false capacity-exhaustion date.
- Expect `forecast_reliability` to move around for a lot of resources on
  the next run — some up (genuinely good models that were previously
  under-scored by in-sample R² alone), some down (models that fit their
  own history well but don't generalize, which in-sample R² could never
  catch).

## What's still genuinely missing (asked directly, answered honestly)

This closes the biggest structural gap, but a few things a mature
production forecasting system would still have that this doesn't:

- **Rolling-origin cross-validation, not a single split.** This backtest
  uses one train/holdout cutoff per resource. A production system handling
  business-critical capacity decisions should walk that cutoff forward
  across several points and require consistency across them before fully
  trusting a "high" label — a single split can still be lucky or unlucky,
  especially with a short holdout window (this pipeline's holdout is
  capped at 14 points).
- **True live backtesting against real future actuals.** Everything here
  validates against history that already happened. The real test — did
  last week's forecast for today match what actually happened today — needs
  predictions to be archived and compared against actuals as they arrive.
  That requires a feedback loop this offline pipeline doesn't have by
  itself: predictions written today need to be joined against real data N
  days from now, which means a scheduled job and a place to store the
  comparison, not just this codebase.
- **Weekly seasonality.** `seasonal_phase_forecast` buckets by hour-of-day
  only. Traffic that also varies by day-of-week (weekday vs. weekend) isn't
  captured — a resource with a strong Monday-vs-Saturday difference will
  still get a single hour-of-day-only band. Worth adding a day-of-week ×
  hour-of-day bucket once there's enough history (needs several weeks of
  data per bucket to be meaningful, versus a few days for hour-of-day
  alone).
- **Drift monitoring / model staleness detection.** Nothing here tracks
  whether a resource's behavior has fundamentally shifted since its last
  backtest (e.g. a workload migration, a config change) — that requires
  comparing recent live accuracy against the backtest's implied accuracy
  over time and alerting when they diverge.
- **A human-in-the-loop review path for "high" reliability + high business
  impact.** Confidence scores, however well validated, are still a model's
  self-assessment. Real enterprise deployments usually gate the biggest
  capacity/cost decisions behind a person looking at the forecast, not an
  automated threshold alone.

None of these are code bugs to "fix" the way v8's were — they're process
and infrastructure decisions that depend on how this pipeline gets deployed
(how often it runs, where predictions get archived, who reviews what). Flagging
them here so they're a visible roadmap item, not a silent gap.
