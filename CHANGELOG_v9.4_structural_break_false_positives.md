# v9.4 — Structural break false-positive fix (Network utilization metrics)

## Background

Aggregate accuracy review of live `22prediction_logs_network` output showed
`adaptive_trend_post_break` (n=136, MASE 4.18) and
`adaptive_naive_persistence_post_break` (n=44, MASE 5.49) scoring far worse
than the SAME base methods without a detected break
(`adaptive_naive_persistence`, n=97, MASE 0.62). Since these branch off the
same candidate-method logic, the difference had to be coming from
`detect_last_structural_break()` and/or how its output was used to trim
training data — not from the forecasting methods themselves.

`BURSTY_METRICS` (net_in/out bytes, active_connections, etc.) don't go
through this path at all, so the affected predictions are Network's
percentage/utilization features (subnet utilization, DDoS probability,
load-balancer saturation) — the same feature class flagged in the v9.1
changelog (`canonical_subnet_util_pct`, MASE mean 10.9 pre-v9.1) as prone
to pathological scores.

## Root cause (confirmed via synthetic reproduction, not just inferred)

`detect_last_structural_break()`'s only check was: does a `window`-sized
(6-point) slice immediately after a candidate split differ from the slice
immediately before it, by more than `t_threshold` (4.0) standard errors.

This cannot distinguish a genuine level shift (capacity resize, config
change) from a transient spike — a burst of DDoS-like traffic, a
connection-saturation event — that is still elevated when the training
window ends. In that case there is no reversion visible in the available
data at all, so no within-window statistical test can rule it out; the
detector fires, discards most of the resource's history, and fits a
multi-horizon forecast (out to 30 days) on a handful of noisy post-"break"
points. See `verify_break_persistence_gate.py` for the synthetic
reproduction (scenario 6: spike ongoing at window end — old detector fires
at a split leaving only 14 points of history, new detector correctly
abstains).

## Fix — `src/backtest.py`, `detect_last_structural_break()`

Two new gates, both must pass in addition to the existing `t_threshold`
check:

1. **Persistence confirmation**: the shift must also hold up in a second,
   disjoint check using the LAST `persist_min_points` (default 6) points
   of the training window, against its own (lower) `persist_t_threshold`
   (default 2.5), in the same direction as the initial detection. Catches
   the case where a spike fully reverts before the training window ends.
2. **Minimum retained history**: even after passing (1), require at least
   `min_post_break_points` (default 16 — double this module's
   `MIN_TRAIN_POINTS`) to remain after the split. A break this pipeline
   can't leave at least 16 points to fit on isn't safe to build a 30-day
   forecast on, whether or not it's "real" — this is the gate that
   actually catches the still-ongoing-spike case that (1) alone cannot,
   since there's nothing after the split to disagree with.

If either gate fails, `detect_last_structural_break()` returns `None` and
the caller falls back to using the resource's full history, same as if no
break had been scanned for.

**Both new parameters are keyword args with defaults** — no call sites in
`backtest.py` or `prediction.py` needed to change; existing callers get
the fix automatically.

## Verification performed

`verify_break_persistence_gate.py` (included in this delivery) reimplements
the pre-v9.4 detector verbatim for side-by-side comparison and runs 8
synthetic scenarios:

| # | Scenario | Old | New | Expected |
|---|---|---|---|---|
| 1 | Transient spike, fully reverts | detects (false positive) | abstains | new correct |
| 2 | Genuine persistent break | detects | detects | both correct |
| 3 | Pure trend, no break | abstains | abstains | both correct |
| 4 | Pure noise, no break | abstains | abstains | both correct |
| 5 | Diurnal seasonality, no break | abstains | abstains | both correct |
| 6 | Spike still ongoing at window end (thin tail) | detects (false positive) | abstains | new correct |
| 7 | Spike + revert, thin recovered tail | detects | abstains | new correct |
| 8 | Genuine break + trend, ample post-break data | detects | detects | both correct (no over-blocking) |

7/7 scored scenarios (7 is a bonus, unscored sanity check) behave as
intended. Cases 3–5 confirm the fix does not regress the original v9.3
false-positive guarantees on non-break data.

## What this does NOT claim

This was validated on synthetic data with a known ground truth, the same
methodology the original detector's docstring uses for its own six
documented design iterations. It has **not** been validated against your
actual Network utilization telemetry. The honest next step is: re-run this
pipeline against real data and re-run the same mongosh accuracy summary
used before — specifically watch `adaptive_trend_post_break` and
`adaptive_naive_persistence_post_break`'s `n` (should drop, since fewer
predictions will qualify as post-break) and `mase_avg` for those methods
(should improve, since the ones that remain are now confirmed rather than
speculative). If `n` drops to near-zero for `_post_break` methods on
Network, that's expected and correct, not a regression — it means the
detector is (correctly) abstaining on data it can't validate rather than
manufacturing a false signal.

## What this does NOT fix

- Storage's reliance on the v9.1 low-variance floor (362/569 predictions,
  64%) is a data-thinness issue, not a break-detection issue — this
  change doesn't touch it. `forecast_reliability` is already correctly
  capped at "moderate" for those; downstream consumers filtering on
  `backtest_quality_score >= 80` alone should filter on
  `backtest_low_variance_training == false` too if they want the honest
  subset.
- Compute's `flat_signal` bucket (52/123) is a legitimate result for
  genuinely idle/constant resources, not a bug — `backtest_mase` is
  correctly null there because `naive_scale()` can't score a series with
  no real variance, not because backtesting failed to run.
- Databases (n=27) and Container (n=6) don't have enough real backtests
  to diagnose or fix from aggregate review — more resources reporting
  history is the only real lever there, not a code change.
