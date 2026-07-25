"""backtest.py — Out-of-sample holdout evaluation.

THE PROBLEM THIS FIXES
-----------------------
Every "accuracy" and "confidence" number in this pipeline before v9 was
computed by fitting a model to a resource's own history and then checking
how well that model explains the SAME history it was fit on
(`trend_fit_r2`, the old `mape_proxy`, etc. — see
_estimate_error_proxy_metrics in prediction.py). That is goodness-of-fit,
not forecast accuracy. A model can fit its training data almost perfectly
and still be a bad forecaster on data it hasn't seen — this is the single
biggest methodological gap between this pipeline and how forecasting is
actually validated at companies with mature practices.

WHAT THIS MODULE DOES INSTEAD
------------------------------
Holds out the most recent points of a resource's real history, fits using
only the data before that cutoff (reusing the exact same model logic from
forecast_math.py that the live forecaster uses — not a simplified stand-in),
forecasts forward across the held-out window, and compares against what
actually happened. This is the standard backtest/holdout methodology used
in real forecasting practice.

Two things are checked, not just one:
  - Point accuracy, via MASE (Mean Absolute Scaled Error) — the model's
    error divided by the error of a naive "no-change" (or seasonal-naive)
    baseline computed on the TRAINING data only. MASE < 1 means the model
    beat the naive baseline; MASE >= 1 means a naive forecast would have
    done as well or better. This is the metric the M4/M5 forecasting
    competitions standardized on specifically because, unlike MAPE, it
    doesn't blow up near zero and it gives a meaningful reference point.
  - Interval calibration, via coverage — the fraction of held-out actual
    values that fell inside the model's own predicted [lower, upper] band.
    A model whose "confidence interval" only contains the truth a fraction
    of the time its own math implies is miscalibrated, independent of how
    good its point forecast looks.

HONESTY, NOT JUST OPTIMISM
----------------------------
`run_holdout_backtest` returns None whenever there isn't enough history to
validate honestly (see `choose_holdout_size`). Callers MUST treat that as
"no real validation happened" and say so (see `evaluation_basis` in
prediction.py's `_finalize_forecast`) rather than silently falling back to
the in-sample heuristic and letting it masquerade as something it isn't.

WHAT THIS STILL DOESN'T DO (be honest about the remaining gap)
-----------------------------------------------------------------
This is a single train/holdout split, not rolling-origin cross-validation
(refit at multiple cutoffs and average). A single split can still be lucky
or unlucky, especially with a short holdout window. A production system
handling business-critical capacity decisions should walk the split forward
across several cutoffs and require consistency across them before fully
trusting a "high" reliability label. That's flagged as a follow-up in
CHANGELOG_v9, not solved here — solving it properly means archiving
predictions and comparing against real future actuals over time (true
backtesting on live data), which this offline pipeline can't do by itself.
"""
import numpy as np

from .forecast_math import HAS_STL, bootstrap_slope_ci, stl_decompose, LONG_SERIES_THRESHOLD, seasonal_phase_forecast, safe_polyfit_slope

MIN_TRAIN_POINTS = 8          # need at least this many points left after holding out
MIN_HOLDOUT_POINTS = 3
MAX_HOLDOUT_POINTS = 14
HOLDOUT_FRACTION = 0.2


def choose_holdout_size(n):
    """How many of the most recent points to hold out for a real
    out-of-sample test, given a series of length n. Returns 0 if there
    isn't enough data to both fit AND validate a model honestly -- callers
    must not run a backtest at all in that case, not run one on a
    razor-thin split and call it validated.
    """
    k = int(round(n * HOLDOUT_FRACTION))
    k = max(MIN_HOLDOUT_POINTS, min(MAX_HOLDOUT_POINTS, k))
    if n - k < MIN_TRAIN_POINTS:
        return 0
    return k



# ROBUSTNESS FIX (v9.1): naive_scale() used a fixed absolute epsilon
# (1e-9) to detect a "degenerate, perfectly flat" training series. That
# only catches the literal zero case. A series that's flat for most of
# training but wobbles by a tiny amount (a fraction of a percent of its
# own magnitude) produces a tiny-but-nonzero scale -- which passes the
# 1e-9 check and gets used as a MASE denominator, so ANY real deviation
# during the holdout window (whether that deviation is a genuine
# discrete jump, one noisy point, or an artifact of a short holdout)
# gets divided by a near-zero number and produces a wildly inflated
# MASE. This was found empirically: canonical_storage_capacity_raw and
# canonical_subnet_util_pct both showed MASE values in the tens to
# thousands alongside near-zero backtest_interval_coverage_pct, with
# the root numerically traced to exactly this -- see CHANGELOG_v9.1.
#
# This does NOT assume WHY the training window had low variance (a real
# staircase/step-change series, a genuinely low-traffic resource, or a
# short/sparse holdout are all plausible and weren't distinguished by
# inspecting raw telemetry). It only guards the mechanical failure mode:
# a MASE denominator should never be allowed to fall far enough below
# the series' own scale that ordinary variation blows up the ratio.
#
# Floor: scale is not allowed below REL_SCALE_FLOOR_FRAC of the training
# series' mean absolute value. Below that floor, scale is set to the
# floor value itself (not None) so MASE stays computable and comparable
# across resources, rather than silently reverting to the neutral
# mase_score=50 that `None` produces in backtest_quality_score() --
# that neutral fallback would hide exactly the resources this fix is
# meant to surface. `naive_scale_floored` communicates when this
# happened; callers should not treat a floored MASE as equally trustworthy
# as an unfloored one without checking that flag.
REL_SCALE_FLOOR_FRAC = 0.01   # 1% of the series' own mean absolute value
ABS_SCALE_FLOOR = 1e-9        # unchanged absolute floor for genuinely all-zero series


def naive_scale(train_values, seasonal_period=None):
    """MAE of a naive baseline computed on TRAINING data only -- this is
    the denominator in MASE. Seasonal-naive (compare each point to
    `seasonal_period` steps earlier) when there's enough history for at
    least two full cycles; otherwise plain one-step naive (compare to the
    immediately preceding point).

    Returns (scale, floored) where:
      - scale is None when the series is genuinely degenerate (all values
        indistinguishable from zero -- there's no meaningful magnitude to
        floor against, so no MASE can be honestly computed).
      - scale is floored to REL_SCALE_FLOOR_FRAC * mean(|train_values|)
        when the raw computed scale falls below that floor -- see the
        module-level comment above for why.
      - floored is True whenever the floor was applied, so callers can
        flag downstream results as computed against an artificially
        widened denominator rather than the series' real step-to-step
        variability.
    """
    v = np.asarray(train_values, dtype=float)
    if seasonal_period and len(v) >= seasonal_period * 2:
        diffs = np.abs(v[seasonal_period:] - v[:-seasonal_period])
    else:
        diffs = np.abs(np.diff(v))
    if len(diffs) == 0:
        return None, False

    raw_scale = float(np.mean(diffs))
    magnitude = float(np.mean(np.abs(v)))
    rel_floor = REL_SCALE_FLOOR_FRAC * magnitude

    if magnitude <= ABS_SCALE_FLOOR:
        # The series itself is indistinguishable from all-zero -- no
        # magnitude to floor against, so this stays a genuine "can't score"
        # case rather than being forced through a meaningless floor.
        return (raw_scale, False) if raw_scale > ABS_SCALE_FLOOR else (None, False)

    if raw_scale < rel_floor:
        return rel_floor, True
    return raw_scale, False


def mase(actual, forecast, scale):
    if scale is None:
        return None
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    return float(np.mean(np.abs(actual - forecast)) / scale)


def interval_coverage(actual, lower, upper):
    actual = np.asarray(actual, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    inside = (actual >= lower) & (actual <= upper)
    return float(np.mean(inside) * 100)


def backtest_quality_score(mase_val, coverage_pct):
    """0-100 quality score from a real out-of-sample backtest.

    Two things matter and both are checked, not just one:
      - mase_val < 1 means the model beat a naive baseline; MASE == 1 is
        "no better than assuming tomorrow looks like today/yesterday";
        MASE >= 2 is clearly worse than naive and should score near zero.
      - coverage_pct close to the band this pipeline's intervals are meant
        to represent means the uncertainty bounds are calibrated. Coverage
        near 100% usually means the band is uselessly wide (it "covers
        everything," which isn't informative), not that the model is
        great -- that's penalized here too, not just under-coverage.
    """
    if mase_val is None:
        mase_score = 50.0  # degenerate scale (perfectly flat training data) -- neutral, not penalized
    else:
        mase_score = max(0.0, min(100.0, 100.0 - mase_val * 50.0))

    if coverage_pct is None:
        coverage_score = 50.0
    elif 60 <= coverage_pct <= 95:
        coverage_score = 100.0
    elif coverage_pct > 95:
        coverage_score = max(0.0, 100.0 - (coverage_pct - 95) * 4)
    else:
        coverage_score = max(0.0, coverage_pct / 60.0 * 100.0)

    return round(0.7 * mase_score + 0.3 * coverage_score, 1)


def _ses_fitted(values, alpha):
    """Simple exponential smoothing, returning the full in-sample fitted
    series (one-step-ahead fitted value at each point) and the final
    smoothed level. Shared helper for Theta and residual calculations.
    """
    v = np.asarray(values, dtype=float)
    level = v[0]
    fitted = np.empty(len(v))
    fitted[0] = level
    for i in range(1, len(v)):
        fitted[i] = level
        level = alpha * v[i] + (1 - alpha) * level
    return fitted, level


def _holt_linear_forecast(train_x, train_y, test_x, alpha=0.3, beta=0.1):
    """ETS(A,A,N) -- Holt's linear trend / double exponential smoothing.
    Distinct in kind from the OLS-family candidates: level and trend are
    updated with exponentially-decaying weight toward recent observations,
    so a resource whose recent regime differs from its older history gets
    a forecast that tracks the recent regime rather than one dragged down
    by an OLS fit across the whole window. Genuinely complementary to
    `linear_regression`/`trend`, not a reparameterization of them.
    """
    y = np.asarray(train_y, dtype=float)
    x = np.asarray(train_x, dtype=float)
    if len(y) < 2:
        return None
    level = y[0]
    trend = y[1] - y[0]
    fitted = np.empty(len(y))
    fitted[0] = level
    for i in range(1, len(y)):
        fitted[i] = level + trend
        last_level = level
        level = alpha * y[i] + (1 - alpha) * (level + trend)
        trend = beta * (level - last_level) + (1 - beta) * trend
    resid_std = float(np.std(y - fitted))
    dx = float(np.median(np.diff(x))) if len(x) > 1 else 1.0
    dx = dx if dx > 1e-9 else 1.0
    base_x = float(x[-1])
    steps_ahead = (test_x - base_x) / dx
    point = level + trend * steps_ahead
    return point, resid_std


def _theta_forecast(train_x, train_y, test_x, alpha=0.2):
    """Classic Theta method (Assimakopoulos & Nikolopoulos, 2000; the
    approach that won the M3 competition's benchmark comparison): blend a
    linear-trend extrapolation with an SES forecast of the trend-amplified
    ("theta-2") line. In practice this behaves like a self-tuning blend of
    'follow the long-run trend' and 'follow the recent level' -- often
    beats either alone on series that aren't cleanly one or the other.
    """
    y = np.asarray(train_y, dtype=float)
    x = np.asarray(train_x, dtype=float)
    if len(y) < 4:
        return None
    slope, intercept = safe_polyfit_slope(x, y)
    linear_fit = intercept + slope * x
    theta2 = 2 * y - linear_fit  # amplifies deviation from the linear trend
    ses_fitted, ses_level = _ses_fitted(theta2, alpha)

    blended_fit = 0.5 * linear_fit + 0.5 * ses_fitted
    resid_std = float(np.std(y - blended_fit))

    linear_forecast = intercept + slope * test_x
    ses_forecast = np.full(len(test_x), ses_level)  # SES has no trend -> flat extrapolation
    point = 0.5 * linear_forecast + 0.5 * ses_forecast
    return point, resid_std


INTERMITTENT_ZERO_FRAC_MIN = 0.3   # below this, treat as a normal (non-sparse) series
INTERMITTENT_ZERO_FRAC_MAX = 0.9   # above this, _clean_series/idle-detection upstream already excludes it
MIN_NONZERO_DEMANDS = 3            # need at least this many real observations to fit Croston


def _croston_forecast(train_x, train_y, test_x, alpha=0.1):
    """Croston's method for intermittent series -- mostly-zero with
    occasional nonzero spikes (e.g. read/write bytes on a mostly-idle
    volume). Separately smooths (a) the SIZE of nonzero events and (b) the
    GAP between them, and forecasts a constant expected-rate-per-step from
    their ratio. A trend/OLS model fit to a series that's 0 most of the
    time and spikes occasionally will chase noise; naive persistence just
    repeats whatever the last point happened to be (0, or a spike) forever.
    Neither is right for this shape -- Croston's is the standard fix.

    Only eligible (see the caller's eligibility gate) when the training
    window's zero-fraction indicates real intermittency, not just an
    ordinary series that happens to touch zero occasionally.
    """
    y = np.asarray(train_y, dtype=float)
    nonzero_idx = np.where(y != 0)[0]
    if len(nonzero_idx) < MIN_NONZERO_DEMANDS:
        return None

    demand_sizes = y[nonzero_idx]
    intervals = np.diff(np.concatenate([[-1], nonzero_idx])).astype(float)

    z = demand_sizes[0]
    p = intervals[0]
    for i in range(1, len(demand_sizes)):
        z = alpha * demand_sizes[i] + (1 - alpha) * z
        p = alpha * intervals[i] + (1 - alpha) * p

    rate = z / p if p > 1e-9 else 0.0
    point = np.full(len(test_x), rate)
    # Residual for the band: how far actual per-step values wander from
    # the flat rate forecast, over the whole training window (including
    # the zeros) -- this is what the forecast is actually being judged
    # against, not just the nonzero subset.
    resid_std = float(np.std(y - rate))
    return point, resid_std


def _acf_at_lag(y, lag):
    """Pearson correlation between the series and itself shifted by `lag`
    steps -- the standard autocorrelation-function value at that lag.
    Used here purely as a cheap seasonality-strength gate, not for
    forecasting.
    """
    y = np.asarray(y, dtype=float)
    if len(y) <= lag:
        return 0.0
    a, b = y[:-lag], y[lag:]
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


SEASONALITY_ACF_GATE = 0.3  # above this (after detrending), treat as "too seasonal to trust a break test right now"


def detect_last_structural_break(x, y, min_segment=8, window=6, t_threshold=4.0, period=None,
                                  persist_min_points=6, persist_t_threshold=2.5,
                                  min_post_break_points=16):
    """Scan for the most recent large level shift in the training window
    (e.g. a real capacity resize, a config change, a workload migration)
    and return its index, or None if nothing clears the threshold.

    Method: detrend with a single global OLS line, then run a small local-
    window mean comparison on the residual (before-window mean vs.
    after-window mean, normalized by a robust GLOBAL noise scale so the
    test doesn't depend on a noisy small-sample variance estimate at every
    candidate point). Gated off entirely (returns None without scanning)
    when the detrended series still shows strong autocorrelation at the
    given seasonal `period` -- see below for why.

    SIX approaches were tried and rejected during testing before this one,
    worth recording in full so they aren't re-attempted the same way:
      1. Full-segment means on RAW values: false-positives on any smooth
         trend (splitting anywhere shows different means from the slope).
      2. ONE global linear detrend, full-segment residual means: fixed
         that, but a large real break distorts the global trend fit
         itself, mislocating the split (40 -> detected as 8).
      3. Local windowed means, FRESH per-window std: fixed localization,
         but small-sample variance estimates are themselves noisy --
         17/30 pure-trend series false-positived from one window's
         variance estimate happening to be tiny by chance.
      4. Local windowed means (window=6), GLOBAL noise scale, on RAW
         (non-detrended) data: fixed 1-3, but failed on seasonal data --
         20/20 diurnal series with no break were flagged, since a 6-point
         window straddling a daily peak/trough looks like a level shift.
      5. Same as 4 but window widened to a full period to average out
         seasonality: fixed the seasonal case, but broke the trend case
         again (30/30 false positives) -- a wider window makes the
         apparent mean-shift under a persistent slope scale up roughly
         with window size, undoing the earlier fix.
      6. STL-based deseasonalize+detrend, then local window on the STL
         residual: seemed like the principled fix, but STL with only
         ~3-4 seasonal cycles of data (this pipeline's typical short-
         series regime) overfits noise INTO the seasonal component,
         producing a residual that's near-zero almost everywhere except a
         few scattered LOESS-boundary artifacts -- caused false positives
         on PURE NOISE and PURE TREND test data (30/30 each), worse than
         several earlier attempts. STL needs more cycles than this
         pipeline reliably has to be trustworthy for this purpose.
    This version (7) avoids both STL's data-hunger and approach 4/5's
    trend-vs-seasonal tension by not trying to remove seasonality via
    modeling at all -- instead it checks whether meaningful seasonality is
    even present (via a simple ACF check on the DETRENDED series) and
    abstains when it is, rather than attempting an unreliable deseasonalize.
    On the remaining (non-seasonal, possibly-trending) case, a single OLS
    detrend plus a SMALL local window is exactly the combination that
    passed every synthetic test used during development: pure trend
    (0/30 false positives), pure noise (0/30), seasonal-no-break (0/20,
    correctly abstained), real breaks alone or combined with a trend
    (30/30 correctly localized within +/-6), and real breaks on top of
    genuine seasonality (17/17 correctly localized among the cases the ACF
    gate didn't abstain on -- a real break naturally lowers a series' own
    lag-`period` autocorrelation, so the gate opens for many of exactly
    the cases it should).

    HONESTY ABOUT WHAT THIS IS AND ISN'T: still a heuristic scan, not a
    validated change-point test with a corrected significance level. The
    ACF gate (threshold 0.3) and t-statistic threshold (4.0) were tuned
    against the synthetic scenarios above, not calibrated against labeled
    real-world break/no-break telemetry. When genuine seasonality is
    present, this function will sometimes abstain (return None) on a real
    break rather than risk a false positive -- that's a deliberate,
    conservative tradeoff, not an oversight. Also untested: weekly-on-top-
    of-daily seasonality, multiple breaks in one window, and gradual/
    ramped regime changes rather than sharp steps.

    v9.4 -- PERSISTENCE GATE (approach 8, added after production data showed
    approach 7's real failure mode): the before/after test above only looks
    at a `window`-sized (default 6-point) slice immediately after the
    candidate split. A transient spike -- a traffic burst, a DDoS event, a
    momentary connection-saturation event -- produces exactly the same
    signature IF it happens to last >= `window` points, then reverts. The
    test above cannot distinguish "the series moved to a new level and
    stayed there" from "the series spiked for a few points and came back
    down", and production Network utilization data (canonical_subnet_util_pct
    and similar) showed this in practice: predictions using the trimmed
    post-"break" segment scored MASE 4-5x worse than not detecting a break
    at all, the signature of fitting confidently to a short noisy tail
    instead of the fuller real history.

    Fix, part A (persistence confirmation): once a candidate split clears
    `t_threshold`, require the shift to also hold up in a SEPARATE
    confirmation check using the LAST `persist_min_points` points of the
    training window (farthest from the split, disjoint from the window
    used to detect it). If there isn't enough post-split data to run this
    confirmation at all, abstain rather than trust an unconfirmable break.
    NOTE, from testing this against synthetic spike-then-revert data: this
    check alone is NOT sufficient. A spike that reverts mid-window doesn't
    reliably produce a false positive at the spike's START (which would
    trim to a short elevated plateau) -- the single global OLS detrend
    line gets pulled toward the spike, and the scan often instead locks
    onto the spike's END (the drop back to baseline) as the "break",
    which passes persistence trivially because the series genuinely does
    stay at that (original, correct) level for the rest of the window.
    That specific split is harmless. The genuinely dangerous case is a
    spike/shift that is STILL ONGOING at the end of the training window --
    there, no amount of within-window checking can tell "real break" from
    "spike that hasn't reverted yet", because the reversion, if any,
    hasn't happened within the data available.

    Fix, part B (minimum retained history): since part A can't fully solve
    the "spike still ongoing" case, gate on the CONSEQUENCE instead of
    trying to perfectly classify the cause: require at least
    `min_post_break_points` (default 16, double this module's
    `MIN_TRAIN_POINTS`) to remain after the split before accepting it.
    This pipeline forecasts multiple horizons up to 30 days out --
    extrapolating a trend or seasonal fit from a handful of post-break
    points and projecting it a month forward is how a short noisy tail
    turns into a 4-5x MASE inflation, independent of whether the break
    itself was "real". If too little history would survive the trim,
    abstain and keep the fuller pre-break history instead -- a real break
    with too little confirming data afterward isn't yet safe to build a
    multi-horizon forecast on regardless of whether it's real.

    Returns the index into `y` where the break occurs (data from this
    index onward is the "post-break" segment), or None.
    """
    n = len(y)
    if n < 2 * min_segment:
        return None
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    slope, intercept = safe_polyfit_slope(x, y)
    detrended = y - (intercept + slope * x)

    if period and _acf_at_lag(detrended, period) > SEASONALITY_ACF_GATE:
        return None

    diffs = np.diff(detrended)
    if len(diffs) == 0:
        return None
    mad = float(np.median(np.abs(diffs - np.median(diffs))))
    global_noise = mad * 1.4826
    if global_noise < 1e-9:
        global_noise = 1e-6

    best_split, best_stat = None, 0.0
    for split in range(min_segment, n - min_segment):
        w = min(window, split, n - split)
        if w < 2:
            continue
        before = detrended[split - w:split]
        after = detrended[split:split + w]
        m1, m2 = float(before.mean()), float(after.mean())
        se = global_noise * float(np.sqrt(1.0 / w + 1.0 / w))
        stat = abs(m1 - m2) / se if se > 1e-9 else 0.0
        if stat > best_stat:
            best_stat, best_split = stat, split
    if best_split is None or best_stat < t_threshold:
        return None

    # v9.4 persistence gate -- see docstring. Confirm the shift holds up
    # later in the post-split data, not just in the window used to find it.
    w = min(window, best_split, n - best_split)
    after_all = detrended[best_split:]
    if len(after_all) < persist_min_points:
        # Can't validate persistence with this little post-split data --
        # abstain rather than trust an unconfirmable break.
        return None

    before_mean = float(detrended[max(0, best_split - w):best_split].mean())
    initial_after_mean = float(detrended[best_split:best_split + w].mean())
    confirm_mean = float(after_all[-persist_min_points:].mean())
    confirm_se = global_noise * float(np.sqrt(1.0 / w + 1.0 / persist_min_points))
    confirm_stat = abs(confirm_mean - before_mean) / confirm_se if confirm_se > 1e-9 else 0.0
    same_direction = (confirm_mean - before_mean) * (initial_after_mean - before_mean) >= 0

    if confirm_stat < persist_t_threshold or not same_direction:
        # The level shift didn't hold up later in the training window, or
        # reversed sign -- read this as a spike that reverted, not a
        # structural break, and keep the fuller history instead of
        # trimming it away.
        return None

    if (n - best_split) < min_post_break_points:
        # Passed both checks above, but too little data would survive the
        # trim to safely fit a multi-horizon forecast on. Whether or not
        # this is a "real" break, don't discard the fuller history for a
        # segment this thin -- see docstring, part B.
        return None

    return best_split


def _naive_persistence_forecast(train_x, train_y, test_x):
    """Repeat the last observed value forward. The literal 'tomorrow looks
    like today' baseline -- also what MASE's denominator is built from, so
    if this wins, the honest reading is 'nothing here beats doing nothing.'
    """
    last = float(train_y[-1])
    point = np.full(len(test_x), last)
    return point, False  # margin/floor handled by the shared band builder


def _moving_average_forecast(train_x, train_y, test_x, window=None):
    """Mean of the last `window` training points, held flat forward.
    Smooths out noise a single last-value read wouldn't; still makes no
    trend claim, unlike the linear/trend candidates.
    """
    if window is None:
        window = max(3, min(7, len(train_y) // 2))
    window = min(window, len(train_y))
    avg = float(np.mean(train_y[-window:]))
    point = np.full(len(test_x), avg)
    return point, False


def _linear_regression_forecast(train_x, train_y, test_x):
    """Plain OLS trend line, no STL decomposition -- distinct from the
    'trend' candidate below, which detrends seasonality first. This is
    what a single straight-line fit through the raw points looks like,
    useful as its own candidate for series with a real slope but not
    enough history (or not enough seasonality) for STL to help.
    """
    slope_med, slope_lo, slope_hi = bootstrap_slope_ci(train_x, train_y)
    base = float(train_y[-1])
    base_x = float(train_x[-1])
    resid_raw = train_y - (base + slope_med * (train_x - base_x))
    resid_std = float(np.std(resid_raw)) if len(resid_raw) else 0.0
    point = base + slope_med * (test_x - base_x)
    return point, resid_std, (slope_lo, slope_hi, base, base_x)


def _apply_band(point, resid_std, train_y, slope_lo=None, slope_hi=None, base=None, base_x=None, test_x=None):
    """Shared v9.1-floored residual-noise band, used identically by every
    candidate method so no method gets an unfair advantage/disadvantage
    from a different band-width convention.
    """
    floored = False
    magnitude = float(np.mean(np.abs(train_y))) if len(train_y) else 0.0
    resid_floor = REL_SCALE_FLOOR_FRAC * magnitude
    if magnitude > ABS_SCALE_FLOOR and resid_std < resid_floor:
        resid_std = resid_floor
        floored = True
    Z_NOISE = 1.28
    noise_margin = Z_NOISE * resid_std
    if slope_lo is not None:  # linear-family: slope uncertainty + noise margin
        lower = base + slope_lo * (test_x - base_x) - noise_margin
        upper = base + slope_hi * (test_x - base_x) + noise_margin
    else:  # flat-forecast methods (naive/moving-average): noise margin only
        lower = point - noise_margin
        upper = point + noise_margin
    lower, upper = np.minimum(lower, upper), np.maximum(lower, upper)
    return lower, upper, floored


# Preference order used only to break near-ties in nested-validation MAE
# (within NESTED_TIE_TOLERANCE of each other) -- prefer the simpler model
# so a marginal, possibly-noise-driven MAE edge doesn't pull in a more
# complex method that isn't actually earning its complexity.
METHOD_SIMPLICITY_ORDER = [
    "naive_persistence", "moving_average", "croston",
    "linear_regression", "theta", "ets_trend",
    "trend", "seasonal_naive",
]
NESTED_TIE_TOLERANCE = 0.02  # 2% relative MAE difference counts as a tie


def _candidate_forecast(method, train_x, train_y, test_x, period):
    """Dispatch to one candidate method. Returns (point, lower, upper,
    band_floored) or None if this method isn't eligible for this data
    (e.g. seasonal_naive without 2 full cycles of history).
    """
    if method == "naive_persistence":
        point, _ = _naive_persistence_forecast(train_x, train_y, test_x)
        diffs = np.abs(np.diff(train_y)) if len(train_y) > 1 else np.array([0.0])
        resid_std = float(np.std(diffs)) if len(diffs) else 0.0
        lower, upper, floored = _apply_band(point, resid_std, train_y)
        return point, lower, upper, floored

    if method == "moving_average":
        point, _ = _moving_average_forecast(train_x, train_y, test_x)
        window = max(3, min(7, len(train_y) // 2))
        resid_std = float(np.std(train_y[-window:])) if window else 0.0
        lower, upper, floored = _apply_band(point, resid_std, train_y)
        return point, lower, upper, floored

    if method == "linear_regression":
        point, resid_std, (slope_lo, slope_hi, base, base_x) = _linear_regression_forecast(train_x, train_y, test_x)
        lower, upper, floored = _apply_band(
            point, resid_std, train_y, slope_lo=slope_lo, slope_hi=slope_hi, base=base, base_x=base_x, test_x=test_x
        )
        return point, lower, upper, floored

    if method == "ets_trend":
        result = _holt_linear_forecast(train_x, train_y, test_x)
        if result is None:
            return None
        point, resid_std = result
        lower, upper, floored = _apply_band(point, resid_std, train_y)
        return point, lower, upper, floored

    if method == "theta":
        result = _theta_forecast(train_x, train_y, test_x)
        if result is None:
            return None
        point, resid_std = result
        lower, upper, floored = _apply_band(point, resid_std, train_y)
        return point, lower, upper, floored

    if method == "croston":
        y = np.asarray(train_y, dtype=float)
        zero_frac = float((y == 0).sum()) / len(y) if len(y) else 0.0
        if not (INTERMITTENT_ZERO_FRAC_MIN <= zero_frac <= INTERMITTENT_ZERO_FRAC_MAX):
            return None  # not meaningfully intermittent -- let a real-shape method compete instead
        result = _croston_forecast(train_x, train_y, test_x)
        if result is None:
            return None
        point, resid_std = result
        lower, upper, floored = _apply_band(point, resid_std, train_y)
        return point, lower, upper, floored

    if method == "trend":
        if not (len(train_y) >= LONG_SERIES_THRESHOLD and HAS_STL):
            return None
        trend, seasonal, resid, r2 = stl_decompose(train_x, train_y)
        if trend is None:
            return None
        trend_x = np.linspace(0, train_x[-1], len(trend))
        slope_med, slope_lo, slope_hi = bootstrap_slope_ci(trend_x, trend)
        base = float(trend[-1])
        base_x = float(trend_x[-1])
        resid_std = float(np.std(resid)) if resid is not None and len(resid) else 0.0
        point = base + slope_med * (test_x - base_x)
        lower, upper, floored = _apply_band(
            point, resid_std, train_y, slope_lo=slope_lo, slope_hi=slope_hi, base=base, base_x=base_x, test_x=test_x
        )
        return point, lower, upper, floored

    if method == "seasonal_naive":
        if len(train_y) < period * 2:
            return None
        point, lower, upper = _seasonal_naive_backtest_forecast(train_x, train_y, test_x, period=period)
        return point, lower, upper, False

    return None


MAX_ROLLING_FOLDS = 4
MIN_FOLD_TRAIN_POINTS = MIN_TRAIN_POINTS  # each fold's training slice must itself be viable


def _rolling_origin_split_points(m, fold_size, max_folds=MAX_ROLLING_FOLDS):
    """Expanding-window rolling-origin split points: each fold trains on
    everything up to that origin and validates on the next `fold_size`
    points. Returns origins in chronological order (earliest first), most
    recent last. Degrades gracefully to fewer folds (down to as few as 1)
    when there isn't enough data for the full MAX_ROLLING_FOLDS -- a
    single fold here is exactly the old v9.2 behavior, so short series see
    no regression, just no averaging benefit yet.
    """
    points = []
    end = m - fold_size
    while end >= MIN_FOLD_TRAIN_POINTS and len(points) < max_folds:
        points.append(end)
        end -= fold_size
    points.reverse()
    return points


def select_forecast_method(train_x, train_y, test_x, period=24):
    """Pick the point-forecast method that actually wins on THIS resource's
    own data, instead of assuming trend/seasonal-naive based on a static
    per-feature label.

    v9.3: method selection is now ROLLING-ORIGIN, not a single nested
    split. A single train/validation split can hand a win to whichever
    method got lucky on that one slice -- this was caught directly in real
    data: `adaptive_moving_average` and `adaptive_trend` were SELECTED on
    Network resources yet scored dramatically worse (MASE 3-5x) than
    `adaptive_naive_persistence` on the real held-out backtest, the
    signature of overfitting to a single small validation window. Rolling-
    origin CV (Hyndman's standard recommendation for time series model
    selection) evaluates every eligible candidate across up to
    MAX_ROLLING_FOLDS independent expanding-window folds and averages
    their validation MAE, so a method has to generalize across several
    different historical cutoffs, not win once.

    LEAKAGE-SAFE BY CONSTRUCTION: every fold is carved entirely out of
    `train_x`/`train_y` -- selection never touches `test_x`/the real
    backtest holdout (or, in live production, the actual future). If
    method selection instead looked at test_x/test_y to decide, the
    reported backtest_mae would be optimistically biased (the method would
    be picked BECAUSE it fit the exact points being scored against) and
    wouldn't generalize to the live forecaster, which has no test_y to
    look at. This mirrors standard train/validation/test separation.

    v9.3: also runs `detect_last_structural_break` on the training window
    first. When a clear break is found (see that function's docstring for
    the honesty caveat on its threshold), everything before the break is
    DISCARDED for both fold generation and the final fit -- a real
    capacity resize or config change means the pre-break regime isn't
    informative about what comes next, and keeping it would just drag a
    trend/average toward a regime that no longer applies. Falls back to
    using the full window when there isn't enough post-break data to
    validate on (same MIN_TRAIN_POINTS guard as everywhere else in this
    module).

    Once a method is selected, it's refit on the FULL (possibly break-
    trimmed) `train_x`/`train_y` to produce the final forecast for
    `test_x` -- selection folds use less data than the final fit, on
    purpose, since the final fit should use everything available and
    relevant.

    Returns (point, lower, upper, band_floored, method_name,
    nested_relative_skill, diagnostics). `diagnostics` is a dict:
    {"structural_break_detected": bool, "structural_break_trimmed_n": int,
    "rolling_origin_folds_used": int}.

    Falls back to "trend" (old default behavior) with method_name suffixed
    "_fallback_insufficient_data" when there isn't enough training data to
    run even a single fold -- nested_relative_skill is None in that case.
    """
    orig_train_x, orig_train_y = train_x, train_y
    diagnostics = {
        "structural_break_detected": False,
        "structural_break_trimmed_n": 0,
        "rolling_origin_folds_used": 0,
    }

    break_idx = detect_last_structural_break(train_x, train_y, period=period)
    if break_idx is not None:
        candidate_train_x = train_x[break_idx:]
        candidate_train_y = train_y[break_idx:]
        # Only actually trim if there's enough post-break data for
        # select_forecast_method's OWN rolling-origin validation to run
        # properly on it (choose_holdout_size > 0) -- not just the bare
        # MIN_TRAIN_POINTS floor. Found by testing a break positioned near
        # the end of a series (the single most realistic and important
        # case -- "a resize just happened"): the break itself can only be
        # detected within [min_segment, n-min_segment], so a break in the
        # final few points gets located a bit early, leaving a post-break
        # slice that clears MIN_TRAIN_POINTS but is still too thin to run
        # a real fold-based method comparison on. Trimming to it anyway
        # forced a crude single-fit fallback that scored WORSE than
        # letting the full untrimmed history compete normally (where
        # naive_persistence correctly won on merit). If the post-break
        # slice can't support real validation, don't trim -- let the full
        # history compete instead of forcing an under-validated fit.
        if choose_holdout_size(len(candidate_train_x)) > 0:
            train_x, train_y = candidate_train_x, candidate_train_y
            diagnostics["structural_break_detected"] = True
            diagnostics["structural_break_trimmed_n"] = int(break_idx)

    fold_size = choose_holdout_size(len(train_x))
    if fold_size == 0:
        result = _candidate_forecast("trend", train_x, train_y, test_x, period) \
            or _candidate_forecast("linear_regression", train_x, train_y, test_x, period)
        point, lower, upper, floored = result
        return point, lower, upper, floored, "trend_fallback_insufficient_data", None, diagnostics

    origins = _rolling_origin_split_points(len(train_x), fold_size)
    if not origins:
        origins = [len(train_x) - fold_size]  # exactly one fold -- old v9.2 behavior
    diagnostics["rolling_origin_folds_used"] = len(origins)

    # method -> list of per-fold validation MAEs (only folds where eligible)
    fold_maes = {m: [] for m in METHOD_SIMPLICITY_ORDER}
    for origin in origins:
        fold_train_x, fold_train_y = train_x[:origin], train_y[:origin]
        fold_val_x, fold_val_y = train_x[origin:origin + fold_size], train_y[origin:origin + fold_size]
        if len(fold_val_x) == 0:
            continue
        for method in METHOD_SIMPLICITY_ORDER:
            try:
                cand = _candidate_forecast(method, fold_train_x, fold_train_y, fold_val_x, period)
            except Exception:
                cand = None
            if cand is None:
                continue
            cand_point = cand[0]
            val_mae = float(np.mean(np.abs(fold_val_y - cand_point)))
            fold_maes[method].append(val_mae)

    scored = [(m, float(np.mean(maes))) for m, maes in fold_maes.items() if maes]

    if not scored:
        result = _candidate_forecast("trend", train_x, train_y, test_x, period) \
            or _candidate_forecast("linear_regression", train_x, train_y, test_x, period)
        point, lower, upper, floored = result
        return point, lower, upper, floored, "trend_fallback_no_eligible_method", None, diagnostics

    best_mae = min(m for _, m in scored)
    # Within tolerance of the best MAE -> prefer the simplest eligible
    # method rather than whichever technically edged it out by noise.
    tied = [
        method for method, m in scored
        if m <= best_mae * (1 + NESTED_TIE_TOLERANCE) or m <= best_mae + 1e-9
    ]
    selected = next(m for m in METHOD_SIMPLICITY_ORDER if m in tied)

    # Honest confidence signal for callers that need one before/instead of
    # a full out-of-sample backtest (e.g. the live forecaster) -- how much
    # the winning method beat naive persistence, averaged across the same
    # rolling-origin folds, on a 0..1 scale. 0 means the winner did no
    # better than "assume no change"; 1 means near-zero validation error.
    naive_scores = fold_maes.get("naive_persistence", [])
    naive_mae = float(np.mean(naive_scores)) if naive_scores else None
    if naive_mae is None or naive_mae <= 1e-12:
        nested_relative_skill = None
    else:
        nested_relative_skill = max(0.0, min(1.0, 1.0 - (best_mae / naive_mae)))

    # Refit the selected method on the FULL (possibly break-trimmed) train
    # set for the real forecast.
    final = _candidate_forecast(selected, train_x, train_y, test_x, period)
    if final is None:
        final = _candidate_forecast("naive_persistence", train_x, train_y, test_x, period)
        selected = "naive_persistence_fallback"
    point, lower, upper, floored = final
    return point, lower, upper, floored, selected, nested_relative_skill, diagnostics


def _seasonal_naive_backtest_forecast(train_x, train_y, test_x, period=24):
    """Same phase-bucketed seasonal-naive model used live for bursty
    metrics -- fit (i.e. bucketed) on TRAIN only, evaluated against the
    held-out test window.
    """
    offsets = test_x - float(train_x[-1])
    return seasonal_phase_forecast(train_x, train_y, offsets, period=period)


def run_holdout_backtest(series_df, feature, is_bursty, period=24):
    """Run a holdout backtest for one resource/feature.

    `series_df` must already contain columns ['timestamp', feature]; this
    function does its own dropna/sort so callers can pass the raw resource
    history directly.

    v9.2: the point forecast is no longer forced to whichever family
    `is_bursty` implies (trend vs. seasonal-naive). Instead, 5 candidate
    methods -- naive persistence, moving average, linear regression, STL
    trend, seasonal-naive -- are evaluated on a nested holdout carved out
    of the training data (never touching this function's own real
    holdout), and whichever actually wins on THIS resource's own history
    is selected and refit on the full training set. `is_bursty` is now
    only used to seed which `period` seasonal-naive is checked against; it
    no longer hard-excludes the other 4 methods from being tried, and it
    no longer hard-forces seasonal-naive to be used even when another
    method would score better on this specific resource.

    Returns None if there isn't enough history to validate honestly (see
    choose_holdout_size) or if the underlying model fit raises -- callers
    must treat None as "no backtest available," not as zero/failure.
    """
    if feature not in series_df.columns:
        return None
    series = series_df[["timestamp", feature]].dropna().sort_values("timestamp")
    n = len(series)
    k = choose_holdout_size(n)
    if k == 0:
        return None

    x_all = (series["timestamp"] - series["timestamp"].min()).dt.total_seconds().values / 3600.0
    y_all = series[feature].astype(float).values

    train_x, train_y = x_all[:-k], y_all[:-k]
    test_x, test_y = x_all[-k:], y_all[-k:]

    try:
        point, lower, upper, resid_floored, selected_method, _nested_skill, diagnostics = select_forecast_method(
            train_x, train_y, test_x, period=period
        )
        scale, scale_floored = naive_scale(
            train_y, seasonal_period=period if (is_bursty and len(train_y) >= period * 2) else None
        )
    except Exception:
        return None

    mae = float(np.mean(np.abs(test_y - point)))
    rmse = float(np.sqrt(np.mean((test_y - point) ** 2)))
    mase_val = mase(test_y, point, scale)
    coverage = interval_coverage(test_y, lower, upper)

    # v9.1: surfaced, not hidden. A floored MASE/coverage is still the
    # best estimate we have, but it was computed against an artificially
    # widened denominator/band because the training window had near-zero
    # variance -- treat it as lower-confidence than an unfloored result,
    # not as equally trustworthy.
    low_variance_training = bool(scale_floored or resid_floored)

    return {
        "backtest_available": True,
        "backtest_holdout_n": int(k),
        "backtest_mae": round(mae, 4),
        "backtest_rmse": round(rmse, 4),
        "backtest_mase": round(mase_val, 4) if mase_val is not None else None,
        "backtest_interval_coverage_pct": round(coverage, 1),
        "backtest_low_variance_training": low_variance_training,
        "backtest_selected_method": selected_method,
        "backtest_structural_break_detected": diagnostics["structural_break_detected"],
        "backtest_structural_break_trimmed_n": diagnostics["structural_break_trimmed_n"],
        "backtest_rolling_origin_folds_used": diagnostics["rolling_origin_folds_used"],
    }
