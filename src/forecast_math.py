"""forecast_math.py — Pure forecasting math primitives.

Extracted out of prediction.py (v9) so that the live forecaster
(behavioral_forecast) and the holdout backtester (backtest.py) share the
exact same model logic instead of two hand-maintained copies drifting apart
over time. Nothing in this file talks to Mongo, a DataFrame's full schema,
or a resource document — it only knows about arrays of numbers.
"""
import numpy as np
import pandas as pd
import warnings

try:
    from statsmodels.tsa.seasonal import STL
    HAS_STL = True
except ImportError:
    HAS_STL = False

MIN_POINTS_FOR_FORECAST = 5
LONG_SERIES_THRESHOLD = 72      # >= 3 days of hourly data -> STL decomposition
N_BOOTSTRAP = 200                # bootstrap resamples for slope CI


def safe_polyfit_slope(x, y):
    """Degree-1 np.polyfit that never emits a RankWarning / divide-by-zero
    and never returns a non-finite slope.

    np.polyfit builds a Vandermonde matrix from x and normalizes each
    column by its scale; when x has (near-)zero variance -- all points at
    the same timestamp, a single-point series, or (for bootstrap
    resampling) an unlucky resample that happens to redraw the same point
    repeatedly -- that scale is ~0, which triggers "RankWarning: Polyfit
    may be poorly conditioned" and a "RuntimeWarning: invalid value
    encountered in divide", and can silently return a huge or NaN/inf
    slope. A slope genuinely isn't identifiable in that case (there's no
    x-direction to fit against), so this returns a flat fallback --
    (0.0, mean(y)) -- instead of letting a degenerate fit produce a
    fictitious number.

    Returns (slope, intercept).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or len(y) < 2 or np.ptp(x) < 1e-9:
        return 0.0, (float(np.mean(y)) if len(y) else 0.0)
    try:
        with warnings.catch_warnings():
            # np.RankWarning moved to numpy.exceptions.RankWarning in
            # numpy>=2.0 and was removed from the top-level namespace; a
            # blanket ignore here is safe since this block only ever calls
            # polyfit, and we already validate finiteness below regardless
            # of whether anything was actually suppressed.
            warnings.simplefilter("ignore")
            with np.errstate(all="ignore"):
                slope, intercept = np.polyfit(x, y, 1)
        if not (np.isfinite(slope) and np.isfinite(intercept)):
            return 0.0, float(np.mean(y))
        return float(slope), float(intercept)
    except Exception:
        return 0.0, float(np.mean(y))


def exp_smooth_forecast(y_vals, alpha=0.3):
    s = float(y_vals[0])
    for v in y_vals[1:]:
        s = alpha * float(v) + (1 - alpha) * s
    return s


def bootstrap_slope_ci(x, y, n_boot=N_BOOTSTRAP, ci=90):
    """Bootstrap confidence interval on the OLS slope.

    Returns (slope_median, slope_lo, slope_hi) — all in raw y-units per hour.

    Resampling with replacement can occasionally draw a set of indices that
    all land on the same (or near-identical) x value -- e.g. duplicate
    timestamps, or plain bad luck on a small n. That resample has no real
    x-variance, so a slope isn't identifiable from it; including it would
    mean either a RankWarning-poisoned garbage slope or (with
    safe_polyfit_slope) a fabricated 0.0 diluting the real distribution.
    Either way it doesn't belong in the bootstrap sample, so it's skipped
    rather than counted.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)

    if n < 2 or np.ptp(x) < 1e-9:
        # No usable x-variance at all in the full series -- no slope is
        # estimable, flat is the honest answer.
        return 0.0, 0.0, 0.0

    if n < 4:
        slope, _ = safe_polyfit_slope(x, y)
        return float(slope), float(slope), float(slope)

    slopes = []
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xs, ys = x[idx], y[idx]
        if np.ptp(xs) < 1e-9:
            continue  # degenerate resample, see docstring
        slope, _ = safe_polyfit_slope(xs, ys)
        slopes.append(slope)

    if not slopes:
        # Every resample happened to be degenerate (can occur with very
        # small n and repeated x values) -- fall back to a single fit on
        # the full series, which we already know has usable variance.
        slope, _ = safe_polyfit_slope(x, y)
        return float(slope), float(slope), float(slope)

    lo = float(np.percentile(slopes, (100 - ci) / 2))
    hi = float(np.percentile(slopes, 100 - (100 - ci) / 2))
    med = float(np.median(slopes))
    return med, lo, hi


def stl_decompose(x_hours, y, period=24):
    """STL decomposition. Returns (trend_component, seasonal_component, residual, r2).

    r2 measures how much of the series' total variance is explained by
    trend + seasonal combined (residual = original - trend - seasonal), so
    a resource with a strong, low-noise repeating pattern (e.g. a clean
    diurnal cycle) scores high here even with zero long-term drift.

    Falls back to (None, None, None, None) if STL is unavailable, fails, or
    there isn't enough history for at least 2 full periods.
    """
    if not HAS_STL or len(y) < period * 2:
        return None, None, None, None

    try:
        series = pd.Series(y, index=pd.to_timedelta(x_hours, unit="h"))
        series = series[~series.index.duplicated()].resample("1h").mean().interpolate("linear")
        stl = STL(series, period=period, robust=True)
        res = stl.fit()
        trend = np.array(res.trend)
        seasonal = np.array(res.seasonal)
        residual = np.array(res.resid)

        ss_tot = float(np.var(series.values))
        ss_res = float(np.var(residual))
        if ss_tot < 1e-9:
            r2 = 1.0
        else:
            r2 = max(0.0, min(1.0, 1.0 - ss_res / ss_tot))
        return trend, seasonal, residual, r2
    except Exception:
        return None, None, None, None


def seasonal_phase_forecast(x_hours, y_vals, forecast_offsets_hours, period=24, lo_pct=10, hi_pct=90):
    """Seasonal-naive baseline: forecast future values by phase-bucketing
    historical observations at the same point in a repeating cycle
    (default: hour-of-day) and taking that bucket's median/percentile band.

    This is the standard approach forecasting teams use for bursty,
    non-trending metrics (network traffic, IOPS, request counts) instead of
    forcing a trend line onto data that was never trending in the first
    place — see behavioral_forecast()'s BURSTY_METRICS branch and
    CHANGELOG_v9 for the full rationale. It makes no claim about future
    drift; it only says "expect roughly what usually happens at this point
    in the cycle," which is what a naive-but-honest network traffic
    forecast actually looks like.

    Default band is the 10th-90th percentile of each phase bucket
    (nominal ~80%, matching the Z_NOISE=1.28 margin used by the
    trend-based forecaster elsewhere) — note this is a genuine percentile
    of whatever history lands in that bucket, so with only a handful of
    historical cycles (few points per bucket) the percentile estimate
    itself is noisy; the real calibration check for any given resource is
    still the out-of-sample backtest_interval_coverage_pct, not this
    default choice.

    `x_hours` are hours-from-series-start for the historical data;
    `forecast_offsets_hours` are hours-from-the-LAST historical point to
    forecast (e.g. [1, 6, 24, 168, 720]).

    Returns (point, lower, upper) arrays aligned with forecast_offsets_hours.
    """
    x_hours = np.asarray(x_hours, dtype=float)
    y_vals = np.asarray(y_vals, dtype=float)
    phase = np.mod(np.round(x_hours).astype(int), period)
    buckets = {p: y_vals[phase == p] for p in range(period)}
    global_med = float(np.median(y_vals)) if len(y_vals) else 0.0
    last_x = float(x_hours[-1]) if len(x_hours) else 0.0

    def _for_phase(p):
        vals = buckets.get(int(p) % period)
        if vals is not None and len(vals) >= 2:
            return (float(np.median(vals)),
                    float(np.percentile(vals, lo_pct)),
                    float(np.percentile(vals, hi_pct)))
        return global_med, global_med, global_med

    point, lower, upper = [], [], []
    for off in forecast_offsets_hours:
        target_x = last_x + off
        p = int(round(target_x)) % period
        m, l, u = _for_phase(p)
        point.append(m)
        lower.append(min(l, u))
        upper.append(max(l, u))
    return np.array(point), np.array(lower), np.array(upper)


def long_run_drift_direction(y_vals, rel_threshold=0.2):
    """A bursty metric can still have a genuine long-run trend underneath
    the noise (e.g. egress bytes creeping up month over month even though
    hour-to-hour is spiky). The seasonal-naive forecast above deliberately
    doesn't model drift — this is a cheap, honest secondary check so that
    kind of structural growth doesn't get silently thrown away: compare the
    median of the first half of the series against the second half. This is
    intentionally simple (no significance test) — it's a coarse flag for
    "worth a human look," not a precise slope estimate.
    """
    y_vals = np.asarray(y_vals, dtype=float)
    n = len(y_vals)
    if n < 8:
        return "stable"
    half = n // 2
    first_med = float(np.median(y_vals[:half]))
    second_med = float(np.median(y_vals[half:]))
    denom = max(abs(first_med), 1e-6)
    change = (second_med - first_med) / denom
    if change > rel_threshold:
        return "increasing"
    if change < -rel_threshold:
        return "decreasing"
    return "stable"
