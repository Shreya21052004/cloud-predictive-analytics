"""verify_break_persistence_gate.py -- synthetic verification for the v9.4
persistence-gate fix to detect_last_structural_break().

Run with: python verify_break_persistence_gate.py

Compares the OLD detector (single-window test only, reimplemented here
verbatim from pre-v9.4 backtest.py) against the NEW detector (imported
from src.backtest, includes the persistence confirmation) across:
  1. A transient spike that REVERTS (the production failure mode this
     fix targets) -- old should false-positive, new should abstain.
  2. A genuine, persistent level shift -- both should detect it, at
     close to the true break index.
  3. Pure trend, pure noise, pure seasonal-no-break (regression checks
     from the original v9.3 test suite, described in
     CHANGELOG_v9.1_to_v9.3.md) -- new detector must not regress these.

This is a synthetic sanity check, not a replacement for re-running the
real mongosh accuracy summary against your own MongoDB data after
deploying this fix -- do that too (see README note added below).
"""
import numpy as np

from src.backtest import detect_last_structural_break as new_detect, _acf_at_lag, SEASONALITY_ACF_GATE

rng = np.random.default_rng(42)


def old_detect(x, y, min_segment=8, window=6, t_threshold=4.0, period=None):
    """Pre-v9.4 detector, reimplemented verbatim for comparison."""
    n = len(y)
    if n < 2 * min_segment:
        return None
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
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
    if best_split is not None and best_stat >= t_threshold:
        return best_split
    return None


def run_case(name, x, y, period=None, expect_old=None, expect_new=None):
    o = old_detect(x, y, period=period)
    n = new_detect(x, y, period=period)
    print(f"{name:45s} old={str(o):6s} new={str(n):6s}", end="  ")
    ok = True
    if expect_old is not None:
        ok &= (o is not None) == expect_old
    if expect_new is not None:
        ok &= (n is not None) == expect_new
    print("OK" if ok else "MISMATCH")
    return ok


results = []
n = 60
x = np.arange(n, dtype=float)

# 1. Transient spike (network burst / DDoS-like): flat baseline, spike for
#    8 points around index 40, then reverts back to baseline.
y = np.full(n, 10.0) + rng.normal(0, 0.3, n)
y[40:48] += 15.0  # spike, long enough to fill the old detector's window
results.append(run_case("1. Transient spike (reverts)", x, y,
                         expect_old=True, expect_new=False))

# 2. Genuine persistent level shift (real capacity resize) at index 40.
y = np.concatenate([np.full(40, 10.0), np.full(20, 25.0)]) + rng.normal(0, 0.3, n)
results.append(run_case("2. Genuine persistent break", x, y,
                         expect_old=True, expect_new=True))

# 3. Pure trend, no break.
y = 10.0 + 0.5 * x + rng.normal(0, 0.5, n)
results.append(run_case("3. Pure trend (no break)", x, y,
                         expect_old=False, expect_new=False))

# 4. Pure noise, no break.
y = 10.0 + rng.normal(0, 1.0, n)
results.append(run_case("4. Pure noise (no break)", x, y,
                         expect_old=False, expect_new=False))

# 5. Diurnal seasonality, no break (period=24).
n2 = 96
x2 = np.arange(n2, dtype=float)
y2 = 10.0 + 3.0 * np.sin(2 * np.pi * x2 / 24.0) + rng.normal(0, 0.3, n2)
results.append(run_case("5. Seasonal, no break (period=24)", x2, y2, period=24,
                         expect_old=False, expect_new=False))

# 6. Spike still ONGOING at the end of the training window (spike starts
#    at 46, never reverts within available data -- the genuinely
#    unresolvable case discussed in the docstring). Only 14 points remain
#    after it -- below min_post_break_points=16.
y = np.full(n, 10.0) + rng.normal(0, 0.3, n)
y[46:] += 15.0
o = old_detect(x, y)
nv = new_detect(x, y)
ok6 = (o is not None) and (nv is None)
print(f"{'6. Spike ongoing at window end (thin tail)':45s} old={str(o):6s} new={str(nv):6s}  " +
      ("OK (new abstains: too little post-break data to trust)" if ok6 else "MISMATCH"))
results.append(ok6)

# 7. Spike-then-revert where the recovered tail is also thin.
y = np.full(n, 10.0) + rng.normal(0, 0.3, n)
y[40:52] += 15.0
o = old_detect(x, y)
nv = new_detect(x, y)
print(f"{'7. Spike+revert, thin recovered tail':45s} old={str(o):6s} new={str(nv):6s}")

# 8. Real break combined with an underlying trend, with plenty of
#    post-break data -- must still be detected (not over-blocked by the
#    new gates).
y = np.concatenate([
    10.0 + 0.1 * np.arange(35),
    10.0 + 0.1 * 34 + 12.0 + 0.1 * np.arange(25),
]) + rng.normal(0, 0.3, n)
results.append(run_case("8. Break + trend, ample post-break data", x, y,
                         expect_old=True, expect_new=True))

print()
n_pass = sum(results)
print(f"{n_pass}/{len(results)} core scenarios behave as intended.")
