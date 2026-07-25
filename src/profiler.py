"""profiler.py — Per-resource-type (pooled) baseline profiling.

v7 change: profiles are no longer built per resource_id. A resource is never
trained/profiled on its own history in isolation. Instead, one profile is
built per (canonical_resource_type, category) — e.g. every EC2 + Compute_Engine
+ Virtual_Machines resource (all mapped to canonical type `vm_instance`) is
pooled into a single `vm_instance` profile. Individual resources only ever
*look up* their type's profile at prediction time; they never contribute a
profile that only reflects their own behaviour.

The profile captures, pooled across every resource of that canonical type:
  - Per-metric mean, std, p5, p95 (over all pooled rows)
  - Diurnal pattern: median per (day_of_week, hour_of_day) bucket — 168 values,
    computed across all resources of the type so a thin/new resource still
    gets a sensible expectation for "3pm on a Tuesday"
  - Reporting cadence: expected interval between non-null values, pooled
  - Regime change detection: sustained mean shift (pooled, across the type)
    resets the baseline

The profile is the foundation for all anomaly detection. Residuals (a
resource's current value minus its TYPE's diurnal expectation) are what get
anomaly-scored — not raw values compared to a global population distribution,
and not a baseline fit to that one resource's own history.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .metric_registry import canonical_features

WARMUP_DAYS = 14           # minimum pooled days before a type profile is trusted
PROFILE_WINDOW_DAYS = 30   # rolling window for mean/std
REGIME_SHIFT_DAYS = 3      # consecutive days of shifted mean → regime change
REGIME_SHIFT_SIGMA = 2.5   # how many sigmas counts as a shift


def _now() -> datetime:
    return datetime.now(timezone.utc)


def build_profile(canonical_resource_type: str, category: str, history: pd.DataFrame) -> Dict:
    """Build a pooled (canonical_resource_type, category) profile.

    `history` must contain rows from EVERY resource of this canonical type
    (not a single resource) — columns: timestamp (tz-aware), resource_id,
    plus canonical feature cols. Returns a dict suitable for upsert into a
    resource-type-profiles collection.
    """
    features = [f for f in canonical_features(category) if f in history.columns]
    h = history.sort_values("timestamp").copy()
    h["timestamp"] = pd.to_datetime(h["timestamp"], utc=True)

    n_days = (h["timestamp"].max() - h["timestamp"].min()).days if len(h) > 1 else 0
    n_resources_pooled = int(h["resource_id"].nunique()) if "resource_id" in h.columns else None
    # Confidence blends how long we've observed the type AND how many distinct
    # resources contributed — a type profile built from 1 resource over 14
    # days is not as trustworthy as one built from 50 resources over 14 days,
    # even though n_days is identical, so pool breadth counts too.
    day_confidence = min(1.0, n_days / WARMUP_DAYS)
    breadth_confidence = min(1.0, (n_resources_pooled or 1) / 5.0)
    profile_confidence = round(0.6 * day_confidence + 0.4 * breadth_confidence, 3)

    metrics: Dict[str, Dict] = {}
    for feat in features:
        col = h[feat].dropna()
        if len(col) < 5:
            continue

        vals = col.values.astype(float)
        mean = float(np.mean(vals))
        std = float(np.std(vals)) if len(vals) > 1 else 0.0
        p5 = float(np.percentile(vals, 5))
        p95 = float(np.percentile(vals, 95))

        # Diurnal pattern: median value per (day_of_week × hour_of_day) bucket,
        # pooled across every resource of this canonical type.
        ts_col = h.loc[col.index, "timestamp"]
        dow = ts_col.dt.dayofweek.values   # 0=Mon…6=Sun
        hod = ts_col.dt.hour.values        # 0-23
        diurnal = {}
        for d in range(7):
            for hh in range(24):
                mask = (dow == d) & (hod == hh)
                bucket_vals = vals[mask]
                if len(bucket_vals) >= 2:
                    diurnal[f"{d}_{hh}"] = float(np.median(bucket_vals))

        # Reporting cadence: typical gap between consecutive non-null points
        # (hours), pooled per-resource then aggregated so a mix of fast- and
        # slow-reporting resources within the same type doesn't collapse into
        # a single misleading number.
        expected_interval_h = None
        if "resource_id" in h.columns:
            per_resource_gaps = []
            for _, grp in h.loc[col.index].assign(_ts=ts_col).groupby(h.loc[col.index, "resource_id"]):
                g_ts = grp["_ts"].sort_values()
                if len(g_ts) < 2:
                    continue
                gaps_h = g_ts.diff().dropna().dt.total_seconds() / 3600.0
                gaps_h = gaps_h[gaps_h > 0]
                if not gaps_h.empty:
                    per_resource_gaps.append(float(gaps_h.median()))
            if per_resource_gaps:
                expected_interval_h = float(np.median(per_resource_gaps))
        elif len(ts_col) >= 2:
            gaps_h = ts_col.sort_values().diff().dropna().dt.total_seconds() / 3600.0
            gaps_h = gaps_h[gaps_h > 0]
            expected_interval_h = float(gaps_h.median()) if not gaps_h.empty else None

        # Regime detection: is the last 3 days (pooled across the type)
        # shifted from the type's own baseline?
        regime_changed = False
        if n_days >= REGIME_SHIFT_DAYS + WARMUP_DAYS and std > 0:
            cutoff = h["timestamp"].max() - pd.Timedelta(days=REGIME_SHIFT_DAYS)
            recent = h.loc[col.index[ts_col >= cutoff], feat].dropna()
            if len(recent) >= 3:
                recent_mean = float(recent.mean())
                if abs(recent_mean - mean) > REGIME_SHIFT_SIGMA * std:
                    regime_changed = True

        metrics[feat] = {
            "mean": mean,
            "std": std,
            "p5": p5,
            "p95": p95,
            "n_points": int(len(col)),
            "diurnal": diurnal,
            "expected_interval_h": expected_interval_h,
            "regime_changed": regime_changed,
        }

    return {
        "canonical_resource_type": canonical_resource_type,
        "category": category,
        "profile_confidence": profile_confidence,
        "n_days": n_days,
        "n_resources_pooled": n_resources_pooled,
        "metrics": metrics,
        "updated_at": _now().isoformat(),
    }


def get_diurnal_expected(profile: Dict, feature: str, timestamp) -> Optional[float]:
    """Return the diurnal median for a given feature and timestamp from a stored profile."""
    metric_profile = profile.get("metrics", {}).get(feature)
    if not metric_profile:
        return None
    ts = pd.Timestamp(timestamp)
    key = f"{ts.dayofweek}_{ts.hour}"
    return metric_profile.get("diurnal", {}).get(key)


def get_residual(value: float, profile: Dict, feature: str, timestamp) -> Optional[float]:
    """Return value minus its diurnal expected value (the anomaly residual)."""
    expected = get_diurnal_expected(profile, feature, timestamp)
    if expected is None:
        mp = profile.get("metrics", {}).get(feature)
        if mp:
            expected = mp.get("mean")
    if expected is None:
        return None
    return value - expected


def detect_reporting_gap(profile: Dict, feature: str, last_seen_ts, current_ts) -> bool:
    """Return True if the gap between last_seen and current exceeds 2× expected interval."""
    mp = profile.get("metrics", {}).get(feature)
    if not mp:
        return False
    interval = mp.get("expected_interval_h")
    if not interval or interval <= 0:
        return False
    gap_h = (pd.Timestamp(current_ts) - pd.Timestamp(last_seen_ts)).total_seconds() / 3600.0
    return gap_h > interval * 2.5


def classify_zero(profile: Dict, feature: str, series_zero_frac: float) -> str:
    """Classify whether zeros in a series are idle, missing, or sparse.

    Returns: 'idle' | 'missing' | 'sparse' | 'ok'
    """
    mp = profile.get("metrics", {}).get(feature)
    if not mp:
        return "ok"
    # Use heuristics: if the type profile shows low mean (< p5 * 1.1),
    # resources of this type are usually idle.
    mean = mp.get("mean", 0)
    p5 = mp.get("p5", 0)
    if mean <= 0.01 and p5 <= 0:
        return "idle"
    if series_zero_frac > 0.5:
        return "missing"
    if series_zero_frac > 0.1:
        return "sparse"
    return "ok"


def upsert_profile(uri: str, db_name: str, profiles_collection: str, profile: Dict):
    """Upsert a pooled type profile document into MongoDB."""
    from pymongo import MongoClient
    client = MongoClient(uri)
    col = client[db_name][profiles_collection]
    col.update_one(
        {"canonical_resource_type": profile["canonical_resource_type"], "category": profile["category"]},
        {"$set": profile},
        upsert=True,
    )


def load_profile(uri: str, db_name: str, profiles_collection: str,
                 canonical_resource_type: str, category: str) -> Optional[Dict]:
    """Load a pooled type profile document from MongoDB. Returns None if not found."""
    from pymongo import MongoClient
    client = MongoClient(uri)
    col = client[db_name][profiles_collection]
    return col.find_one(
        {"canonical_resource_type": canonical_resource_type, "category": category},
        {"_id": 0},
    )


def load_all_profiles(uri: str, db_name: str, profiles_collection: str,
                      category: str) -> Dict[str, Dict]:
    """Load all pooled type profiles for a category, keyed by canonical_resource_type."""
    from pymongo import MongoClient
    client = MongoClient(uri)
    col = client[db_name][profiles_collection]
    return {
        doc["canonical_resource_type"]: doc
        for doc in col.find({"category": category}, {"_id": 0})
    }


def build_all_profiles(df: pd.DataFrame) -> Dict[Tuple[str, str], Dict]:
    """Build one pooled profile per (canonical_resource_type, category) in df.

    Every resource sharing a canonical_resource_type contributes rows to the
    SAME profile — no profile is ever built from a single resource_id's
    history alone. Returns dict keyed by (canonical_resource_type, category).
    Rows with a missing/null canonical_resource_type are skipped (they should
    not occur post-normalization, since resource_types.canonical_resource_type
    always returns an `other_<category>` fallback bucket rather than null).
    """
    profiles = {}
    if "canonical_resource_type" not in df.columns:
        return profiles
    grouped = df.dropna(subset=["canonical_resource_type"]).groupby(
        ["canonical_resource_type", "category"], sort=False
    )
    for (ctype, cat), grp in grouped:
        profiles[(str(ctype), str(cat))] = build_profile(str(ctype), str(cat), grp)
    return profiles
