"""anomaly.py — Resource-type-pooled residual anomaly detection.

v7 architecture: all three signals operate on the RESIDUAL after subtracting
the canonical-resource-type diurnal pattern (from profiler.py, which pools
across every resource sharing a canonical_resource_type), not on raw values.
Critically, no signal ever fits a model to a single resource's own history —
every learned component (the diurnal baseline, the IsolationForest) is fit
once per (category, canonical_resource_type) using data pooled across all
resources of that type, and an individual resource is only ever *scored*
against its type's shared model. This still eliminates the global-population
bias that made consistent high-utilisation resources appear permanently
anomalous, because the pooled baseline is type-specific (a busy vm_instance
is compared to other vm_instances, not to storage buckets) — it just does so
without giving each resource_id its own private model.

Signal 1 — Residual z-score
    (value - type_diurnal_expected) / type_std  for each canonical feature.
    Catches: "this value is unusual FOR THIS RESOURCE TYPE at this time of
    day/week". Falls back to a rolling mean computed over the pooled type
    history when no type profile is available yet.

Signal 2 — Max/Avg burst ratio
    Unchanged — intra-hour spike detection on the resource's current row.
    Purely descriptive of that one reading, not a fitted model, so pooling
    doesn't apply.

Signal 3 — Type-pooled IsolationForest on normalised residuals
    Trained ONCE per (category, canonical_resource_type) — see
    `train_type_isolation_forests` — on residuals pooled across every
    resource that maps to that canonical type. A resource with almost no
    history of its own still gets a fully-formed score because it is scored
    against the shared type model rather than needing to fit its own.

Composite: OR logic for alerting (any signal triggers), weighted average for
the continuous risk_score to preserve gradations in the 0-100 output.
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from .metric_registry import canonical_features
from .profiler import get_diurnal_expected

ROLLING_WINDOW_H   = 7 * 24
MIN_HISTORY_FOR_ROLL = 12
Z_SCORE_THRESHOLD  = 3.0
MAX_AVG_RATIO_THRESHOLD = 5.0
IF_CONTAMINATION   = 0.05

# Minimum pooled rows (across ALL resources of a canonical type) before we
# trust a type-level IsolationForest at all. Below this, Signal 3 is simply
# switched off (weight 0) for that type rather than falling back to fitting
# a model on any single resource.
MIN_TYPE_POOL_FOR_IF = 100
# Pool size at which the type model is considered "well-trained" and gets
# boosted weight in the composite score.
LARGE_TYPE_POOL_FOR_IF = 2000

W_ZSCORE = 0.45
W_MAXAVG = 0.20
W_IF     = 0.35


def _primary_features(category: str) -> List[str]:
    return canonical_features(category)


def _compute_residuals(history: pd.DataFrame, category: str,
                       profile: Optional[Dict]) -> pd.DataFrame:
    """Return a DataFrame of per-feature residuals (value - diurnal_expected).

    `profile` is the pooled (canonical_resource_type, category) profile, not
    anything fit to `history` alone. When a profile exists, subtract the
    (day_of_week, hour_of_day) median from that pooled profile. When no
    profile exists yet for the type, subtract the rolling mean computed over
    `history` (shift(1) so the current point is not in its own baseline) —
    this fallback path does not fit any model, it's just a moving average.
    """
    feats = [f for f in _primary_features(category) if f in history.columns]
    h = history.sort_values("timestamp").copy()
    residuals = pd.DataFrame(index=h.index)

    for feat in feats:
        col = h[feat].astype(float)
        if col.notna().sum() < 3:
            residuals[feat] = np.nan
            continue

        if profile and profile.get("metrics", {}).get(feat):
            mp = profile["metrics"][feat]
            diurnal = mp.get("diurnal", {})
            expected = h["timestamp"].apply(
                lambda ts: diurnal.get(f"{pd.Timestamp(ts).dayofweek}_{pd.Timestamp(ts).hour}")
            )
            # Fill missing diurnal buckets with the type's overall mean
            expected = expected.fillna(mp.get("mean", col.mean()))
            residuals[feat] = col - expected
        else:
            # Fallback: subtract rolling mean (shift to avoid look-ahead).
            # Descriptive smoothing only — not a fitted model.
            roll_mean = col.shift(1).rolling(ROLLING_WINDOW_H, min_periods=6).mean()
            roll_mean = roll_mean.fillna(col.mean())
            residuals[feat] = col - roll_mean

    return residuals


def _residual_zscore_scores(history: pd.DataFrame, category: str,
                             profile: Optional[Dict]) -> pd.Series:
    """Return per-row anomaly score [0,1] based on residual z-score.

    Score of 1.0 means residual >= 3σ on at least one feature, where σ comes
    from the pooled type profile (not a per-resource std).
    """
    residuals = _compute_residuals(history, category, profile)
    feats = [f for f in residuals.columns if f in _primary_features(category)]

    if not feats or len(history) < MIN_HISTORY_FOR_ROLL:
        return pd.Series(0.0, index=history.index)

    max_z = pd.Series(0.0, index=residuals.index)

    for feat in feats:
        res = residuals[feat].dropna()
        if len(res) < MIN_HISTORY_FOR_ROLL:
            continue

        if profile and profile.get("metrics", {}).get(feat):
            std = profile["metrics"][feat].get("std", 0)
        else:
            std = float(res.rolling(ROLLING_WINDOW_H, min_periods=6).std().shift(1).median())

        if not std or std <= 0:
            continue

        z = (residuals[feat].abs() / std).fillna(0.0)
        max_z = np.maximum(max_z, z)

    # Sigmoid-style normalisation: z=3 → ~0.88, z=6 → 1.0
    score = (max_z / (Z_SCORE_THRESHOLD * 1.14)).clip(0.0, 1.0)
    return score.reindex(history.index).fillna(0.0)


def _burst_ratio_flag(history: pd.DataFrame, category: str) -> pd.Series:
    """Return per-row burst flag [0,1]: 1.0 if any feature's max/avg > threshold."""
    feats = [f for f in _primary_features(category) if f in history.columns]
    max_ratio = pd.Series(0.0, index=history.index)

    for feat in feats:
        max_col = f"{feat}_max"
        if max_col not in history.columns:
            continue
        avg_vals = history[feat].astype(float).replace(0, np.nan)
        max_vals = history[max_col].astype(float)
        ratio = (max_vals / avg_vals).fillna(1.0).clip(1.0, 50.0)
        max_ratio = np.maximum(max_ratio, ratio)

    # Normalise: ratio=1 → 0.0, ratio≥MAX_AVG_RATIO_THRESHOLD → 1.0
    score = ((max_ratio - 1.0) / (MAX_AVG_RATIO_THRESHOLD - 1.0)).clip(0.0, 1.0)
    return score.reindex(history.index).fillna(0.0)


def _residual_feature_matrix(history: pd.DataFrame, category: str,
                              profile: Optional[Dict]) -> pd.DataFrame:
    """Build the residual + burst-ratio feature matrix used to fit/score
    the IsolationForest. Shared by training (pooled) and scoring (single
    resource) so both use identical feature construction.
    """
    residuals = _compute_residuals(history, category, profile)
    feats = [f for f in residuals.columns if f in _primary_features(category)]
    if not feats:
        return pd.DataFrame(index=history.index)

    X = residuals[feats].copy()
    for feat in feats:
        max_col = f"{feat}_max"
        if max_col in history.columns:
            avg_vals = history[feat].replace(0, np.nan)
            ratio = (history[max_col] / avg_vals).clip(upper=50)
            X[f"{feat}_burst_ratio"] = ratio
    return X


def train_type_isolation_forests(df: pd.DataFrame, category: str,
                                  profiles: Dict[Tuple[str, str], Dict]) -> Dict[str, Dict]:
    """Fit one IsolationForest per canonical_resource_type, pooling residuals
    across EVERY resource that maps to that type — never per individual
    resource_id. This mirrors how models.train_category pools the risk
    classifier at category level, but one level finer (resource type within
    category), matching the granularity used for slope-prior pooling.

    Returns {canonical_resource_type: {"scaler", "model", "feature_cols",
    "medians", "n_pooled_rows"}}. Types with fewer than
    MIN_TYPE_POOL_FOR_IF pooled rows are skipped — no model is fit for them,
    so scoring for those resources simply omits Signal 3 rather than
    fitting a resource-specific fallback.
    """
    type_models: Dict[str, Dict] = {}
    if "canonical_resource_type" not in df.columns:
        return type_models

    cat_df = df[df["category"] == category]
    if cat_df.empty:
        return type_models

    for ctype, type_df in cat_df.groupby("canonical_resource_type", sort=False):
        if not ctype or len(type_df) < MIN_TYPE_POOL_FOR_IF:
            continue

        profile = profiles.get((str(ctype), category))
        X = _residual_feature_matrix(type_df.sort_values("timestamp"), category, profile)
        if X.empty or X.shape[1] == 0:
            continue

        medians = {}
        for col in X.columns:
            med = X[col].median()
            med = 0.0 if pd.isna(med) else float(med)
            medians[col] = med
            X[col] = X[col].fillna(med)

        if len(X) < MIN_TYPE_POOL_FOR_IF:
            continue

        scaler = RobustScaler()
        try:
            X_scaled = scaler.fit_transform(X.astype(float))
        except Exception:
            continue

        pos_rate_guess = 0.05
        contamination = float(np.clip(
            min(IF_CONTAMINATION, max(0.02, 5.0 / len(X))),
            0.01, 0.49,
        ))
        clf = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            max_samples=min(1024, len(X)),
            random_state=42,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_scaled)

        type_models[str(ctype)] = {
            "scaler": scaler,
            "model": clf,
            "feature_cols": list(X.columns),
            "medians": medians,
            "n_pooled_rows": int(len(X)),
        }

    return type_models


def _isolation_forest_scores(history: pd.DataFrame, category: str,
                              profile: Optional[Dict],
                              type_if_artifact: Optional[Dict]) -> pd.Series:
    """Score `history` (one resource's rows) against a pre-trained,
    type-pooled IsolationForest. Never fits a model here — if no type
    artifact is available (type pool too small / not yet trained), returns
    all zeros rather than fitting one on this resource's own data.
    """
    if not type_if_artifact:
        return pd.Series(0.0, index=history.index)

    X = _residual_feature_matrix(history, category, profile)
    feature_cols = type_if_artifact["feature_cols"]
    if X.empty:
        return pd.Series(0.0, index=history.index)

    # Align to the training-time feature set exactly (missing cols -> median).
    for col in feature_cols:
        if col not in X.columns:
            X[col] = np.nan
    X = X[feature_cols]
    for col in feature_cols:
        med = type_if_artifact["medians"].get(col, 0.0)
        X[col] = X[col].fillna(med)

    try:
        X_scaled = type_if_artifact["scaler"].transform(X.astype(float))
        raw = type_if_artifact["model"].decision_function(X_scaled)
    except Exception:
        return pd.Series(0.0, index=history.index)

    # Invert: more negative (anomalous) → higher score. Normalise against the
    # training pool's score range so a single new point doesn't get its own
    # min/max (which would always score it 0 or 1).
    inv = -raw
    # decision_function is roughly centered on 0 with the anomalous tail
    # negative; map [-0.3, 0.3] -> [1, 0] as a stable, pool-independent scale
    # rather than min/max-normalising over just this resource's rows.
    norm = np.clip((inv + 0.3) / 0.6, 0.0, 1.0)
    return pd.Series(norm, index=X.index).reindex(history.index).fillna(0.0)


def compute_anomaly_score(
    resource_history: pd.DataFrame,
    category: str,
    profile: Optional[Dict] = None,
    type_if_artifact: Optional[Dict] = None,
) -> Tuple[float, Dict]:
    """Composite anomaly score [0, 100] for the latest row.

    `profile` and `type_if_artifact` are both pooled at the
    canonical_resource_type level — nothing here is fit to
    `resource_history` alone. `resource_history` (this one resource's own
    timeseries) is only used to know its current/recent values, never to
    train a model.

    Returns (score_0_to_100, detail_dict).
    """
    if resource_history.empty:
        return 0.0, {"error": "no_history"}

    h = resource_history.sort_values("timestamp").copy()
    n = len(h)

    z_scores  = _residual_zscore_scores(h, category, profile)
    br_scores = _burst_ratio_flag(h, category)
    if_scores = _isolation_forest_scores(h, category, profile, type_if_artifact)

    # Adaptive weights: boost Signal 3 when the TYPE's pooled model was
    # trained on a large pool (a well-trained shared model is a cleaner
    # signal), not when this individual resource has a lot of history.
    pool_size = type_if_artifact.get("n_pooled_rows", 0) if type_if_artifact else 0
    if not type_if_artifact or pool_size < MIN_TYPE_POOL_FOR_IF:
        w_if = 0.0
    elif pool_size >= LARGE_TYPE_POOL_FOR_IF:
        w_if = min(0.55, W_IF + 0.20)
    else:
        w_if = W_IF

    w_z  = W_ZSCORE + (W_IF - w_if) * 0.7
    w_br = W_MAXAVG + (W_IF - w_if) * 0.3
    total_w = w_z + w_br + w_if

    composite = (w_z * z_scores + w_br * br_scores + w_if * if_scores) / total_w

    latest_idx = h.index[-1]
    def _get(series, idx):
        return float(series.loc[idx]) if idx in series.index else 0.0

    lz  = _get(z_scores,  latest_idx)
    lbr = _get(br_scores, latest_idx)
    lif = _get(if_scores, latest_idx)
    lc  = _get(composite, latest_idx)

    score_100 = round(lc * 100, 2)

    # OR-logic alert trigger: ANY signal dominant
    any_dominant = lz > 0.5 or lbr > 0.6 or lif > 0.55
    drivers = []
    if lz  > 0.4: drivers.append(f"residual_zscore={lz:.2f}")
    if lbr > 0.4: drivers.append(f"burst_ratio={lbr:.2f}")
    if lif > 0.4: drivers.append(f"isolation_forest={lif:.2f}")

    profile_confidence = profile.get("profile_confidence", 0.0) if profile else 0.0
    profile_age_days = profile.get("n_days", 0) if profile else 0
    canonical_resource_type = profile.get("canonical_resource_type") if profile else None

    detail = {
        "residual_zscore_signal":    round(lz,  3),
        "burst_ratio_signal":        round(lbr, 3),
        "isolation_forest_signal":   round(lif, 3),
        "composite_anomaly_score":   score_100,
        "any_signal_dominant":       any_dominant,
        "history_points_used":       n,
        "canonical_resource_type":   canonical_resource_type,
        "profile_confidence":        round(profile_confidence, 3),
        "profile_age_days":          profile_age_days,
        "type_pool_size":            pool_size,
        "drivers":                   drivers if drivers else ["none_dominant"],
        "method":                    "residual_type_pooled_composite",
        "weights_used": {
            "residual_zscore":       round(w_z  / total_w, 3),
            "burst_ratio":           round(w_br / total_w, 3),
            "isolation_forest":      round(w_if / total_w, 3),
        },
    }
    return score_100, detail


def score_resource(resource_history: pd.DataFrame, category: str,
                   profile: Optional[Dict] = None,
                   type_if_artifact: Optional[Dict] = None) -> Tuple[float, Dict]:
    """Public entry point. Returns (risk_score_0_to_100, anomaly_detail).

    `profile` and `type_if_artifact` must both be looked up by the
    resource's canonical_resource_type (see profiler.build_all_profiles and
    train_type_isolation_forests) — this function scores one resource
    against its type's shared baseline/model, it does not fit anything.
    """
    try:
        return compute_anomaly_score(resource_history, category, profile, type_if_artifact)
    except Exception as e:
        return 0.0, {"error": str(e), "method": "residual_type_pooled_composite"}
