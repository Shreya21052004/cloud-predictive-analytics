"""models.py — Risk classifier training.

Key changes from v2:
1. weak_risk_label z-score fallback now only fires on HIGH values (z > 2.5),
   not on absolute deviation. Consistently busy resources are no longer
   mislabelled as risky.
2. IsolationForest contamination derived from actual positive rate in training
   data, not hardcoded at 0.05.
3. XGBoost depth and estimator count scale with training set size to prevent
   overfitting on small per-category splits.
4. canonical_cpu_idle_pct → cpu_pct derivation: if cpu_pct is null but idle_pct
   is available, derive cpu_pct = 100 - idle_pct before feature extraction.
5. All-NaN canonical columns are now correctly classified as numeric (not pushed
   into OneHotEncoder as object dtype).
"""
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    HAS_XGBOOST = False
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer

from .metric_registry import canonical_features
from .resource_types import is_well_populated
from .forecast_math import safe_polyfit_slope
from .config import TARGET_CATEGORIES


# Pooled slope-prior model — used to shrink noisy local slope estimates for
# resource/metric pairs with thin history (see build_slope_prior_table below).
SLOPE_PRIOR_MIN_POINTS        = 20  # min points for a (resource,feature) series to seed a training row
SLOPE_PRIOR_MIN_TRAINING_ROWS = 30  # min pooled training rows before we trust a learned prior at all
SLOPE_PRIOR_SHRINK_K          = 15  # empirical-Bayes shrinkage constant: weight = n / (n + K)

MODEL_VERSION = {
    "Compute":    "compute-risk-v6.0",
    "Network":    "network-risk-v6.0",
    "Storage":    "storage-risk-v6.0",
    "Container":  "container-risk-v6.0",
    "Databases":  "databases-risk-v6.0",
    "Governance": "governance-risk-v6.0",
    "Security":   "security-risk-v6.0",
}


def is_real(value):
    return value is not None and not pd.isna(value)


def ge(row, key, threshold):
    value = row.get(key)
    return is_real(value) and value >= threshold


def gt(row, key, threshold):
    value = row.get(key)
    return is_real(value) and value > threshold


def lt(row, key, threshold):
    value = row.get(key)
    return is_real(value) and value < threshold


def derive_missing_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Derive canonical_cpu_pct from canonical_cpu_idle_pct when absent.

    AWS Telegraf agents often report cpu_usage_idle but not CPUUtilization.
    Without this, Compute rows for those resources have no CPU signal at all.
    """
    df = df.copy()
    if "canonical_cpu_pct" in df.columns and "canonical_cpu_idle_pct" in df.columns:
        missing_mask = df["canonical_cpu_pct"].isna() & df["canonical_cpu_idle_pct"].notna()
        df.loc[missing_mask, "canonical_cpu_pct"] = 100.0 - df.loc[missing_mask, "canonical_cpu_idle_pct"]
        df.loc[missing_mask, "canonical_cpu_pct_available"] = True
    return df


def feature_columns(df, category):
    bases = canonical_features(category)
    numeric_cols = []
    categorical_cols = []
    for col in bases:
        # Always treat canonical_* as numeric regardless of actual dtype
        if col in df.columns:
            numeric_cols.append(col)
        avail = f"{col}_available"
        if avail in df.columns:
            numeric_cols.append(avail)
    for col in ("service_name", "component", "location"):
        if col in df.columns:
            categorical_cols.append(col)
    return numeric_cols + categorical_cols, numeric_cols, categorical_cols


def build_preprocessor(df, numeric_cols, categorical_cols, for_xgboost=False):
    transformers = []
    if numeric_cols:
        if for_xgboost:
            transformers.append(("num", FunctionTransformer(), numeric_cols))
        else:
            numeric_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
            ])
            transformers.append(("num", numeric_pipeline, numeric_cols))
    if categorical_cols:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", categorical_pipeline, categorical_cols))
    return ColumnTransformer(transformers, remainder="drop")


def weak_risk_label(row, category, col_stats=None):
    """Threshold-based risk labeller with directional z-score fallback.

    col_stats: dict of {col: (mean, std)} from training population.
    Z-score fallback only fires for HIGH values (z > 2.5) — a consistently
    busy but healthy resource should not be labelled risky just because it's
    above the population mean.
    """
    score = 0
    if category == "Compute":
        score += 35 if ge(row, "canonical_cpu_pct", 80) else 0
        score += 30 if ge(row, "canonical_mem_used_pct", 85) else 0
        score += 20 if ge(row, "canonical_iowait_pct", 20) else 0
        score += 30 if gt(row, "status_check_failed", 0) else 0
        score += 20 if gt(row, "lambda_throttles", 0) else 0
        score += 15 if lt(row, "cpu_credit_balance", 20) else 0
    elif category == "Network":
        score += 45 if gt(row, "canonical_ddos_signal", 0) else 0
        score += 35 if lt(row, "canonical_health_pct", 90) else 0
        score += 30 if ge(row, "canonical_subnet_util_pct", 80) else 0
        score += 15 if gt(row, "canonical_active_connections", 10000) else 0
    elif category == "Storage":
        score += 35 if ge(row, "canonical_storage_used_pct", 80) else 0
        score += 30 if lt(row, "canonical_burst_balance_pct", 20) else 0
        score += 25 if gt(row, "canonical_queue_depth", 10) else 0
        score += 25 if lt(row, "availability_pct", 99) else 0
        score += 20 if gt(row, "backup_health_event", 0) else 0

    if score >= 50:
        return 1

    # Directional z-score fallback: only HIGH values count
    # (a resource consistently above mean is NOT anomalous — only a sudden spike is)
    if col_stats:
        for col, (mean, std) in col_stats.items():
            val = row.get(col)
            if val is not None and not pd.isna(val) and std > 0:
                z = (float(val) - mean) / std
                if z > 2.5:
                    return 1
    return 0


def add_missingness_features(df, numeric_cols):
    """Binary indicator columns: 1 = metric was NaN (not reported), 0 = was present."""
    df = df.copy()
    for col in numeric_cols:
        df[f"{col}_missing"] = df[col].isna().astype(int)
    return df


def time_aware_split(df, test_frac=0.2):
    """Per-resource chronological split. Most-recent test_frac of each resource = test."""
    train_idx, test_idx = [], []
    for _, grp in df.groupby("resource_id", sort=False):
        grp_sorted = grp.sort_values("timestamp")
        n = len(grp_sorted)
        split_point = max(1, int(n * (1 - test_frac)))
        train_idx.extend(grp_sorted.index[:split_point].tolist())
        test_idx.extend(grp_sorted.index[split_point:].tolist())
    return df.loc[train_idx], df.loc[test_idx]


def evaluate_classifier(model, X_test, y_test):
    if len(X_test) == 0:
        return {}
    y_pred = np.array(model.predict(X_test)).astype(int)
    y_test_arr = np.array(y_test).astype(int)

    label_positive_rate = round(float(y_test_arr.mean()), 4)
    majority_class = int(y_test_arr.mean() >= 0.5)
    baseline_acc = round(float((y_test_arr == majority_class).mean()), 4)

    metrics = {
        "accuracy":             round(float(accuracy_score(y_test_arr, y_pred)), 4),
        "baseline_accuracy":    baseline_acc,
        "precision":            round(float(precision_score(y_test_arr, y_pred, zero_division=0)), 4),
        "recall":               round(float(recall_score(y_test_arr, y_pred, zero_division=0)), 4),
        "f1":                   round(float(f1_score(y_test_arr, y_pred, zero_division=0)), 4),
        "label_positive_rate":  label_positive_rate,
        "model_type":           "xgboost" if HAS_XGBOOST else "random_forest",
        "label_source":         "weak_rule_labels_directional_zscore",
        "test_size":            len(X_test),
    }
    estimator = model.named_steps["estimator"]
    if hasattr(estimator, "predict_proba") and y_test.nunique() > 1:
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
            metrics["roc_auc"] = round(float(roc_auc_score(y_test_arr, y_prob)), 4)
        except Exception:
            pass
    return metrics


def evaluate_anomaly(model, X_test, y_test):
    if len(X_test) == 0:
        return {}
    scores = model.decision_function(X_test)
    return {
        "predicted_anomaly_rate": round(float((scores < 0).sum() / len(scores)), 4),
        "actual_risk_rate":       round(float(y_test.mean()) if len(y_test) > 0 else 0.0, 4),
        "test_size":              len(X_test),
    }


def _normalized_slope(hours, values):
    """OLS slope normalized by the series' own value range (units: range/hour).

    Normalizing lets us pool slope targets across metrics with wildly
    different scales (bytes vs percent vs connection counts) into one model.
    """
    if len(hours) < 2:
        return None
    y = np.asarray(values, dtype=float)
    value_range = float(y.max() - y.min())
    if value_range <= 0:
        return 0.0
    slope, _ = safe_polyfit_slope(np.asarray(hours, dtype=float), y)
    return float(slope / value_range)


def build_slope_prior_table(df, category):
    """One pooled training row per (resource_id, canonical_feature) with
    enough history (>= SLOPE_PRIOR_MIN_POINTS). Target: normalized hourly
    slope. Context features: latest value + availability of every canonical
    metric for that resource, plus metadata — so the model can learn e.g.
    'CPU tends to trend up when iowait is also elevated' and apply that to a
    different resource that only has a handful of CPU points so far.
    """
    feats = canonical_features(category)
    if not feats or df.empty:
        return pd.DataFrame()

    cat_df = df[df["category"] == category].copy()
    if cat_df.empty:
        return pd.DataFrame()
    cat_df = derive_missing_signals(cat_df)

    rows = []
    for _, grp in cat_df.groupby("resource_id", sort=False):
        grp = grp.sort_values("timestamp")
        hours_all = (grp["timestamp"] - grp["timestamp"].min()).dt.total_seconds() / 3600.0
        latest_ctx = {
            f: (grp[f].dropna().iloc[-1] if f in grp.columns and grp[f].notna().any() else None)
            for f in feats
        }
        meta = grp.iloc[-1]
        for feature in feats:
            if feature not in grp.columns:
                continue
            mask = grp[feature].notna()
            if mask.sum() < SLOPE_PRIOR_MIN_POINTS:
                continue
            slope_n = _normalized_slope(hours_all[mask].values, grp.loc[mask, feature].values)
            if slope_n is None:
                continue
            row = {
                "target_feature": feature,
                "point_count":    int(mask.sum()),
                "value_range":    float(grp.loc[mask, feature].max() - grp.loc[mask, feature].min()),
                "service_name":   meta.get("service_name"),
                "component":      meta.get("component"),
                # Canonical resource type (see resource_types.py) alongside
                # the raw `component` -- this is what lets a thin
                # Compute_Engine resource borrow strength from the much
                # larger pooled EC2/Virtual_Machines population instead of
                # only ever seeing its own raw-component one-hot column.
                "canonical_resource_type": meta.get("canonical_resource_type"),
                "location":       meta.get("location"),
                "y":              slope_n,
            }
            for f in feats:
                v = latest_ctx.get(f)
                row[f"ctx_{f}"] = np.nan if v is None else float(v)
            rows.append(row)
    return pd.DataFrame(rows)


def train_slope_prior(df, category):
    """Train the pooled slope-shrinkage prior for one category. Returns None
    if there isn't enough pooled data to trust a learned prior — in that case
    callers fall back to unshrunk local slope estimates, same as today.
    """
    table = build_slope_prior_table(df, category)
    if len(table) < SLOPE_PRIOR_MIN_TRAINING_ROWS:
        return None

    ctx_cols = [c for c in table.columns if c.startswith("ctx_")]
    numeric_cols = ctx_cols + ["point_count", "value_range"]
    categorical_cols = [
        c for c in ("target_feature", "service_name", "component",
                     "canonical_resource_type", "location")
        if c in table.columns
    ]

    preprocessor = build_preprocessor(table, numeric_cols, categorical_cols, for_xgboost=HAS_XGBOOST)

    n_train = len(table)
    if HAS_XGBOOST:
        estimator = XGBRegressor(
            n_estimators=200 if n_train < 500 else 300,
            max_depth=4 if n_train < 500 else 5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.5,
            reg_alpha=0.1,
            random_state=42,
            missing=np.nan,
            verbosity=0,
        )
    else:
        estimator = RandomForestRegressor(n_estimators=150, random_state=42)

    model = Pipeline([("preprocessor", preprocessor), ("estimator", estimator)])
    X = table[numeric_cols + categorical_cols]
    y = table["y"]
    model.fit(X, y)

    return {
        "model":              model,
        "numeric_cols":       numeric_cols,
        "categorical_cols":   categorical_cols,
        "canonical_features": canonical_features(category),
        "n_train":            n_train,
        "model_type":         "xgboost" if HAS_XGBOOST else "random_forest",
    }


def predict_slope_prior(slope_prior_artifact, feature, resource_history, current_row):
    """Predict the pooled-model's normalized hourly slope for one
    (resource, feature) pair. Returns None if unavailable/inapplicable —
    callers must treat that as "no prior, use local slope only".
    """
    if not slope_prior_artifact:
        return None
    feats = slope_prior_artifact["canonical_features"]
    row = {
        "target_feature": feature,
        "service_name":   current_row.get("service_name"),
        "component":      current_row.get("component"),
        "canonical_resource_type": current_row.get("canonical_resource_type"),
        "location":       current_row.get("location"),
    }
    if feature in resource_history.columns and resource_history[feature].notna().any():
        vals = resource_history[feature].dropna()
        row["point_count"] = int(vals.shape[0])
        row["value_range"] = float(vals.max() - vals.min())
    else:
        row["point_count"] = 0
        row["value_range"] = 0.0
    for f in feats:
        if f in resource_history.columns and resource_history[f].notna().any():
            row[f"ctx_{f}"] = float(resource_history[f].dropna().iloc[-1])
        else:
            row[f"ctx_{f}"] = np.nan

    X = pd.DataFrame([row])
    for c in slope_prior_artifact["numeric_cols"]:
        if c not in X.columns:
            X[c] = np.nan
        X[c] = pd.to_numeric(X[c], errors="coerce")
    try:
        return float(slope_prior_artifact["model"].predict(X)[0])
    except Exception:
        return None


def build_type_specific_slope_priors(df, category):
    """Precompute pooled slope priors for all (canonical_type, feature) pairs in a category.
    
    This is computed once per category and cached, then looked up quickly for each resource.
    Returns dict: {(canonical_type, feature): slope_norm}
    """
    feats = canonical_features(category)
    if not feats:
        return {}
    
    cat_df = df[df["category"] == category].copy()
    if cat_df.empty:
        return {}
    
    type_priors = {}
    
    # Get all unique canonical types in this category
    canonical_types = cat_df.get("canonical_resource_type").dropna().unique()
    
    for canonical_type in canonical_types:
        if not canonical_type:
            continue
        
        # Filter to resources of this type
        type_df = cat_df[cat_df.get("canonical_resource_type") == canonical_type].copy()
        if type_df.empty or len(type_df) < 2:
            continue
        
        # Compute slopes for each feature
        for feature in feats:
            if feature not in type_df.columns:
                continue
            
            slopes = []
            for _, grp in type_df.groupby("resource_id", sort=False):
                mask = grp[feature].notna()
                # Lower threshold: only need 5+ points to contribute to pool
                # (resources with 20+ points are strong, but even 5+ helps)
                min_points_for_pool = min(5, max(2, SLOPE_PRIOR_MIN_POINTS // 4))
                if mask.sum() < min_points_for_pool:
                    continue
                
                hours = (grp["timestamp"] - grp["timestamp"].min()).dt.total_seconds() / 3600.0
                slope_norm = _normalized_slope(hours[mask].values, grp.loc[mask, feature].values)
                
                if slope_norm is not None:
                    slopes.append(slope_norm)
            
            if slopes:
                # Store median slope AND pool size for this type/feature pair
                type_priors[(canonical_type, feature)] = (float(np.median(slopes)), len(slopes))
    
    return type_priors


def compute_type_specific_slope_prior(type_priors_cache, canonical_type, feature):
    """Look up precomputed type-specific slope prior and pool size.
    
    Args:
        type_priors_cache: dict from build_type_specific_slope_priors()
        canonical_type: resource's canonical type
        feature: metric name
    
    Returns: (slope_norm, pool_size) or (None, 0) if not available
    """
    if not canonical_type or not feature:
        return None, 0
    result = type_priors_cache.get((canonical_type, feature))
    if result is None:
        return None, 0
    return result


def train_category(df, category):
    category_df = df[df["category"] == category].copy()
    if category_df.empty:
        return None

    # Derive missing CPU signal from idle before feature extraction
    category_df = derive_missing_signals(category_df)

    cols, numeric_cols, categorical_cols = feature_columns(category_df, category)
    if not cols:
        return None

    # Force all canonical_* columns to numeric (prevents object-dtype columns
    # from being misrouted into OneHotEncoder when they're all-NaN)
    for col in numeric_cols:
        category_df[col] = pd.to_numeric(category_df[col], errors="coerce")

    col_stats = {
        col: (float(category_df[col].mean()), float(category_df[col].std()))
        for col in numeric_cols
        if col in category_df.columns and category_df[col].std() > 0
    }
    _t_labels = time.monotonic()
    y_all = category_df.apply(lambda row: weak_risk_label(row, category, col_stats), axis=1)
    print(f"    [train:{category}] weak_risk_label computed for {len(category_df)} row(s) "
          f"in {time.monotonic() - _t_labels:.0f}s (row-wise, not vectorized — "
          f"this is usually the slowest step on a large category).")

    category_df = add_missingness_features(category_df, numeric_cols)
    missing_cols = [f"{c}_missing" for c in numeric_cols if f"{c}_missing" in category_df.columns]
    all_cols = cols + missing_cols

    train_df, test_df = time_aware_split(category_df)
    X_train = train_df[all_cols].copy()
    y_train = y_all.loc[train_df.index]
    X_test  = test_df[all_cols].copy()
    y_test  = y_all.loc[test_df.index]

    kind = "classifier" if (y_train.nunique() > 1 and len(train_df) >= 10) else "anomaly"
    preprocessor = build_preprocessor(
        X_train, numeric_cols + missing_cols, categorical_cols,
        for_xgboost=(kind == "classifier" and HAS_XGBOOST),
    )

    if kind == "classifier":
        pos = int(y_train.sum())
        neg = len(y_train) - pos
        scale_pos = neg / pos if pos > 0 else 1.0
        n_train = len(X_train)

        if HAS_XGBOOST:
            xgb_depth = 4 if n_train < 500 else (5 if n_train < 2000 else 6)
            xgb_n = 200 if n_train < 500 else 300
            estimator = XGBClassifier(
                n_estimators=xgb_n,
                max_depth=xgb_depth,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=max(1, int(pos * 0.05)),
                reg_lambda=1.5,
                reg_alpha=0.1,
                scale_pos_weight=scale_pos,
                eval_metric="logloss",
                random_state=42,
                missing=np.nan,
                verbosity=0,
            )
        else:
            estimator = RandomForestClassifier(
                n_estimators=120, random_state=42, class_weight="balanced"
            )
    else:
        actual_pos_rate = float(y_train.mean()) if len(y_train) > 0 else 0.05
        if_contam = float(np.clip(actual_pos_rate, 0.01, 0.49))
        estimator = IsolationForest(
            n_estimators=200,
            contamination=if_contam,
            max_samples="auto",
            random_state=42,
        )

    model = Pipeline([("preprocessor", preprocessor), ("estimator", estimator)])
    _t_fit = time.monotonic()
    if kind == "classifier":
        model.fit(X_train, y_train)
        eval_metrics = evaluate_classifier(model, X_test, y_test)
    else:
        model.fit(X_train)
        eval_metrics = evaluate_anomaly(model, X_test, y_test)
    print(f"    [train:{category}] model.fit() ({kind}, {len(X_train)} train row(s)) "
          f"done in {time.monotonic() - _t_fit:.0f}s.")

    return {
        "category":      category,
        "kind":          kind,
        "model":         model,
        "features":      all_cols,
        "model_version": MODEL_VERSION[category],
        "eval_metrics":  eval_metrics,
        "slope_prior":   train_slope_prior(df, category),
    }


def train_models(df, models_dir):
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    trained = {}
    print(f"  [train] starting {len(TARGET_CATEGORIES)} categor(ies) over {len(df)} normalized row(s) total.")
    for category in TARGET_CATEGORIES:
        n_rows = int((df["category"] == category).sum()) if "category" in df.columns else 0
        t0 = time.monotonic()
        print(f"  [train] {category}: {n_rows} row(s) — computing risk labels, then fitting...")
        artifact = train_category(df, category)
        elapsed = time.monotonic() - t0
        if artifact is None:
            print(f"  [train] {category}: skipped (empty or no usable feature columns) in {elapsed:.0f}s.")
            continue
        path = Path(models_dir) / f"{category.lower()}_model.joblib"
        joblib.dump(artifact, path)
        trained[category] = artifact
        print(f"  [train] {category}: done in {elapsed:.0f}s (kind={artifact.get('kind','?')}).")
    return trained


def load_models(models_dir):
    loaded = {}
    for category in TARGET_CATEGORIES:
        path = Path(models_dir) / f"{category.lower()}_model.joblib"
        if path.exists():
            loaded[category] = joblib.load(path)
    return loaded


def risk_from_model(artifact, row):
    X = pd.DataFrame([{col: row.get(col) for col in artifact["features"]}])
    numeric_cols = [
        c for c in artifact["features"]
        if c not in ("service_name", "component", "location")
    ]
    for col in numeric_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    estimator = artifact["model"].named_steps["estimator"]
    if artifact["kind"] == "classifier" and hasattr(estimator, "predict_proba"):
        return float(artifact["model"].predict_proba(X)[0, 1] * 100.0)
    score = float(artifact["model"].decision_function(X)[0])
    return float(np.clip((0.12 - score) / 0.24 * 100.0, 0, 100))
