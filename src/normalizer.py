"""normalizer.py — Raw MongoDB documents → normalized DataFrame rows.

Changes from v2:
1. detect_data_state() classifies each (resource, feature, timestamp) as
   'idle' | 'missing' | 'sparse' | 'ok', written as {feature}_data_state.
   downstream can suppress forecasts on 'missing' data and skip anomaly
   scoring on 'idle' resources.
2. Reporting gap detection: {feature}_gap_flag = True when the gap since
   the previous non-null value for this (resource, feature) exceeds 2.5×
   the resource's median reporting interval. Consumers can distinguish
   "value is zero" from "agent stopped reporting".
3. cpu_idle derivation is done inside the normaliser (not just at predict time),
   so the canonical_cpu_pct_available flag is correctly set even for resources
   that only report cpu_usage_idle.
"""
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .config import CATEGORY_ALIASES, PROVIDER_ALIASES
from .metric_registry import REGISTRY
from .resource_types import canonical_resource_type


def normalize_category(value):
    if value is None:
        return None
    return CATEGORY_ALIASES.get(str(value).strip().lower(), str(value).strip())


def normalize_provider(value):
    if value is None:
        return "unknown"
    return PROVIDER_ALIASES.get(str(value).strip().lower(), str(value).strip())


def get_ci(d, *keys):
    """Look up the first present key in d, case-insensitively."""
    if not isinstance(d, dict):
        return None
    lowered = {str(k).lower(): v for k, v in d.items()}
    for key in keys:
        value = lowered.get(key.lower())
        if value is not None:
            return value
    return None


STAT_PRIORITY = ["average", "delta", "maximum", "sum", "total", "minimum", "count", "samplecount", "value"]
# BUG FIX (root cause of the vpc_network data gap): GCP's cumulative/interval
# counters -- networking.googleapis.com/vm_flow/ingress_bytes_count,
# .../egress_bytes_count, .../vpc_flow/predicted_max_vpc_flow_logs_count,
# and others -- report their reading under a "Delta" key
# (e.g. {"Timestamp": ..., "Delta": "10516"}), not any of
# average/maximum/sum/total/minimum/count/samplecount/value. None of those
# matched, so extract_metric_value() returned (None, None) for every single
# one of these documents -- silently dropping real telemetry for ~84% of all
# raw Network-category docs (236,575 of ~283k), which is why the vast
# majority of vpc_network resources showed forecast_method: "none" /
# data_note: "no_data_points" regardless of any forecast-feature-priority
# fix. Adding "delta" here (case-insensitively, via the existing `lowered`
# lookup) lets this data flow through normally.


def extract_metric_value(doc):
    """Extract (numeric_value, stat_field) from a doc."""
    metric_value = doc.get("metric_value")
    if isinstance(metric_value, dict):
        lowered = {str(k).lower(): (k, v) for k, v in metric_value.items()}
        for stat in STAT_PRIORITY:
            entry = lowered.get(stat)
            if entry is not None and entry[1] is not None:
                return float(entry[1]), stat
    lowered = {str(k).lower(): (k, v) for k, v in doc.items()}
    for stat in STAT_PRIORITY:
        entry = lowered.get(stat)
        if entry is not None and entry[1] is not None:
            try:
                return float(entry[1]), stat
            except (TypeError, ValueError):
                continue
    return None, None


def extract_stat_bundle(item):
    """Extract Average, Maximum, Minimum, SampleCount from a metric_value item."""
    if not isinstance(item, dict):
        return {"average": None, "maximum": None, "minimum": None, "sample_count": None}
    lowered = {str(k).lower(): v for k, v in item.items()}
    def _f(key):
        v = lowered.get(key)
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None
    return {
        "average":      _f("average"),
        "maximum":      _f("maximum"),
        "minimum":      _f("minimum"),
        "sample_count": _f("samplecount"),
    }


def parse_datetime(raw):
    if raw is None:
        return datetime.now(timezone.utc)
    if isinstance(raw, dict) and "$date" in raw:
        raw = raw["$date"]
    parsed = pd.to_datetime(raw, utc=True, errors="coerce")
    if pd.isna(parsed):
        return datetime.now(timezone.utc)
    return parsed.to_pydatetime()


def extract_timestamp(doc):
    metric_value = doc.get("metric_value")
    raw = None
    if isinstance(metric_value, dict):
        raw = get_ci(metric_value, "timestamp")
    raw = raw or doc.get("from_date") or doc.get("created_at")
    return parse_datetime(raw)


def metric_points(doc):
    """Return list of (timestamp, value, stat_field, bundle) tuples for a doc."""
    metric_value = doc.get("metric_value")
    if isinstance(metric_value, list):
        points = []
        for item in metric_value:
            if not isinstance(item, dict):
                continue
            ts_raw = get_ci(item, "timestamp") or doc.get("from_date") or doc.get("created_at")
            value, stat = extract_metric_value({"metric_value": item})
            bundle = extract_stat_bundle(item)
            points.append((parse_datetime(ts_raw), value, stat, bundle))
        return points
    value, stat = extract_metric_value(doc)
    bundle = extract_stat_bundle(doc.get("metric_value") or doc)
    return [(extract_timestamp(doc), value, stat, bundle)]


def resource_id(doc):
    return (doc.get("element_id") or doc.get("resourceId") or
            doc.get("resource_id") or doc.get("inventory_id") or doc.get("_id"))


def resource_name(doc):
    return (doc.get("resource_name") or doc.get("display_name") or
            doc.get("name") or str(resource_id(doc)))


def provider(doc):
    return normalize_provider(doc.get("service_name") or doc.get("provider") or "unknown")


def base_context(doc):
    category = normalize_category(doc.get("category"))
    component = doc.get("component") or doc.get("service_type") or doc.get("resource_type")
    return {
        "resource_id":   str(resource_id(doc)),
        "resource_name": resource_name(doc),
        "account_id":    doc.get("account_id"),
        "account_name":  doc.get("account_name"),
        "project":       doc.get("project"),
        "category":      category,
        "component":     component,
        # Additive field: coarser real-world type used for pooled forecasting
        # (see resource_types.py). Kept alongside the raw `component` field
        # rather than replacing it -- existing consumers of `component` are
        # unaffected.
        "canonical_resource_type": canonical_resource_type(category, component),
        "location":      doc.get("location"),
        "service_name":  provider(doc),
        "tags":          doc.get("tags"),
        # Free-text description of the resource (what it's for, owning
        # team, etc.), carried through from whichever source doc has it so
        # the LLM project-summary step can ground its recommendations in
        # more than just the metric numbers.
        "summary_details": doc.get("summary_details") or doc.get("description"),
    }


def _derive_cpu_from_idle(rows_list):
    """Post-process: derive canonical_cpu_pct from canonical_cpu_idle_pct where missing.

    This ensures cpu_pct_available is True for AWS Telegraf resources that only
    report cpu_usage_idle, not CPUUtilization.
    """
    for row in rows_list:
        if row.get("canonical_cpu_pct") is None and row.get("canonical_cpu_idle_pct") is not None:
            idle = row["canonical_cpu_idle_pct"]
            row["canonical_cpu_pct"] = 100.0 - idle
            row["canonical_cpu_pct_available"] = True


def _add_data_state_and_gap_flags(df):
    """Add {feature}_data_state and {feature}_gap_flag columns per resource.

    data_state:
      'idle'    — value is consistently zero (metric exists but resource not busy)
      'missing' — last gap was > 2.5× expected interval (agent silence)
      'sparse'  — sporadic: >10% zeros but not clearly missing
      'ok'      — normal

    gap_flag:
      True when the gap since the previous non-null timestamp for this
      (resource_id, feature) exceeds 2.5× the per-resource median interval.
    """
    feature_cols = [c for c in df.columns
                    if c.startswith("canonical_") and not c.endswith((
                        "_available", "_stat", "_max", "_min", "_sample_count",
                        "_data_state", "_gap_flag",
                    ))]

    state_frames = []
    gap_frames = []

    for (rid, cat), grp in df.groupby(["resource_id", "category"], sort=False):
        grp = grp.sort_values("timestamp")

        for feat in feature_cols:
            if feat not in grp.columns:
                continue

            col = grp[feat]
            vals = pd.to_numeric(col, errors='coerce').values.astype(float)
            n = len(vals)
            non_null_mask = ~np.isnan(vals)

            # Compute per-resource median reporting interval for this feature
            ts_non_null = grp.loc[non_null_mask, "timestamp"]
            if len(ts_non_null) >= 2:
                gaps_h = (ts_non_null.diff().dropna().dt.total_seconds() / 3600.0)
                gaps_h = gaps_h[gaps_h > 0]
                median_interval = float(gaps_h.median()) if not gaps_h.empty else None
            else:
                median_interval = None

            # Gap flag: True when gap since last non-null > 2.5× median interval
            gap_flags = pd.Series(False, index=grp.index)
            if median_interval and median_interval > 0:
                last_ts = None
                for idx in grp.index:
                    ts = grp.loc[idx, "timestamp"]
                    val = grp.loc[idx, feat]
                    is_null = pd.isna(val)
                    if not is_null:
                        if last_ts is not None:
                            gap_h = (ts - last_ts).total_seconds() / 3600.0
                            if gap_h > median_interval * 2.5:
                                gap_flags.loc[idx] = True
                        last_ts = ts

            gap_frames.append((grp.index, feat, gap_flags))

            # Data state: one label per (resource, feature, row)
            zero_frac = float((vals == 0).sum()) / max(1, n)
            null_frac  = float(np.isnan(vals).sum()) / max(1, n)

            if zero_frac > 0.8 or (null_frac > 0.8 and (col.notna() & (col == 0)).mean() > 0.5):
                state = "idle"
            elif gap_flags.any() and null_frac > 0.3:
                state = "missing"
            elif zero_frac > 0.1:
                state = "sparse"
            else:
                state = "ok"

            state_series = pd.Series(state, index=grp.index)
            state_frames.append((grp.index, feat, state_series))

    # Write back into df. Build every new column up front and concat once,
    # rather than inserting columns one at a time in a loop (which
    # fragments the DataFrame and is O(n^2) — see pandas PerformanceWarning
    # this replaces).
    new_cols = {}
    for feat in feature_cols:
        new_cols.setdefault(f"{feat}_data_state", pd.Series("ok", index=df.index, dtype=object))
        new_cols.setdefault(f"{feat}_gap_flag", pd.Series(False, index=df.index, dtype=bool))

    for idx, feat, series in state_frames:
        new_cols[f"{feat}_data_state"].loc[idx] = series.values

    for idx, feat, series in gap_frames:
        new_cols[f"{feat}_gap_flag"].loc[idx] = series.values

    # Drop any of these columns that already existed on df (defensive —
    # matches the previous "if col_name not in df.columns" behavior) so
    # concat doesn't create duplicate column names.
    existing = [c for c in new_cols if c in df.columns]
    if existing:
        df = df.drop(columns=existing)

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df


def _accumulate_documents(documents, grouped, raw_metrics, resource_tags):
    """Fold one batch of raw documents into the running (grouped, raw_metrics,
    resource_tags) accumulators. Identical logic to the first loop that used
    to live inside normalize_documents() -- pulled out so pipeline.py can
    call it once per batch from iter_document_batches() and drop the raw
    batch afterward, instead of needing the full document list in memory
    at once. The accumulators are the same size either way (one entry per
    unique (resource_id, category, timestamp)), so nothing about the
    resulting DataFrame changes -- only how the raw documents got there.
    """
    for doc in documents:
        category = normalize_category(doc.get("category"))
        metric = doc.get("metric")
        if not category or not metric:
            continue
        rid = str(resource_id(doc))
        if doc.get("tags") is not None:
            resource_tags[rid] = doc.get("tags")
        for ts, value, stat, bundle in metric_points(doc):
            key = (rid, category, ts)
            if key not in grouped:
                grouped[key] = {**base_context(doc), "timestamp": ts}
            if value is not None:
                existing = raw_metrics[key].get(metric)
                if existing is None:
                    raw_metrics[key][metric] = (value, stat, bundle)
                else:
                    existing_val, existing_stat, _ = existing
                    if stat == "average" and existing_stat != "average":
                        raw_metrics[key][metric] = (value, stat, bundle)


def _finalize_normalized(grouped, raw_metrics, resource_tags):
    """Second half of the old normalize_documents(): turn the accumulated
    (grouped, raw_metrics, resource_tags) into the final feature-complete
    DataFrame. Unchanged from before -- this still runs once, over the full
    accumulated set, so sorting and gap-flag computation (which need a
    resource's complete timeline, not a single batch's slice of it) behave
    exactly as they did before batching was introduced.
    """
    rows = []
    for key, row in grouped.items():
        category = row["category"]
        service_name = row["service_name"]
        rid = key[0]
        row["tags"] = resource_tags.get(rid, row.get("tags"))
        metrics = raw_metrics[key]

        for feature, provider_map in REGISTRY.get(category, {}).items():
            value = None
            stat_used = None
            available = False
            f_max = f_min = f_sample_count = None

            transforms = provider_map.get(service_name, {})
            for raw_name, transform in transforms.items():
                if raw_name in metrics and metrics[raw_name] is not None:
                    try:
                        raw_val, stat_used, bundle = metrics[raw_name]
                        value = transform(raw_val)
                        available = True
                        if bundle:
                            bmax = bundle.get("maximum")
                            bmin = bundle.get("minimum")
                            bsc  = bundle.get("sample_count")
                            f_max          = transform(bmax) if bmax is not None else None
                            f_min          = transform(bmin) if bmin is not None else None
                            f_sample_count = bsc
                        break
                    except (TypeError, ValueError):
                        value = None

            row[feature] = value
            row[f"{feature}_available"]     = available
            row[f"{feature}_stat"]          = stat_used
            row[f"{feature}_max"]           = f_max
            row[f"{feature}_min"]           = f_min
            row[f"{feature}_sample_count"]  = f_sample_count

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    # cpu_idle → cpu_pct derivation before any other processing
    _derive_cpu_from_idle(rows)

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["category", "resource_id", "timestamp"]).reset_index(drop=True)

    # Add data_state and gap_flag columns (vectorised per resource)
    df = _add_data_state_and_gap_flags(df)

    return df


def normalize_documents(documents):
    """Unchanged public behavior — kept for every caller that already loads
    its (bounded) document list in one shot: the predict path, cmd_inspect,
    load_normalized_for_resources, etc. Internally just calls the two phases
    back to back. Only cmd_train's full-collection path (pipeline.py) calls
    _accumulate_documents/_finalize_normalized directly, batch by batch.
    """
    grouped, raw_metrics, resource_tags = {}, defaultdict(dict), {}
    _accumulate_documents(documents, grouped, raw_metrics, resource_tags)
    return _finalize_normalized(grouped, raw_metrics, resource_tags)
