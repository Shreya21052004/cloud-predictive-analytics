"""data_quality.py — Data quality checks for the predictive pipeline.

Covers:
  - Deduplication via whole-document content hash (SHA-256, excludes _id)
  - Late arrival handling (tolerance window before normalization)
  - Stale data / freshness gate (suppress predictions for silent resources)
  - Minimum history guard (don't predict on < 24h of data)
  - Output schema validation (risk/confidence range checks)
  - Idempotent write key builder (resource_id + prediction_timestamp)
  - Drift detection trigger (anomaly rate spike → flag for retraining)
  - Feedback loop staleness check
  - Alert state-change dedup (only alert on HIGH/CRITICAL state transitions)
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple  # noqa: F401 (Optional used in _resolve_now signature)

import pandas as pd

from .config import PipelineConfig

# ---------------------------------------------------------------------------
# Constants (override via PipelineConfig where applicable)
# ---------------------------------------------------------------------------

_DEDUP_KEY_COLS = ("resource_id", "metric", "timestamp")
_FRESHNESS_MAX_AGE_MINUTES = 60
_MIN_HISTORY_HOURS = 24.0
_LATE_ARRIVAL_TOLERANCE_MINUTES = 15
_DRIFT_ANOMALY_RATE_THRESHOLD = 0.40
_FEEDBACK_STALENESS_DAYS = 2


# ===========================================================================
# 1. Deduplication
# ===========================================================================

def _canonical_doc_hash(doc: dict) -> str:
    """Deterministic content hash of a raw Mongo document.

    Hashes the WHOLE document (every field, including nested metric_value
    lists) so that two documents are only ever treated as duplicates when
    they are byte-for-byte identical in content. Any difference — a
    different reading inside a batched metric_value list, a different
    stat, a different tag — produces a different hash and is kept.

    `_id` is excluded: MongoDB assigns it a fresh unique value per insert,
    so keeping it in the hash would make every document unique and dedup
    would never fire, even on true byte-identical re-inserts.
    """
    doc_copy = {k: v for k, v in doc.items() if k != "_id"}
    canonical = json.dumps(doc_copy, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def dedup_documents(documents: List[dict]) -> Tuple[List[dict], int]:
    """Remove exact duplicate raw Mongo documents before normalization.

    Dedup key: a SHA-256 hash of the entire document (minus `_id`) — see
    `_canonical_doc_hash`. Two documents are duplicates only if every field
    matches exactly, including individual points inside a batched
    metric_value list. A document that merely shares the same resource,
    metric, and batch timestamp as another — but carries different
    readings — is NOT considered a duplicate and is kept.

    When duplicates exist, keep the first occurrence (stable sort assumed).

    Returns (deduped_documents, n_dropped).
    """
    seen = set()
    out = []
    dropped = 0

    for doc in documents:
        h = _canonical_doc_hash(doc)
        if h in seen:
            dropped += 1
        else:
            seen.add(h)
            out.append(doc)

    return out, dropped


def dedup_documents_streaming(documents: List[dict], seen: set) -> Tuple[List[dict], int]:
    """Same rule as dedup_documents(), but `seen` is supplied by the caller
    and mutated in place, so it can be reused across successive batches from
    iter_document_batches() without ever holding more than one batch of raw
    documents in memory at a time. Only the (small) set of hex digest
    strings persists across batches — not the documents themselves.

    Returns (deduped_batch, n_dropped_in_this_batch).
    """
    out = []
    dropped = 0
    for doc in documents:
        h = _canonical_doc_hash(doc)
        if h in seen:
            dropped += 1
        else:
            seen.add(h)
            out.append(doc)
    return out, dropped


# ===========================================================================
# 2. Late arrival handling
# ===========================================================================

def _resolve_now(reference_time: Optional[datetime] = None) -> datetime:
    """Return the reference 'now' for all time-relative checks.

    Passing reference_time lets the pipeline evaluate historical/static
    datasets without treating every resource as stale.  None → wall clock
    (correct behaviour for real-time streaming data).
    """
    if reference_time is not None:
        if reference_time.tzinfo is None:
            return reference_time.replace(tzinfo=timezone.utc)
        return reference_time
    return datetime.now(timezone.utc)


def adjust_for_late_arrival(
    ts: datetime,
    tolerance_minutes: int = _LATE_ARRIVAL_TOLERANCE_MINUTES,
    reference_time: Optional[datetime] = None,
) -> datetime:
    """Snap a metric timestamp forward by tolerance_minutes if it falls in the
    late-arrival window relative to now.

    Cloud providers can deliver metrics 5–15 min late. Without this, a metric
    timestamped at T-12min arriving now looks like an ingestion-boundary gap
    and triggers false anomalies.

    Rule: if  now - tolerance  <  ts  <  now, treat ts as 'effectively now'
    (i.e. don't flag it as a gap). We do this by returning `now` so the
    normalizer sees a contiguous series rather than a tail-gap.
    """
    now = _resolve_now(reference_time)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    delta_minutes = (now - ts).total_seconds() / 60.0
    if 0 < delta_minutes <= tolerance_minutes:
        return now
    return ts


# ===========================================================================
# 3. Stale data / freshness gate
# ===========================================================================

def resource_last_seen(df: pd.DataFrame) -> Dict[Tuple[str, str], datetime]:
    """Return {(resource_id, category): last_timestamp} for every resource."""
    if df.empty or "timestamp" not in df.columns:
        return {}
    result = {}
    for (rid, cat), grp in df.groupby(["resource_id", "category"], sort=False):
        ts = grp["timestamp"].dropna().max()
        if not pd.isna(ts):
            result[(str(rid), str(cat))] = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
    return result


def is_resource_fresh(
    last_seen: datetime,
    max_age_minutes: int = _FRESHNESS_MAX_AGE_MINUTES,
    reference_time: Optional[datetime] = None,
) -> bool:
    """Return True if the resource reported within max_age_minutes of reference_time."""
    now = _resolve_now(reference_time)
    if last_seen.tzinfo is None:
        last_seen = last_seen.replace(tzinfo=timezone.utc)
    age_minutes = (now - last_seen).total_seconds() / 60.0
    return age_minutes <= max_age_minutes


def stale_resources(
    df: pd.DataFrame,
    max_age_minutes: int = _FRESHNESS_MAX_AGE_MINUTES,
    reference_time: Optional[datetime] = None,
) -> Dict[Tuple[str, str], float]:
    """Return {(resource_id, category): age_minutes} for stale resources.

    reference_time sets the 'now' anchor.  For static/historical datasets pass
    the dataset's max(to_date) so resources don't appear stale relative to
    today's wall clock.
    """
    last_seen = resource_last_seen(df)
    now = _resolve_now(reference_time)
    result = {}
    for key, ts in last_seen.items():
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age = (now - ts).total_seconds() / 60.0
        if age > max_age_minutes:
            result[key] = round(age, 1)
    return result


# ===========================================================================
# 4. Minimum history guard
# ===========================================================================

def resource_history_hours(df: pd.DataFrame) -> Dict[Tuple[str, str], float]:
    """Return {(resource_id, category): history_span_hours}."""
    if df.empty or "timestamp" not in df.columns:
        return {}
    result = {}
    for (rid, cat), grp in df.groupby(["resource_id", "category"], sort=False):
        ts = grp["timestamp"].dropna()
        if len(ts) < 2:
            result[(str(rid), str(cat))] = 0.0
        else:
            span = (ts.max() - ts.min()).total_seconds() / 3600.0
            result[(str(rid), str(cat))] = round(float(span), 2)
    return result


def has_sufficient_history(
    resource_id: str,
    category: str,
    history_map: Dict[Tuple[str, str], float],
    min_hours: float = _MIN_HISTORY_HOURS,
) -> bool:
    """Return True if resource has >= min_hours of history."""
    return history_map.get((resource_id, category), 0.0) >= min_hours


# ===========================================================================
# 5. Output schema validation
# ===========================================================================

_VALID_SEVERITIES = {"INFO", "WARNING", "HIGH", "CRITICAL"}
_VALID_RISK_RANGE = (0.0, 100.0)
_VALID_CONFIDENCE_RANGE = (0.0, 1.0)


def validate_prediction(prediction: dict) -> List[str]:
    """Validate a single prediction document before writing to MongoDB.

    Returns a list of error strings (empty = valid).
    """
    errors = []
    pid = prediction.get("prediction_id", "<unknown>")

    # anomaly_score in [0, 1]
    score = prediction.get("anomaly_score")
    if score is not None:
        try:
            s = float(score)
            if math.isnan(s) or math.isinf(s):
                errors.append(f"[{pid}] anomaly_score is NaN/Inf")
            elif not (_VALID_CONFIDENCE_RANGE[0] <= s <= _VALID_CONFIDENCE_RANGE[1]):
                errors.append(f"[{pid}] anomaly_score {s} out of [0,1]")
        except (TypeError, ValueError):
            errors.append(f"[{pid}] anomaly_score not numeric: {score!r}")

    # risk_score in payload
    payload = prediction.get("payload", {})
    risk = payload.get("risk_score") if payload else None
    if risk is not None:
        try:
            r = float(risk)
            if math.isnan(r) or math.isinf(r):
                errors.append(f"[{pid}] risk_score is NaN/Inf")
            elif not (_VALID_RISK_RANGE[0] <= r <= _VALID_RISK_RANGE[1]):
                errors.append(f"[{pid}] risk_score {r} out of [0,100]")
        except (TypeError, ValueError):
            errors.append(f"[{pid}] risk_score not numeric: {risk!r}")

    # severity must be a known label
    alert = prediction.get("alert", {})
    sev = alert.get("severity") if alert else None
    if sev and sev not in _VALID_SEVERITIES:
        errors.append(f"[{pid}] unknown severity: {sev!r}")

    # required top-level fields
    for field in ("resource_id", "category", "prediction_timestamp"):
        if not prediction.get(field):
            errors.append(f"[{pid}] missing required field: {field}")

    return errors


def validate_predictions(predictions: List[dict]) -> Tuple[List[dict], List[str]]:
    """Validate a list of predictions. Returns (valid_preds, all_errors)."""
    valid = []
    all_errors = []
    for p in predictions:
        errs = validate_prediction(p)
        if errs:
            all_errors.extend(errs)
        else:
            valid.append(p)
    return valid, all_errors


# ===========================================================================
# 6. Idempotent write key
# ===========================================================================

def idempotent_key(resource_id: str, prediction_timestamp: str, metric_combination: str = "") -> str:
    """Deterministic key for upsert: prevents double-writes on pipeline restart.

    Use as the `filter` key in MongoDB replace_one(..., upsert=True).

    BUG FIX: prediction_timestamp is second-precision (see prediction.now_iso())
    and identical for every prediction generated within one pipeline run. Once
    a resource can produce more than one prediction per run (one per distinct
    metric combination — see prediction.rows_per_metric_combination), keying
    only on (resource_id, prediction_timestamp) made every one of those
    predictions collide on the same upsert filter, so only the last one
    written in the batch survived — silently re-introducing the "only one
    prediction per resourceId" problem at the write layer even after the
    selection logic was fixed. metric_combination disambiguates them.
    """
    raw = f"{resource_id}::{metric_combination}::{prediction_timestamp}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


# ===========================================================================
# 7. Drift detection / retraining trigger
# ===========================================================================

def check_drift_trigger(
    predictions: List[dict],
    threshold: float = _DRIFT_ANOMALY_RATE_THRESHOLD,
) -> Dict[str, dict]:
    """Check if anomaly rate per category exceeds threshold.

    Returns {category: {"anomaly_rate": float, "retrain_flag": bool, "n": int}}.
    A retrain_flag=True means this category's predictions should be reviewed
    and retraining considered.
    """
    from collections import defaultdict
    category_counts: Dict[str, dict] = defaultdict(lambda: {"total": 0, "anomalous": 0})

    for p in predictions:
        cat = p.get("category", "Unknown")
        category_counts[cat]["total"] += 1
        if p.get("is_anomalous"):
            category_counts[cat]["anomalous"] += 1

    result = {}
    for cat, counts in category_counts.items():
        total = counts["total"]
        anomalous = counts["anomalous"]
        rate = anomalous / total if total > 0 else 0.0
        result[cat] = {
            "n": total,
            "anomalous": anomalous,
            "anomaly_rate": round(rate, 3),
            "retrain_flag": rate >= threshold,
        }
    return result


# ===========================================================================
# 8. Feedback loop staleness check
# ===========================================================================

def check_feedback_staleness(
    predictions_col,
    staleness_days: int = _FEEDBACK_STALENESS_DAYS,
) -> dict:
    """Check if the feedback loop has been running.

    Returns {"stale": bool, "last_feedback_at": str|None, "age_days": float|None}.
    A stale=True result means nightly feedback.py hasn't run recently and
    resource profiles may be drifting silently.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=staleness_days)).isoformat()
    try:
        recent = predictions_col.find_one(
            {"feedback.outcome_recorded": True, "feedback.outcome_at": {"$gte": cutoff}},
            {"feedback.outcome_at": 1},
            sort=[("feedback.outcome_at", -1)],
        )
        if recent:
            last_at = recent.get("feedback", {}).get("outcome_at")
            ts = pd.Timestamp(last_at)
            age_days = (pd.Timestamp.now(tz="UTC") - ts).total_seconds() / 86400
            return {"stale": False, "last_feedback_at": last_at, "age_days": round(age_days, 2)}
        else:
            return {"stale": True, "last_feedback_at": None, "age_days": None}
    except Exception as e:
        return {"stale": True, "last_feedback_at": None, "age_days": None, "error": str(e)}


# ===========================================================================
# 9. Alert state-change dedup
# ===========================================================================

def should_alert(
    resource_id: str,
    category: str,
    new_severity: str,
    alert_state_col,
) -> Tuple[bool, str]:
    """Return (should_fire, reason) for alert dedup.

    Only fires when severity changes from the last recorded state.
    Suppresses repeated HIGH/CRITICAL alerts for the same resource to avoid
    flooding downstream systems.

    alert_state_col: a MongoDB collection with schema:
      { resource_id, category, last_severity, last_alerted_at }
    """
    key = {"resource_id": str(resource_id), "category": str(category)}
    try:
        existing = alert_state_col.find_one(key, {"last_severity": 1})
    except Exception:
        existing = None

    last_severity = existing.get("last_severity") if existing else None

    if last_severity == new_severity:
        return False, f"no_change (still {new_severity})"

    # Update state
    try:
        alert_state_col.update_one(
            key,
            {"$set": {
                "last_severity": new_severity,
                "last_alerted_at": datetime.now(timezone.utc).isoformat(),
            }},
            upsert=True,
        )
    except Exception:
        pass  # Don't block prediction write if state update fails

    return True, f"state_change ({last_severity} → {new_severity})"
