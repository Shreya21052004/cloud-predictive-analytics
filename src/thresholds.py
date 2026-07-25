"""src/thresholds.py — user-defined per-resource thresholds, breach
counting, and "when will this next breach" prediction.

This is deliberately separate from the hardcoded severity thresholds
already baked into prediction.py's failure_predictions (CPU>80%, storage>80%,
etc. — those are fixed, built-in rules for a handful of known metrics).
This module is for thresholds an operator sets *manually*, per resource,
on whichever metric they care about, e.g. "alert me on resource X's
network_throughput_mbps once it crosses 900".

Two collections:
  - 22resource_thresholds   one doc per (resource_id, metric_name) rule,
                             set by an operator. Source of truth for what
                             to check.
  - 22threshold_breach_log  append-only event log, one doc per breach
                             *observation* (idempotent per prediction, so
                             re-running a cycle over the same prediction
                             doesn't double-count). This is what
                             "how many times has it breached" counts.

The rolling state (breach_count, last_breached_at, predicted_next_breach)
is recomputed from the breach log + the latest prediction each time
evaluate_resource() runs, and cached onto a third small collection
(22resource_breach_state) so callers (LLM explanation, UI) can read it
with a single lookup instead of re-aggregating the log every time.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

from src.mongo_io import get_collection, _bson_safe

THRESHOLDS_COLLECTION = "22resource_thresholds"
BREACH_LOG_COLLECTION = "22threshold_breach_log"
BREACH_STATE_COLLECTION = "22resource_breach_state"

_OPERATORS = {
    "gt":  lambda v, t: v > t,
    "gte": lambda v, t: v >= t,
    "lt":  lambda v, t: v < t,
    "lte": lambda v, t: v <= t,
}

# (field_on_behavioral_forecast, hours_from_now) — current_value is "now"
# (offset 0). Order matters: must be ascending in time for interpolation.
_HORIZONS = [
    ("current_value", 0),
    ("forecast_1h", 1),
    ("forecast_6h", 6),
    ("forecast_24h", 24),
    ("forecast_7d", 24 * 7),
    ("forecast_30d", 24 * 30),
]


# ---------------------------------------------------------------------------
# Rule CRUD
# ---------------------------------------------------------------------------

def set_threshold(mongo_uri, db_name, resource_id, metric_name, operator, value,
                   category=None, project=None, set_by=None):
    """Create or update the threshold rule for (resource_id, metric_name).
    One active rule per (resource_id, metric_name) — setting again replaces
    the previous value/operator for that pair rather than adding a second
    rule, since "the threshold for X on this resource" should be singular.
    """
    if operator not in _OPERATORS:
        raise ValueError(f"operator must be one of {sorted(_OPERATORS)}, got {operator!r}")
    col = get_collection(mongo_uri, db_name, THRESHOLDS_COLLECTION)
    now = datetime.now(timezone.utc).isoformat()
    doc = {
        "resource_id": resource_id,
        "metric_name": metric_name,
        "operator": operator,
        "value": float(value),
        "category": category,
        "project": project,
        "set_by": set_by,
        "updated_at": now,
    }
    col.update_one(
        {"resource_id": resource_id, "metric_name": metric_name},
        {"$set": doc, "$setOnInsert": {"created_at": now}},
        upsert=True,
    )
    return doc


def delete_threshold(mongo_uri, db_name, resource_id, metric_name):
    col = get_collection(mongo_uri, db_name, THRESHOLDS_COLLECTION)
    result = col.delete_one({"resource_id": resource_id, "metric_name": metric_name})
    return result.deleted_count


def get_thresholds_for_resource(mongo_uri, db_name, resource_id):
    col = get_collection(mongo_uri, db_name, THRESHOLDS_COLLECTION)
    return list(col.find({"resource_id": resource_id}))


def list_thresholds(mongo_uri, db_name, project=None, category=None):
    col = get_collection(mongo_uri, db_name, THRESHOLDS_COLLECTION)
    query = {}
    if project:
        query["project"] = project
    if category:
        query["category"] = category
    return list(col.find(query))


# ---------------------------------------------------------------------------
# Breach evaluation
# ---------------------------------------------------------------------------

def _value_for_horizon(forecast, field):
    v = forecast.get(field)
    return None if v is None else float(v)


def predict_next_breach(forecast, operator, threshold_value, prediction_timestamp):
    """Interpolate the earliest time the metric is projected to cross
    threshold_value, using the same current_value/forecast_1h.../forecast_30d
    points prediction.py already computed (no new modeling — just reads the
    existing forecast curve).

    Returns (eta_iso_str, reasoning) or (None, reasoning) if either already
    breached (eta = now) or not projected to cross within the 30-day horizon
    the forecaster produces.
    """
    check = _OPERATORS[operator]
    points = []
    for field, hours in _HORIZONS:
        v = _value_for_horizon(forecast, field)
        if v is not None:
            points.append((hours, v))
    if not points:
        return None, "no forecast values available to project from"

    try:
        base_ts = datetime.fromisoformat(prediction_timestamp.replace("Z", "+00:00"))
    except (TypeError, ValueError, AttributeError):
        base_ts = datetime.now(timezone.utc)

    h0, v0 = points[0]
    if check(v0, threshold_value):
        return base_ts.isoformat(), "already breaching as of the current value"

    for (h1, v1), (h2, v2) in zip(points, points[1:]):
        breached_at_1 = check(v1, threshold_value)
        breached_at_2 = check(v2, threshold_value)
        if not breached_at_1 and breached_at_2 and v2 != v1:
            frac = (threshold_value - v1) / (v2 - v1)
            frac = max(0.0, min(1.0, frac))
            cross_hours = h1 + frac * (h2 - h1)
            eta = base_ts + timedelta(hours=cross_hours)
            return eta.isoformat(), f"interpolated crossing between the {h1}h and {h2}h forecast points"

    return None, "not projected to cross the threshold within the 30-day forecast horizon"


def _breach_event_key(resource_id, metric_name, prediction_timestamp):
    # One breach event per (resource, metric, prediction cycle) — reprocessing
    # the same prediction (e.g. pipeline restart re-reading a window) must not
    # inflate the breach count.
    raw = f"{resource_id}::{metric_name}::{prediction_timestamp}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def evaluate_resource(mongo_uri, db_name, prediction):
    """Check every threshold rule set for this resource against its latest
    prediction. For each rule: log a breach event (idempotent) if the
    *current* value breaches, and refresh the rolling breach-state doc
    (breach_count, last_breached_at, predicted_next_breach) regardless of
    whether it's currently breaching, since "not breaching now but will be
    in 6h" is exactly the useful signal this exists to surface.

    Returns list of the resource's current breach-state docs (one per
    metric with a rule), so callers (LLM explanation) can attach them
    without a second Mongo round trip.
    """
    resource_id = prediction.get("resource_id")
    if not resource_id:
        return []
    rules = get_thresholds_for_resource(mongo_uri, db_name, resource_id)
    if not rules:
        return []

    forecast = prediction.get("behavioral_forecast") or {}
    prediction_timestamp = prediction.get("prediction_timestamp") or datetime.now(timezone.utc).isoformat()

    log_col = get_collection(mongo_uri, db_name, BREACH_LOG_COLLECTION)
    state_col = get_collection(mongo_uri, db_name, BREACH_STATE_COLLECTION)
    states = []

    for rule in rules:
        metric_name = rule["metric_name"]
        operator = rule["operator"]
        threshold_value = rule["value"]
        current_value = _value_for_horizon(forecast, "current_value")

        currently_breaching = (
            current_value is not None and _OPERATORS[operator](current_value, threshold_value)
        )
        if currently_breaching:
            event = {
                "_id": _breach_event_key(resource_id, metric_name, prediction_timestamp),
                "resource_id": resource_id,
                "metric_name": metric_name,
                "operator": operator,
                "threshold_value": threshold_value,
                "observed_value": current_value,
                "prediction_timestamp": prediction_timestamp,
                "breached_at": datetime.now(timezone.utc).isoformat(),
            }
            try:
                log_col.insert_one(_bson_safe(event))
            except Exception:
                pass  # duplicate _id == already logged this cycle's breach, that's fine

        breach_count = log_col.count_documents({"resource_id": resource_id, "metric_name": metric_name})
        last_event = log_col.find_one(
            {"resource_id": resource_id, "metric_name": metric_name},
            sort=[("breached_at", -1)],
        )
        next_breach_eta, next_breach_reason = predict_next_breach(
            forecast, operator, threshold_value, prediction_timestamp,
        )

        state = {
            "resource_id": resource_id,
            "metric_name": metric_name,
            "operator": operator,
            "threshold_value": threshold_value,
            "currently_breaching": currently_breaching,
            "current_value": current_value,
            "breach_count": breach_count,
            "last_breached_at": last_event["breached_at"] if last_event else None,
            "predicted_next_breach": next_breach_eta,
            "predicted_next_breach_reason": next_breach_reason,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        state_col.update_one(
            {"resource_id": resource_id, "metric_name": metric_name},
            {"$set": _bson_safe(state)},
            upsert=True,
        )
        states.append(state)

    return states


def get_breach_state_for_resource(mongo_uri, db_name, resource_id):
    col = get_collection(mongo_uri, db_name, BREACH_STATE_COLLECTION)
    return list(col.find({"resource_id": resource_id}))
