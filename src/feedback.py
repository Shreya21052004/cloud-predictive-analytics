"""feedback.py — Feedback loop: close predictions against incident/change records.

This module is the single most valuable addition for long-term accuracy.
Without it the system never learns whether its predictions correlate with
real incidents — thresholds can't improve, false-positive rates are unknown,
and the classifier labels stay "weak rules forever".

How it works:
  1. A nightly job calls `run_feedback_pass()`.
  2. It loads all predictions older than MIN_AGE_H hours that have
     outcome_recorded = False.
  3. For each prediction it checks an incidents collection for a matching
     (resource_id, time_window) record.
  4. Writes outcome_label: 'true_positive' | 'false_positive' | 'unknown'
     back to the prediction document.
  5. Computes per-category precision/recall from the last 30d of labelled
     predictions and logs them.
  6. (Optional) Exports labelled rows as training data for future supervised
     model iterations.

Incident matching:
  A prediction is a TRUE POSITIVE if:
    - An incident record with the same resource_id exists
    - The incident opened_at is within TP_WINDOW_H hours AFTER prediction_timestamp
  A prediction is a FALSE POSITIVE if:
    - No such incident was found
    - AND the prediction is older than FP_TIMEOUT_H hours

Assumptions about the incidents collection schema:
  { resource_id, opened_at (ISO datetime), closed_at, severity, description }
  This is intentionally flexible — any collection that has those fields works.
  Set incidents_collection=None to skip TP matching and only record FPs.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import pandas as pd

TP_WINDOW_H   = 6    # prediction → incident must open within this window
FP_TIMEOUT_H  = 24   # prediction older than this with no incident = FP
MIN_AGE_H     = 6    # don't label predictions younger than this (incident may not be recorded yet)
LOOKBACK_DAYS = 30   # window for precision/recall stats


def _now():
    return datetime.now(timezone.utc)


def _parse(ts):
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts
    try:
        return pd.Timestamp(ts).to_pydatetime().replace(tzinfo=timezone.utc)
    except Exception:
        return None


def load_unlabelled_predictions(collection, min_age_h=MIN_AGE_H):
    """Load prediction docs that haven't been labelled yet and are old enough."""
    cutoff = _now() - timedelta(hours=min_age_h)
    docs = list(collection.find(
        {
            "feedback.outcome_recorded": False,
            "prediction_timestamp": {"$lt": cutoff.isoformat()},
        },
        {"_id": 1, "resource_id": 1, "prediction_timestamp": 1,
         "category": 1, "alert.severity": 1},
    ))
    return docs


def load_incidents_in_window(incidents_col, resource_id, after, before):
    """Return incident docs for a resource_id within a time window."""
    if incidents_col is None:
        return []
    return list(incidents_col.find({
        "resource_id": str(resource_id),
        "opened_at": {
            "$gte": after.isoformat(),
            "$lte": before.isoformat(),
        },
    }, {"_id": 1, "opened_at": 1, "severity": 1, "description": 1}))


def label_prediction(predictions_col, doc_id, outcome_label, matched_incident_id=None):
    """Write outcome_label back to the prediction document."""
    predictions_col.update_one(
        {"_id": doc_id},
        {"$set": {
            "feedback.outcome_recorded": True,
            "feedback.outcome_label":    outcome_label,
            "feedback.outcome_at":       _now().isoformat(),
            "feedback.matched_incident": str(matched_incident_id) if matched_incident_id else None,
        }},
    )


def run_feedback_pass(
    predictions_col,
    incidents_col=None,
    tp_window_h=TP_WINDOW_H,
    fp_timeout_h=FP_TIMEOUT_H,
) -> Dict:
    """Label all mature unlabelled predictions. Returns summary dict."""
    docs = load_unlabelled_predictions(predictions_col)
    if not docs:
        return {"labelled": 0, "true_positives": 0, "false_positives": 0, "unknown": 0}

    tp = fp = unknown = 0
    for doc in docs:
        pred_ts = _parse(doc.get("prediction_timestamp"))
        if pred_ts is None:
            continue
        rid = doc.get("resource_id")
        age_h = (_now() - pred_ts).total_seconds() / 3600.0

        if incidents_col is not None:
            window_end = pred_ts + timedelta(hours=tp_window_h)
            incidents = load_incidents_in_window(incidents_col, rid, pred_ts, window_end)
            if incidents:
                label_prediction(predictions_col, doc["_id"], "true_positive",
                                 matched_incident_id=incidents[0].get("_id"))
                tp += 1
                continue

        if age_h >= fp_timeout_h:
            label_prediction(predictions_col, doc["_id"], "false_positive")
            fp += 1
        else:
            # Too young to declare FP — revisit on next nightly pass
            unknown += 1

    return {
        "labelled":       tp + fp,
        "true_positives": tp,
        "false_positives": fp,
        "unknown":        unknown,
        "processed":      len(docs),
    }


def compute_performance_metrics(predictions_col, lookback_days=LOOKBACK_DAYS) -> Dict:
    """Precision, recall, FP rate from the last `lookback_days` of labelled predictions."""
    since = (_now() - timedelta(days=lookback_days)).isoformat()
    labelled = list(predictions_col.find(
        {
            "feedback.outcome_recorded": True,
            "feedback.outcome_at":       {"$gte": since},
        },
        {"feedback.outcome_label": 1, "category": 1, "alert.severity": 1},
    ))
    if not labelled:
        return {"error": "no_labelled_predictions_in_window", "window_days": lookback_days}

    df = pd.DataFrame([
        {
            "outcome":   d["feedback"]["outcome_label"],
            "category":  d.get("category", "Unknown"),
            "severity":  d.get("alert", {}).get("severity", "UNKNOWN"),
        }
        for d in labelled
    ])

    total = len(df)
    tp_count = (df["outcome"] == "true_positive").sum()
    fp_count = (df["outcome"] == "false_positive").sum()
    precision = tp_count / max(1, tp_count + fp_count)
    # Recall can't be computed without knowing total real incidents, but alert_rate
    # is a useful proxy for how aggressively the model is alerting.
    alert_rate = (tp_count + fp_count) / max(1, total)

    by_category = {}
    for cat, grp in df.groupby("category"):
        cat_tp = (grp["outcome"] == "true_positive").sum()
        cat_fp = (grp["outcome"] == "false_positive").sum()
        by_category[cat] = {
            "precision":   round(cat_tp / max(1, cat_tp + cat_fp), 3),
            "tp":          int(cat_tp),
            "fp":          int(cat_fp),
            "total":       len(grp),
        }

    return {
        "window_days":       lookback_days,
        "total_labelled":    int(total),
        "true_positives":    int(tp_count),
        "false_positives":   int(fp_count),
        "overall_precision": round(float(precision), 3),
        "alert_rate":        round(float(alert_rate), 3),
        "by_category":       by_category,
    }


def export_training_labels(predictions_col, output_path: str, lookback_days=90):
    """Export labelled prediction rows as a CSV for supervised re-training.

    The resulting CSV can be joined back to the normalized DataFrame by
    (resource_id, prediction_timestamp) to add real outcome labels.
    """
    since = (_now() - timedelta(days=lookback_days)).isoformat()
    labelled = list(predictions_col.find(
        {
            "feedback.outcome_recorded": True,
            "feedback.outcome_label":    {"$in": ["true_positive", "false_positive"]},
            "feedback.outcome_at":       {"$gte": since},
        },
        {
            "_id": 0,
            "resource_id": 1, "category": 1, "prediction_timestamp": 1,
            "anomaly_score": 1, "alert.severity": 1,
            "feedback.outcome_label": 1,
        },
    ))
    if not labelled:
        return 0
    rows = [
        {
            "resource_id":          d.get("resource_id"),
            "category":             d.get("category"),
            "prediction_timestamp": d.get("prediction_timestamp"),
            "anomaly_score":        d.get("anomaly_score"),
            "severity":             d.get("alert", {}).get("severity"),
            "outcome_label":        d["feedback"]["outcome_label"],
            "is_true_positive":     int(d["feedback"]["outcome_label"] == "true_positive"),
        }
        for d in labelled
    ]
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return len(rows)
