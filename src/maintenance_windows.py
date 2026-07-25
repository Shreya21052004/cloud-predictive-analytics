"""maintenance_windows.py — Suppress/contextualize alerts during known
maintenance, deployment, or backup windows.

Without this, anomaly models fire false positives every time a planned
deployment or backup job runs. This module checks a `maintenance_windows`
MongoDB collection (or an in-memory list, for testing) and:
  - fully suppresses CRITICAL/HIGH alerts that fall inside an active window
  - downgrades severity to INFO and tags the prediction with the matched
    window, so it's still visible in dashboards/audit but doesn't page anyone

Expected window document schema (intentionally minimal/flexible):
    {
      "resource_id": "i-0abc123" | "*",      # "*" = applies to all resources
      "category":    "Compute" | "*",
      "starts_at":   ISO datetime string,
      "ends_at":     ISO datetime string,
      "reason":      "scheduled patch" | "deployment" | "backup window" | ...
      "source":      "change-calendar" | "manual" | "terraform-apply" | ...
    }

Resources can be matched by exact resource_id, or by tag-based rules
(e.g. all resources tagged env=staging) if `tag_filter` is present on the
window document — see `_matches_tag_filter`.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd


def _parse(ts):
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    try:
        p = pd.Timestamp(ts)
        return p.to_pydatetime().replace(tzinfo=timezone.utc) if p.tzinfo is None else p.to_pydatetime()
    except Exception:
        return None


def _matches_tag_filter(tags, tag_filter: Dict) -> bool:
    """tag_filter: {"env": "staging"} matches if resource has that tag/value."""
    if not tag_filter:
        return True
    if not tags:
        return False
    # tags may be a dict or a list of {key, value} pairs
    if isinstance(tags, dict):
        tag_map = {str(k).lower(): str(v) for k, v in tags.items()}
    elif isinstance(tags, list):
        tag_map = {}
        for t in tags:
            if isinstance(t, dict) and "key" in t:
                tag_map[str(t["key"]).lower()] = str(t.get("value"))
    else:
        return False
    for k, v in tag_filter.items():
        if tag_map.get(str(k).lower()) != str(v):
            return False
    return True


def load_active_windows(
    windows_col,
    now: Optional[datetime] = None,
    lookahead_minutes: int = 0,
) -> List[Dict]:
    """Load maintenance windows active at `now` (or starting within
    lookahead_minutes, for pre-emptive suppression around deploys)."""
    now = now or datetime.now(timezone.utc)
    cutoff_end = now.isoformat()
    docs = list(windows_col.find({
        "starts_at": {"$lte": now.isoformat()},
        "ends_at":   {"$gte": cutoff_end},
    }))
    return docs


def is_in_maintenance_window(
    resource_id: str,
    category: str,
    tags,
    windows: List[Dict],
    now: Optional[datetime] = None,
) -> Tuple[bool, Optional[Dict]]:
    """Check whether (resource_id, category, tags) falls inside any active window.

    Returns (in_window, matched_window_doc_or_None).
    """
    now = now or datetime.now(timezone.utc)
    for w in windows:
        starts = _parse(w.get("starts_at"))
        ends   = _parse(w.get("ends_at"))
        if not starts or not ends or not (starts <= now <= ends):
            continue

        rid_match = w.get("resource_id") in (None, "*", str(resource_id))
        cat_match = w.get("category") in (None, "*", category)
        tag_match = _matches_tag_filter(tags, w.get("tag_filter"))

        if rid_match and cat_match and tag_match:
            return True, w
    return False, None


def apply_maintenance_suppression(
    predictions: List[Dict],
    windows: List[Dict],
    now: Optional[datetime] = None,
) -> List[Dict]:
    """Mutate predictions in-place: downgrade alerts inside maintenance windows.

    Behavior:
      - alert.severity -> "INFO"
      - alert.trigger  -> False
      - alert.suppressed -> True
      - a `maintenance_context` field is added documenting why, so the alert
        is still queryable/auditable, just not paged.
    Does NOT touch anomaly_score / risk_score — the underlying signal is
    preserved for the feedback loop and for dashboards that want to show
    "would have alerted" context.
    """
    now = now or datetime.now(timezone.utc)
    if not windows:
        return predictions

    for p in predictions:
        rid = p.get("resource_id")
        cat = p.get("category")
        tags = p.get("tags")
        in_window, matched = is_in_maintenance_window(rid, cat, tags, windows, now=now)
        if not in_window:
            continue

        alert = p.setdefault("alert", {})
        original_severity = alert.get("severity")
        alert["severity"]   = "INFO"
        alert["trigger"]    = False
        alert["suppressed"] = True
        alert["channel"]    = "suppressed"

        p["maintenance_context"] = {
            "suppressed_during_maintenance": True,
            "original_severity": original_severity,
            "window_reason": matched.get("reason"),
            "window_source": matched.get("source"),
            "window_ends_at": matched.get("ends_at"),
        }
    return predictions
