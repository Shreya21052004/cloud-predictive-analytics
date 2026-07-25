"""cross_domain.py — Correlate simultaneous anomalies across categories on
the SAME underlying infrastructure to surface composite incidents.

Problem this solves: each category model (Compute, Network, Storage,
Container, Databases) runs independently in prediction.py / anomaly.py.
A CPU spike + network spike + disk queue spike on the *same host/cluster*
at the *same time* is a much stronger incident signal than any one alone —
and is exactly the pattern that precedes most real outages. Independent
per-category thresholds miss this because no single category crosses its
own alert bar.

Approach:
  1. Group same-run predictions by a correlation key — `project` (a single
     project can span multiple cloud accounts/subscriptions across
     providers, so account_id alone under-groups; project is the boundary
     that actually represents "same infrastructure" here). Falls back to
     `account_id` only when a record has no project (legacy/unbackfilled
     data), so older documents don't silently drop out of correlation.
  2. Within each correlation group, count how many distinct categories
     have is_anomalous=True or alert.severity in (HIGH, CRITICAL).
  3. If >= MIN_DOMAINS_FOR_COMPOSITE categories are simultaneously
     anomalous, emit a synthetic composite_incident record with a higher
     confidence than any single-domain alert, and tag the contributing
     predictions with `composite_incident_id` so they can be displayed
     together in a single incident view instead of N separate pages.

This does not replace per-category models — it's a second pass over their
outputs, so it is cheap and has no dependency on retraining anything.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Tuple
from uuid import uuid4

MIN_DOMAINS_FOR_COMPOSITE = 2
HIGH_SEVERITIES = {"HIGH", "CRITICAL"}

# Correlation weight by category pair — some combinations are much more
# diagnostic than others. Used only to write a human-readable "likely_cause"
# hint, not to gate the composite trigger itself.
LIKELY_CAUSE_RULES = [
    (frozenset({"Compute", "Network"}),   "Resource under network-driven load — possible traffic surge or DDoS-adjacent pattern"),
    (frozenset({"Compute", "Storage"}),   "Compute saturation coinciding with storage I/O pressure — possible noisy-neighbour or runaway job"),
    (frozenset({"Storage", "Databases"}), "Storage pressure on a resource tied to a database — possible query-driven disk pressure or backup contention"),
    (frozenset({"Network", "Security"}),  "Network anomaly coinciding with a security signal — review for exfiltration or DDoS"),
    (frozenset({"Container", "Compute"}), "Node-level compute pressure with container scheduling impact — cluster may need scaling"),
    (frozenset({"Databases", "Compute"}), "Database CPU pressure coinciding with host compute saturation — check co-located workloads"),
]


def _correlation_key(p: Dict) -> Tuple[str, str]:
    """Build the key used to group predictions that likely share underlying
    infrastructure.

    A project can span multiple cloud accounts/subscriptions/compartments
    across different providers (e.g. Compute in AWS, Database in Azure for
    the same project), so account_id is too narrow a boundary on its own —
    it would never group those together since they're never in the same
    account. project is the level that actually ties them back to one
    piece of "infrastructure" for incident-correlation purposes.

    Falls back to account_id when project is missing (older/unbackfilled
    records), so those records still correlate within their own account
    rather than being silently excluded.
    """
    project = p.get("project")
    if project:
        return ("project", str(project))
    account = str(p.get("account_id") or "unknown_account")
    return ("account", account)


def _is_domain_anomalous(p: Dict) -> bool:
    if p.get("is_anomalous"):
        return True
    sev = (p.get("alert", {}) or {}).get("severity")
    return sev in HIGH_SEVERITIES


def _likely_cause(categories: set) -> str:
    for pair, msg in LIKELY_CAUSE_RULES:
        if pair.issubset(categories):
            return msg
    return f"Simultaneous anomalies across {', '.join(sorted(categories))} — investigate shared infrastructure/blast radius"


def detect_composite_incidents(
    predictions: List[Dict],
    min_domains: int = MIN_DOMAINS_FOR_COMPOSITE,
) -> List[Dict]:
    """Scan a batch of same-run predictions for cross-domain correlation.

    Returns a list of composite_incident dicts (NOT mutated into the input
    predictions — caller decides how to persist/route them). Each contributing
    prediction dict is mutated in-place to carry `composite_incident_id`.
    """
    groups: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for p in predictions:
        groups[_correlation_key(p)].append(p)

    incidents = []
    for (key_type, key_value), group in groups.items():
        anomalous = [p for p in group if _is_domain_anomalous(p)]
        categories_hit = {p.get("category") for p in anomalous if p.get("category")}

        if len(categories_hit) < min_domains:
            continue

        incident_id = f"composite_{datetime.now(timezone.utc).strftime('%Y%m%d')}_{uuid4().hex[:8]}"
        max_risk = max((p.get("dashboard", {}).get("display_score", 0) for p in anomalous), default=0)

        for p in anomalous:
            p["composite_incident_id"] = incident_id

        incidents.append({
            "composite_incident_id": incident_id,
            "project":                key_value if key_type == "project" else None,
            "account_id":             key_value if key_type == "account" else None,
            "correlation_unit":      f"{key_type}::{key_value}",
            "categories_involved":   sorted(categories_hit),
            "n_resources":           len(anomalous),
            "resource_ids":          [p.get("resource_id") for p in anomalous],
            "max_risk_score":        max_risk,
            "likely_cause":          _likely_cause(categories_hit),
            "detected_at":           datetime.now(timezone.utc).isoformat(),
            "severity":              "CRITICAL" if max_risk >= 75 else "HIGH",
            "recommendation":        (
                "Treat as a single incident — review the contributing resources together "
                "rather than triaging each category alert independently."
            ),
        })

    return incidents
