"""anomaly_clustering.py — Detect when many resources anomaly simultaneously,
indicating a platform/region/provider-wide issue rather than N independent
single-resource problems.

If 50 VMs across us-east-1 all spike at 14:02 UTC, that's almost certainly
a regional event (AZ degradation, a bad deploy pushed fleet-wide, a shared
dependency failing) — not 50 unrelated incidents. Without clustering, this
shows up as 50 separate pages and the on-call team has no way to see the
blast radius without manually correlating timestamps.

Approach: pure rule-based grouping (no ML needed) —
  group anomalous predictions from the same run by (provider, location/region,
  category), and flag any group where the anomalous fraction exceeds a
  threshold as a "fleet anomaly" rather than N individual ones.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Tuple
from uuid import uuid4

MIN_CLUSTER_SIZE = 5          # don't bother clustering below this count
MIN_ANOMALOUS_FRACTION = 0.30  # 30%+ of a group anomalous → likely shared cause


def _group_key(p: Dict) -> Tuple[str, str, str]:
    return (
        str(p.get("provider") or "unknown_provider"),
        str(p.get("location") or "unknown_location"),
        str(p.get("category") or "unknown_category"),
    )


def _is_anomalous(p: Dict) -> bool:
    if p.get("is_anomalous"):
        return True
    sev = (p.get("alert", {}) or {}).get("severity")
    return sev in ("HIGH", "CRITICAL")


def cluster_anomalies(
    predictions: List[Dict],
    min_cluster_size: int = MIN_CLUSTER_SIZE,
    min_fraction: float = MIN_ANOMALOUS_FRACTION,
) -> List[Dict]:
    """Scan a batch of same-run predictions for fleet-wide anomaly clusters.

    Returns a list of fleet_anomaly dicts. Contributing predictions are
    mutated in-place to carry `fleet_anomaly_id`, so individual alerts can
    be visually grouped/deduplicated downstream instead of paging on every
    single resource.
    """
    groups: Dict[Tuple[str, str, str], List[Dict]] = defaultdict(list)
    for p in predictions:
        groups[_group_key(p)].append(p)

    clusters = []
    for (provider, location, category), group in groups.items():
        if len(group) < min_cluster_size:
            continue

        anomalous = [p for p in group if _is_anomalous(p)]
        fraction = len(anomalous) / len(group)

        if fraction < min_fraction:
            continue

        cluster_id = f"fleet_{datetime.now(timezone.utc).strftime('%Y%m%d')}_{uuid4().hex[:8]}"
        for p in anomalous:
            p["fleet_anomaly_id"] = cluster_id

        clusters.append({
            "fleet_anomaly_id":  cluster_id,
            "provider":          provider,
            "location":          location,
            "category":          category,
            "total_resources":   len(group),
            "anomalous_resources": len(anomalous),
            "anomalous_fraction": round(fraction, 3),
            "resource_ids":      [p.get("resource_id") for p in anomalous],
            "detected_at":       datetime.now(timezone.utc).isoformat(),
            "interpretation": (
                f"{len(anomalous)}/{len(group)} ({fraction:.0%}) of {category} resources "
                f"in {location} ({provider}) are anomalous simultaneously — "
                "likely a shared/platform-level cause rather than independent incidents."
            ),
            "recommendation": (
                "Investigate provider status page, recent fleet-wide deploys, or shared "
                "dependency (DNS, IAM, control plane) before triaging individual resources."
            ),
        })

    return clusters
