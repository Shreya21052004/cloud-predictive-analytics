"""src/llm_resource_explain.py — Structured LLM explanation layer for individual resources.

Architecture
────────────
A dedicated preprocessing step (build_llm_explanation_context) transforms raw
prediction documents into a rich "LLM Explanation Context" before the LLM is
called. The LLM receives only engineering-friendly, pre-computed facts — it
never performs arithmetic.

Preprocessing pipeline:
  1. Detect available schema (category, provider, feature, payload shape)
  2. Assess current state (health classification, threshold proximity, headroom)
  3. Compute trend analysis (growth rates, volatility, regime changes)
  4. Build per-horizon forecast summaries with health labels
  5. Perform breach analysis (days to threshold, estimated breach dates)
  6. Rank dominant metrics across all available signals (cross-signal view)
  7. Extract category-specific operational signals (OOM probability, breach ETAs,
     DDoS score, node utilisation, database connection counts, etc.)
  8. Build anomaly context (why flagged, which signals dominate)
  9. Summarise evaluation quality (backtest scores, pooling, MASE, coverage)
 10. Sanitise resource metadata from summary_details

Dynamic schema handling:
  - No field access ever assumes the field exists
  - Every nested dict access uses _safe_get() with a default
  - Missing fields are silently omitted from the context (no errors, no noise)
  - Category-specific signals are discovered from whatever is actually in the payload

Output schema (8 required sections + metadata):
  resource_overview, current_state, trend_analysis, forecast, evidence,
  operational_impact, recommended_actions, confidence

Sensitive fields are NEVER included in the LLM prompt:
  account_id, subscription_id, tenant_id, resource_id, SSH config, access keys,
  tokens, certificates, connection strings, tags with internal labels.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from src.llm_summary import _get_client, _call_llm, DEFAULT_MODEL, DEFAULT_PROVIDER

RESOURCE_EXPLANATION_COLLECTION = "22llm_resource_explanations"

# The only values the UI understands. "HEALTHY" is not in the LLM prompt spec
# but Qwen3/other models sometimes emit it; map it and other aliases to the
# canonical set before writing to MongoDB.
_VALID_RISK_LEVELS: frozenset = frozenset({"LOW", "MEDIUM", "HIGH", "CRITICAL"})
_RISK_LEVEL_ALIASES: dict = {
    "HEALTHY": "LOW",
    "NONE":    "LOW",
    "MINIMAL": "LOW",
    "MODERATE": "MEDIUM",
    "SEVERE":  "HIGH",
    "URGENT":  "HIGH",
    "EXTREME": "CRITICAL",
}


def _normalize_risk_level(raw: str | None, prediction: dict) -> str:
    """Return a canonical risk level: LOW / MEDIUM / HIGH / CRITICAL.

    If the LLM returns an invalid or missing value, derive it from
    dashboard.display_score instead of falling back to "UNKNOWN".
    "UNKNOWN" is reserved for docs where the LLM call itself failed (_error=True).
    """
    if raw:
        upper = str(raw).strip().upper()
        if upper in _VALID_RISK_LEVELS:
            return upper
        mapped = _RISK_LEVEL_ALIASES.get(upper)
        if mapped:
            return mapped
    # Derive from the prediction's own risk score
    score_raw = (prediction.get("dashboard") or {}).get("display_score")
    try:
        score = float(score_raw) if score_raw is not None else None
    except (TypeError, ValueError):
        score = None
    if score is None:
        return "LOW"
    if score >= 90:
        return "CRITICAL"
    if score >= 75:
        return "HIGH"
    if score >= 50:
        return "MEDIUM"
    return "LOW"

# ─────────────────────────────────────────────────────────────────────────────
# Human-readable context for the LLM — resource types, categories, units
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Dynamic resource and metric description builders
# No static per-type or per-category text. Everything is derived from the
# actual fields available in the prediction document.
# ─────────────────────────────────────────────────────────────────────────────

def _extract_resource_type_label(prediction: dict) -> str:
    """Derive a human-readable resource type label from available metadata fields.

    Priority order:
      1. summary_details.type  (Azure: "Microsoft.Compute/virtualMachines")
      2. summary_details.kind  (GCP: "compute#instance", "storage#bucket")
      3. summary_details.resourceType / resource_type / instance_type / machine_type / sku
      4. component (EC2, GCS, VPC, etc.)
      5. canonical_resource_type with underscores stripped
      6. category as last resort
    Returns a short label (not a sentence), e.g. "VirtualMachine", "storage#bucket",
    "EC2 instance", "vm instance".
    """
    summary = prediction.get("summary_details") if isinstance(prediction.get("summary_details"), dict) else {}

    # Try summary_details structural type fields
    for key in ("type", "kind", "resourceType", "resource_type", "instance_type",
                "machine_type", "sku", "tier"):
        val = summary.get(key)
        if val and isinstance(val, str) and 1 < len(val) < 120:
            # Azure: "Microsoft.Compute/virtualMachines" → take last segment
            if "/" in val and not val.startswith("http"):
                val = val.rstrip("/").split("/")[-1]
            return val.strip()

    # Try component (e.g., "EC2", "S3", "GCS", "VPC")
    component = str(prediction.get("component") or "").strip()
    if component:
        return component

    # Fall back to canonical_resource_type with underscores → spaces
    canon = str(prediction.get("canonical_resource_type") or "").strip()
    if canon:
        return canon.replace("_", " ")

    category = str(prediction.get("category") or "").strip()
    return category or "cloud resource"


def _build_resource_type_description(prediction: dict) -> str:
    """Build a plain-English sentence describing what this resource IS.

    Assembled entirely from available prediction fields — never from a static
    lookup table. Reads: summary_details (type, kind, description, storageClass,
    locationType, network, properties), component, service_name,
    canonical_resource_type, category, provider, location.
    """
    category   = str(prediction.get("category") or "").strip()
    provider   = str(prediction.get("provider") or "").strip()
    component  = str(prediction.get("component") or "").strip()
    canon_type = str(prediction.get("canonical_resource_type") or "").strip()
    location   = str(prediction.get("location") or "").strip()
    summary    = prediction.get("summary_details") if isinstance(prediction.get("summary_details"), dict) else {}

    type_label = _extract_resource_type_label(prediction)

    # What the category means operationally — derived, not hardcoded per-type
    _CAT_ROLE = {
        "Compute":   "runs operating systems and application workloads",
        "Storage":   "stores and persists data durably",
        "Network":   "manages network connectivity and traffic routing",
        "Container": "orchestrates containerized workloads across nodes",
        "Databases": "manages structured data with query and transaction support",
    }
    operational_role = _CAT_ROLE.get(category, "provides cloud infrastructure services")

    # Compose: "{provider} {type_label} that {role} in {location}"
    provider_type = f"{provider} {type_label}".strip() if provider else type_label
    parts = [f"A {provider_type} that {operational_role}"]
    if location:
        parts.append(f"in {location}")

    # Append notable specifics from summary_details
    extras: list[str] = []
    if summary.get("storageClass"):
        extras.append(f"storage class: {summary['storageClass']}")
    if summary.get("locationType"):
        extras.append(f"location type: {summary['locationType']}")
    if summary.get("versioning") and isinstance(summary["versioning"], dict):
        if summary["versioning"].get("enabled"):
            extras.append("versioning enabled")
    if category == "Network":
        net = summary.get("network") or ""
        if net:
            net_short = str(net).split("/")[-1] if "/" in str(net) else str(net)
            extras.append(f"attached to network: {net_short}")
        dest = summary.get("destRange")
        if dest:
            extras.append(f"dest range: {dest}")
    if category == "Databases":
        engine = (
            summary.get("databaseVersion")
            or summary.get("engine")
            or summary.get("kind", "").replace("#", " ").strip()
        )
        if engine:
            extras.append(f"engine: {engine}")
    desc_field = summary.get("description")
    if isinstance(desc_field, str) and desc_field.strip() and len(desc_field) < 150:
        extras.append(desc_field.strip())

    sentence = " ".join(parts)
    if extras:
        sentence += f" ({'; '.join(extras)})"
    return sentence + "."


def _build_category_context(category: str, prediction: dict) -> str:
    """Build a short operational context sentence for the resource's category.

    Uses available fields (provider, component, canonical_resource_type) to
    personalise rather than returning a static per-category string.
    """
    provider   = str(prediction.get("provider") or "").strip()
    component  = str(prediction.get("component") or "").strip()
    canon_type = str(prediction.get("canonical_resource_type") or "").strip()
    service_id = component or canon_type.replace("_", " ")

    svc_suffix = f" on {provider} {service_id}".rstrip() if service_id else (
        f" on {provider}" if provider else ""
    )

    if category == "Compute":
        return (
            f"Compute resources{svc_suffix} provide CPU and memory for running workloads. "
            "When CPU or memory is exhausted, applications slow down, become unresponsive, or crash."
        )
    if category == "Storage":
        return (
            f"Storage resources{svc_suffix} persist data durably. "
            "When capacity fills up, write operations fail and applications cannot save new data."
        )
    if category == "Network":
        return (
            f"Network resources{svc_suffix} control how data flows between services and the internet. "
            "Degradation causes latency spikes, packet loss, and dropped connections."
        )
    if category == "Container":
        return (
            f"Container orchestration{svc_suffix} schedules and scales workloads across nodes. "
            "Resource pressure causes pods to fail scheduling or be evicted."
        )
    if category == "Databases":
        return (
            f"Database resources{svc_suffix} manage structured data and handle queries. "
            "Resource exhaustion slows queries, blocks transactions, and risks data loss."
        )
    return (
        f"Cloud {category.lower() or 'infrastructure'} resource{svc_suffix}. "
        "Monitor utilization to prevent service degradation."
    )


def _derive_metric_display_name(feature: str, forecast: Any) -> str:
    """Return the best available human-readable name for the monitored metric.

    Priority:
      1. behavioral_forecast.metric_name — the actual provider metric name
         (e.g. "CPUUtilization", "Percentage CPU", "networking.googleapis.com/...")
      2. Pattern-derived name from the canonical feature key
      3. The raw canonical feature key as last resort
    """
    if isinstance(forecast, dict):
        actual = str(forecast.get("metric_name") or "").strip()
        if actual:
            return actual

    if not feature:
        return "Unknown Metric"

    # Pattern-derive from canonical key (e.g. canonical_cpu_pct → "CPU %" )
    f = feature.replace("canonical_", "")
    _PATTERNS: list[tuple[str, str]] = [
        ("cpu_idle_pct",            "CPU Idle %"),
        ("cpu_pct",                 "CPU Utilization %"),
        ("mem_used_pct",            "Memory Utilization %"),
        ("storage_used_pct",        "Storage Used %"),
        ("burst_balance_pct",       "Burst Credit Balance %"),
        ("health_pct",              "Backend Health %"),
        ("availability_pct",        "Availability %"),
        ("subnet_util_pct",         "Subnet IP Utilization %"),
        ("node_cpu_pct",            "Node CPU %"),
        ("node_mem_pct",            "Node Memory %"),
        ("node_disk_pct",           "Node Disk %"),
        ("db_cpu_pct",              "Database CPU %"),
        ("db_mem_pct",              "Database Memory %"),
        ("db_storage_pct",          "Database Storage %"),
        ("db_connections",          "Active DB Connections"),
        ("db_dtu_used",             "DTU Usage"),
        ("active_connections",      "Active Connections"),
        ("net_in_bytes",            "Network Ingress (bytes/s)"),
        ("net_out_bytes",           "Network Egress (bytes/s)"),
        ("storage_read_bytes",      "Storage Read Throughput"),
        ("storage_write_bytes",     "Storage Write Throughput"),
        ("storage_read_iops",       "Storage Read IOPS"),
        ("storage_write_iops",      "Storage Write IOPS"),
        ("disk_read_bytes",         "Disk Read Throughput"),
        ("disk_write_bytes",        "Disk Write Throughput"),
        ("api_latency_ms",          "API Latency (ms)"),
        ("api_inflight_requests",   "API Inflight Requests"),
        ("unschedulable_pods",      "Unschedulable Pods"),
        ("unneeded_nodes",          "Unneeded Nodes"),
        ("run_failure_pct",         "Run Failure Rate %"),
        ("ddos_signal",             "DDoS Signal"),
        ("throttled_requests",      "Throttled Requests"),
        ("saturation",              "Saturation %"),
    ]
    for pattern, label in _PATTERNS:
        if pattern in f:
            return label

    # Last resort: prettify the raw key
    return f.replace("_", " ").title()


def _derive_metric_description(feature: str, forecast: Any, category: str) -> str:
    """Build a plain-English description of what the metric actually measures.

    Uses the actual metric_name from the forecast document when available,
    and derives meaning from the canonical feature key pattern. Never uses
    a static lookup table.
    """
    actual_name = ""
    if isinstance(forecast, dict):
        actual_name = str(forecast.get("metric_name") or "").strip()

    provider_note = f" (provider metric: {actual_name})" if actual_name else ""
    f = (feature or "").lower()

    if "cpu" in f and "idle" in f:
        return f"Percentage of CPU time sitting idle — 100% means fully idle, falling value means increasing load{provider_note}."
    if "cpu" in f:
        return f"Percentage of total CPU capacity currently in use — higher values mean the processor is busier{provider_note}."
    if "mem" in f:
        return f"Percentage of available RAM currently in use — higher values mean less memory is free for workloads{provider_note}."
    if "burst" in f and "balance" in f:
        return (
            f"Remaining I/O burst credit balance as a percentage. "
            f"This pool drains under sustained workloads and recharges during low usage. "
            f"When it reaches 0%, I/O throughput is throttled to the baseline rate{provider_note}."
        )
    if "storage" in f and "iops" in f:
        return f"Number of storage read or write operations per second{provider_note}."
    if "storage" in f and "bytes" in f:
        return f"Storage data throughput in bytes per second{provider_note}."
    if "storage" in f and "pct" in f:
        return f"Percentage of total storage/disk capacity consumed — 100% means the volume is completely full{provider_note}."
    if "disk" in f and "bytes" in f:
        return f"Disk I/O throughput in bytes per second{provider_note}."
    if "net" in f and "in" in f:
        return f"Inbound network throughput — data flowing into this resource from the network{provider_note}."
    if "net" in f and "out" in f:
        return f"Outbound network throughput — data flowing out of this resource to the network{provider_note}."
    if "health" in f and "pct" in f:
        return (
            f"Percentage of backend targets currently passing health checks. "
            f"Lower values indicate more backends are unhealthy or unreachable{provider_note}."
        )
    if "availability" in f:
        return f"Service availability as a percentage — values below 99% represent significant downtime{provider_note}."
    if "subnet" in f and "util" in f:
        return f"Percentage of available subnet IP addresses already allocated — exhaustion prevents new resources from joining{provider_note}."
    if "db" in f and "connection" in f:
        return f"Number of active connections open to the database engine{provider_note}."
    if "db" in f and "cpu" in f:
        return f"Percentage of CPU capacity used by the database engine{provider_note}."
    if "db" in f and "mem" in f:
        return f"Percentage of memory used by the database engine — high values increase risk of out-of-memory events{provider_note}."
    if "db" in f and "storage" in f:
        return f"Percentage of database storage space consumed{provider_note}."
    if "dtu" in f:
        return f"Azure Database Transaction Units (DTUs) consumed — a blended CPU+memory+I/O capacity unit for Azure SQL{provider_note}."
    if "node" in f and "cpu" in f:
        return f"Average CPU utilization across all active nodes in this cluster{provider_note}."
    if "node" in f and "mem" in f:
        return f"Average memory utilization across all active nodes in this cluster{provider_note}."
    if "node" in f and "disk" in f:
        return f"Average disk utilization across all active nodes in this cluster{provider_note}."
    if "unschedulable" in f or "pod" in f:
        return f"Count of pods that cannot be scheduled because no node has sufficient free resources{provider_note}."
    if "inflight" in f or "api" in f:
        return f"Number of concurrent in-flight requests to the Kubernetes API server{provider_note}."
    if "latency" in f or f.endswith("_ms"):
        return f"Response latency in milliseconds — higher values indicate slower responses{provider_note}."
    if "iops" in f:
        return f"I/O operations per second{provider_note}."
    if "saturation" in f:
        return f"Resource saturation or queue depth as a percentage — high values indicate the resource is a bottleneck{provider_note}."
    if "ddos" in f:
        return f"Probability signal for DDoS attack activity — elevated values indicate anomalous inbound traffic patterns{provider_note}."
    if "failure" in f:
        return f"Percentage of job, function, or service executions that failed{provider_note}."
    if "throttle" in f:
        return f"Count of requests that were rate-limited or throttled due to exceeding service quotas{provider_note}."
    if "active_connection" in f:
        return f"Number of currently open network connections to this resource{provider_note}."

    # Generic fallback — use actual metric name if available
    if actual_name:
        return f"Provider metric '{actual_name}' for this {category.lower() or 'cloud'} resource."
    return f"Cloud infrastructure metric for this {category.lower() or ''} resource ({feature}).".strip()

_METRIC_UNITS: dict[str, str] = {
    "canonical_cpu_pct":                 "%",
    "canonical_cpu_idle_pct":            "%",
    "canonical_mem_used_pct":            "%",
    "canonical_storage_used_pct":        "%",
    "canonical_burst_balance_pct":       "%",
    "canonical_subnet_util_pct":         "%",
    "canonical_node_cpu_pct":            "%",
    "canonical_node_mem_pct":            "%",
    "canonical_node_disk_pct":           "%",
    "canonical_db_cpu_pct":              "%",
    "canonical_db_mem_pct":              "%",
    "canonical_db_storage_pct":          "%",
    "canonical_health_pct":              "%",
    "canonical_availability_pct":        "%",
    "canonical_saturation":              "%",
    "canonical_run_failure_pct":         "%",
    "canonical_db_connections":          "connections",
    "canonical_active_connections":      "connections",
    "canonical_net_in_bytes":            "bytes/s",
    "canonical_net_out_bytes":           "bytes/s",
    "canonical_storage_read_bytes":      "bytes/s",
    "canonical_storage_write_bytes":     "bytes/s",
    "canonical_storage_read_iops":       "IOPS",
    "canonical_storage_write_iops":      "IOPS",
    "canonical_disk_read_bytes":         "bytes/s",
    "canonical_disk_write_bytes":        "bytes/s",
    "canonical_api_latency_ms":          "ms",
    "canonical_unschedulable_pods":      "pods",
    "canonical_unneeded_nodes":          "nodes",
    "canonical_api_inflight_requests":   "requests",
    "canonical_db_dtu_used":             "DTUs",
    "canonical_ddos_signal":             "signal",
    "canonical_throttled_requests":      "requests",
}

# (warning_threshold, critical_threshold) for percentage metrics.
# Inverted metrics degrade as the value FALLS — thresholds applied in reverse.
_PCT_THRESHOLDS: dict[str, tuple[float, float]] = {
    "canonical_cpu_pct":            (80.0, 90.0),
    "canonical_mem_used_pct":       (80.0, 90.0),
    "canonical_storage_used_pct":   (80.0, 90.0),
    "canonical_node_cpu_pct":       (80.0, 90.0),
    "canonical_node_mem_pct":       (80.0, 90.0),
    "canonical_node_disk_pct":      (80.0, 90.0),
    "canonical_db_cpu_pct":         (80.0, 90.0),
    "canonical_db_mem_pct":         (80.0, 90.0),
    "canonical_db_storage_pct":     (85.0, 95.0),
    "canonical_saturation":         (75.0, 90.0),
    "canonical_subnet_util_pct":    (75.0, 90.0),
    "canonical_run_failure_pct":    (40.0, 70.0),
    # Inverted: breach when value DROPS below threshold
    "canonical_burst_balance_pct":  (20.0, 10.0),
    "canonical_health_pct":         (90.0, 80.0),
    "canonical_availability_pct":   (99.0, 95.0),
}

_INVERTED_METRICS: set[str] = {
    "canonical_burst_balance_pct",
    "canonical_health_pct",
    "canonical_availability_pct",
}

# _METRIC_DISPLAY_NAMES removed — use _derive_metric_display_name(feature, forecast)
# which prefers the provider's actual metric_name from the forecast document.


def _derive_unit_from_feature(feature: str) -> str:
    """Derive a measurement unit from the canonical feature key naming pattern.
    Used as a fallback when the feature is not in _METRIC_UNITS."""
    f = (feature or "").lower()
    if f.endswith("_pct") or "pct" in f:
        return "%"
    if "_bytes" in f:
        return "bytes/s"
    if "_iops" in f:
        return "IOPS"
    if "_ms" in f:
        return "ms"
    if "_connections" in f:
        return "connections"
    if "_dtu" in f or "dtu_" in f:
        return "DTUs"
    if "_pods" in f:
        return "pods"
    if "_nodes" in f:
        return "nodes"
    if "_requests" in f:
        return "requests"
    return ""

# Keys in category_payload() output that are excluded from the context
# (already captured via other channels or too verbose).
_PAYLOAD_SKIP_KEYS: set[str] = {"risk_score", "available_signals"}

# Nested breach-date sub-dicts inside the payload → flat descriptive keys.
_PAYLOAD_BREACH_FLATTEN: dict[str, dict[str, str]] = {
    "cpu_saturation_forecast": {
        "breach_80": "cpu_80pct_breach_eta",
        "breach_90": "cpu_90pct_breach_eta",
    },
    "capacity_exhaustion_forecast": {
        "breach_80": "storage_80pct_breach_eta",
        "breach_90": "storage_90pct_breach_eta",
        "breach_95": "storage_95pct_breach_eta",
    },
    "connection_saturation_forecast": {
        "breach_10000": "connection_10k_breach_eta",
        "breach_25000": "connection_25k_breach_eta",
    },
    "subnet_capacity_forecast": {
        "breach_80": "subnet_80pct_breach_eta",
        "breach_90": "subnet_90pct_breach_eta",
    },
}

# Fragments that make a summary_details key potentially sensitive.
_SUMMARY_SENSITIVE_FRAGMENTS: tuple[str, ...] = (
    "password", "passwd", "secret", "token", "key", "credential", "auth",
    "cert", "certificate", "connection_string", "connectionstring",
    "endpoint", "uri", "url", "ssh", "fingerprint", "private", "access",
    "subscription_id", "tenant_id", "client_id", "client_secret", "apikey",
)

# Category-specific sibling signal definitions:
# (payload_key, display_name, unit, warn_threshold, crit_threshold, is_inverted, is_prob)
# is_prob=True: value is on 0-1 scale, multiply by 100 for display.
_CATEGORY_SIBLING_SIGNALS: dict[str, list[tuple]] = {
    "Compute": [
        ("oom_probability",              "OOM Probability",         "%",    40,  70, False, True),
        ("health_failure_probability_24h","Health Failure Risk 24h","%",    40,  70, False, True),
    ],
    "Storage": [
        ("burst_exhaustion_risk",        "Burst Credit Exhaustion", "%",    40,  70, False, True),
        ("io_anomaly_score",             "I/O Anomaly Score",       "%",    40,  70, False, True),
    ],
    "Network": [
        ("ddos_probability",             "DDoS Probability",        "%",    20,  50, False, True),
        ("health_pct",                   "Backend Health",          "%",    90,  80, True,  False),
    ],
    "Container": [
        ("node_cpu_pct",                 "Node CPU",                "%",    80,  90, False, False),
        ("node_mem_pct",                 "Node Memory",             "%",    80,  90, False, False),
        ("node_disk_pct",                "Node Disk",               "%",    80,  90, False, False),
        ("unschedulable_pods",           "Unschedulable Pods",      "pods",  1,   5, False, False),
        ("api_inflight_requests",        "API Inflight Requests",   "req", 300, 500, False, False),
    ],
    "Databases": [
        ("db_cpu_pct",                   "Database CPU",            "%",    80,  90, False, False),
        ("db_mem_pct",                   "Database Memory",         "%",    80,  90, False, False),
        ("db_storage_pct",               "Database Storage",        "%",    85,  95, False, False),
        ("db_connections",               "DB Connections",          "",     70,  90, False, False),
        ("db_dtu_used",                  "DTU Usage",               "DTU", None,None,False, False),
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Low-level helpers — all safe, never raise
# ─────────────────────────────────────────────────────────────────────────────

def _safe_get(d: Any, *keys: str, default: Any = None) -> Any:
    """Safe nested dict access — never raises on missing keys or wrong types."""
    for key in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(key)
        if d is None:
            return default
    return d if d is not None else default


def _safe_float(v: Any, default: float | None = None) -> float | None:
    """Safe float conversion — returns default for None, non-numeric, or NaN."""
    if v is None:
        return default
    try:
        f = float(v)
        return default if (f != f) else f   # NaN guard
    except (TypeError, ValueError):
        return default


def _safe_round(v: Any, decimals: int = 2) -> float | None:
    f = _safe_float(v)
    return None if f is None else round(f, decimals)


def _is_pct_metric(feature: str | None) -> bool:
    if not feature:
        return False
    return feature.endswith("_pct") or feature in {
        "canonical_cpu_pct", "canonical_storage_used_pct",
    }


def _score_to_risk_label(score: Any) -> str:
    s = _safe_float(score)
    if s is None:
        return "UNKNOWN"
    if s >= 90: return "CRITICAL"
    if s >= 75: return "HIGH"
    if s >= 50: return "MEDIUM"
    if s > 0:   return "LOW"
    return "HEALTHY"


def _classify_severity(value: float | None, feature: str | None) -> str:
    """Classify a metric value as HEALTHY / WARNING / CRITICAL based on thresholds."""
    if value is None or not feature:
        return "UNKNOWN"
    thresholds = _PCT_THRESHOLDS.get(feature)
    if not thresholds:
        return "UNKNOWN"
    warn_t, crit_t = thresholds
    is_inv = feature in _INVERTED_METRICS
    if is_inv:
        if value <= crit_t: return "CRITICAL"
        if value <= warn_t: return "WARNING"
        return "HEALTHY"
    else:
        if value >= crit_t: return "CRITICAL"
        if value >= warn_t: return "WARNING"
        return "HEALTHY"


def _growth_rate_label(
    daily_rate: float | None,
    current: float | None,
    feature: str | None,
) -> str | None:
    """Translate a daily growth rate into an engineering-friendly label."""
    if daily_rate is None or current is None:
        return None
    is_inv = (feature or "") in _INVERTED_METRICS
    is_pct = _is_pct_metric(feature)
    unit   = _METRIC_UNITS.get(feature or "", "")

    if abs(daily_rate) < 0.005:
        return "effectively stable (near-zero change per day)"

    sign = "+" if daily_rate > 0 else ""

    if is_pct:
        if not is_inv:
            headroom = max(100.0 - current, 0.1)
            rate_of_fill = abs(daily_rate) / headroom * 100
            direction = "increasing" if daily_rate > 0 else "decreasing"
            speed     = "rapidly" if rate_of_fill > 20 else ("steadily" if rate_of_fill > 5 else "slowly")
            return f"{speed} {direction} ({sign}{daily_rate:.2f}%/day)"
        else:
            headroom = max(current, 0.1)
            if daily_rate < 0:
                rate_of_fall = abs(daily_rate) / headroom * 100
                speed = "rapidly" if rate_of_fall > 20 else ("steadily" if rate_of_fall > 5 else "slowly")
                return f"{speed} declining ({sign}{daily_rate:.2f}%/day) — moving toward exhaustion"
            return f"recovering ({sign}{daily_rate:.2f}%/day)"
    else:
        if abs(current) > 0:
            pct_rate = abs(daily_rate) / abs(current) * 100
            speed = "rapidly" if pct_rate > 20 else ""
            direction = "increasing" if daily_rate > 0 else "decreasing"
            return f"{'rapidly ' if speed else ''}{direction} ({sign}{daily_rate:.4g} {unit}/day)"
        return f"{'increasing' if daily_rate > 0 else 'decreasing'} ({sign}{daily_rate:.4g} {unit}/day)"


def _days_label(days: float | None) -> str | None:
    """Convert a decimal number of days to a human-friendly string."""
    if days is None:
        return None
    if days == 0:
        return "already breached"
    if days < 1:
        h = days * 24
        return f"~{h:.0f} hour{'s' if h != 1 else ''}"
    if days < 7:
        return f"~{days:.1f} day{'s' if days > 1.4 else ''}"
    if days < 30:
        w = days / 7
        return f"~{w:.1f} week{'s' if w > 1.4 else ''}"
    m = days / 30
    return f"~{m:.1f} month{'s' if m > 1.4 else ''}"


def _ci_volatility_label(ci_width: float | None, current: float | None) -> str | None:
    """Classify signal volatility from the 24h confidence interval width."""
    if ci_width is None or current is None or current == 0:
        return None
    rel = ci_width / abs(current)
    if rel < 0.05:  return "very stable (tight CI — high forecast precision)"
    if rel < 0.15:  return "moderately stable"
    if rel < 0.40:  return "moderately volatile (wide CI — forecast less precise)"
    return "highly volatile (very wide CI — treat point forecast with caution)"


def _pooling_explanation(
    pooling_tier: str | None,
    n_resources: int | None,
    canon_type: str | None,
    age_days: int | None,
) -> str | None:
    if not pooling_tier:
        return None
    age_txt = f" over {age_days} days" if age_days else ""
    if pooling_tier == "individual":
        return f"Forecast built entirely from this resource's own history{age_txt}."
    if pooling_tier == "type_pooled":
        pool_txt = (
            f" from {n_resources} similar {canon_type} resources"
            if n_resources and canon_type else ""
        )
        return (
            f"Forecast benefits from pooled data{pool_txt}, stabilising the slope "
            f"estimate when individual history is thin."
        )
    if pooling_tier == "category_pooled":
        return (
            "Forecast uses category-wide pooled data because this resource type "
            "has limited individual history — treat with additional caution."
        )
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Summary-details sanitiser
# ─────────────────────────────────────────────────────────────────────────────

_CRED_VALUE_RE = re.compile(
    r"(-----BEGIN|arn:|[A-Za-z0-9+/]{40,}={0,2}$|[a-f0-9]{32,})",
    re.IGNORECASE,
)


def _sanitize_summary_details(raw: Any, _depth: int = 0) -> Any:
    """Return a sanitised version of summary_details with sensitive fields removed.

    Keeps: instance type, OS, environment, team, application purpose, capacity
           information, tags, any operational metadata.
    Drops: keys/values that look like credentials, connection strings, tokens,
           SSH keys, ARNs, long base64 blobs, and similarly sensitive patterns.

    Handles strings, dicts, lists, and nested structures up to two levels deep.
    """
    if _depth > 2 or raw is None:
        return None

    if isinstance(raw, dict):
        clean: dict = {}
        for k, v in raw.items():
            k_lower = str(k).lower().replace("-", "_").replace(" ", "_")
            # Drop keys that suggest credentials
            if any(frag in k_lower for frag in _SUMMARY_SENSITIVE_FRAGMENTS):
                continue
            sanitised_v = _sanitize_summary_details(v, _depth + 1)
            if sanitised_v is not None:
                clean[k] = sanitised_v
        return clean or None

    if isinstance(raw, list):
        result = [_sanitize_summary_details(item, _depth + 1) for item in raw]
        result = [r for r in result if r is not None]
        return result or None

    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return None
        # Drop values that look like secrets or connection strings
        if _CRED_VALUE_RE.search(s) or "://" in s:
            return None
        # Truncate very long strings
        return (s[:397] + "…") if len(s) > 400 else s

    # Numeric, bool, etc.
    return raw


# ─────────────────────────────────────────────────────────────────────────────
# Sibling metric helpers — populate dominant metrics from the category payload
# ─────────────────────────────────────────────────────────────────────────────

def _sibling_severity(val: float, warn: float | None, crit: float | None, is_inv: bool) -> str:
    if warn is None or crit is None:
        return "UNKNOWN"
    if is_inv:
        if val <= crit: return "CRITICAL"
        if val <= warn: return "WARNING"
        return "HEALTHY"
    else:
        if val >= crit: return "CRITICAL"
        if val >= warn: return "WARNING"
        return "HEALTHY"


def _populate_dominant_from_payload(
    payload: dict,
    category: str,
    primary_feature: str,
    dominant: list,
) -> None:
    """Dynamically add sibling metric signals from the category payload.

    Uses the _CATEGORY_SIBLING_SIGNALS table but never assumes a key exists —
    it only adds entries where payload[key] is a real numeric value.
    """
    if not isinstance(payload, dict):
        return

    for sig_def in _CATEGORY_SIBLING_SIGNALS.get(category, []):
        payload_key, display_name, unit, warn, crit, is_inv, is_prob = sig_def
        raw_val = payload.get(payload_key)
        if raw_val is None:
            continue
        val = _safe_float(raw_val)
        if val is None:
            continue

        display_val = round(val * 100, 1) if is_prob else round(val, 2)
        display_unit = "%" if is_prob else unit
        sev = _sibling_severity(display_val, warn, crit, is_inv)

        dominant.append({
            "metric_name":               display_name,
            "current_value":             display_val,
            "unit":                      display_unit,
            "severity":                  sev,
            "is_primary_forecast_target": False,
        })

    # Label-type signals (not simple scalars)
    if category == "Compute":
        mpc = payload.get("memory_pressure_class")
        if isinstance(mpc, str) and mpc not in ("UNKNOWN", ""):
            dominant.append({
                "metric_name":  "Memory Pressure",
                "classification": mpc,
                "severity":     "CRITICAL" if mpc == "CRITICAL" else ("WARNING" if mpc == "WARNING" else "HEALTHY"),
                "is_primary_forecast_target": primary_feature == "canonical_mem_used_pct",
            })
        if payload.get("iowait_spike_flag"):
            dominant.append({
                "metric_name":  "I/O Wait Spike",
                "classification": "SPIKE DETECTED",
                "severity":     "WARNING",
                "is_primary_forecast_target": False,
            })

    if category == "Storage" and payload.get("backup_failure_risk"):
        dominant.append({
            "metric_name":  "Backup Health",
            "classification": "FAILURE DETECTED",
            "severity":     "CRITICAL",
            "is_primary_forecast_target": False,
        })


def _extract_category_signals(payload: dict, category: str) -> dict:
    """Build a clean operational-signals dict from the category payload.

    Flattens breach-date sub-dicts into descriptive keys. Only includes True
    booleans (False flags add noise). Skips risk_score and available_signals.
    Never fails on missing or unexpected payload shapes.
    """
    if not isinstance(payload, dict):
        return {}

    signals: dict = {}
    for k, v in payload.items():
        if k in _PAYLOAD_SKIP_KEYS or v is None:
            continue

        if k in _PAYLOAD_BREACH_FLATTEN and isinstance(v, dict):
            for src_k, dst_k in _PAYLOAD_BREACH_FLATTEN[k].items():
                eta = v.get(src_k)
                if eta is not None:
                    signals[dst_k] = eta
            continue

        if isinstance(v, dict):
            # Unexpected nested dict — skip to keep the prompt clean
            continue

        if isinstance(v, bool):
            # Only include True booleans — a False flag adds noise
            if v:
                signals[k] = v
        else:
            signals[k] = v

    return signals


# ─────────────────────────────────────────────────────────────────────────────
# Master preprocessing step
# ─────────────────────────────────────────────────────────────────────────────

def build_llm_explanation_context(prediction: dict) -> dict:
    """Transform a raw prediction document into an LLM Explanation Context.

    This is the single preprocessing entry point. It:
      1. Detects what data is actually available (never assumes any field exists)
      2. Pre-computes all engineering-friendly derived facts
      3. Builds a structured, human-readable context dict
      4. Never fails — missing fields are silently omitted

    The returned context is divided into logical sections:
      resource_identity, forecast_target, current_state, trend_analysis,
      forecast_horizons, breach_analysis, dominant_metrics,
      category_operational_signals, anomaly_context, evaluation_quality,
      resource_metadata, pipeline_recommendations, detected_failure_risks,
      risk_score, risk_label, alert_state
    """
    # ── Safe extraction of top-level sub-documents ────────────────────────────
    # Use dict.get() with {} fallback everywhere — if a sub-document is absent
    # or not a dict, it becomes an empty dict and all downstream .get() calls
    # simply return None.
    forecast     = prediction.get("behavioral_forecast") if isinstance(prediction.get("behavioral_forecast"), dict) else {}
    payload      = prediction.get("payload")             if isinstance(prediction.get("payload"), dict)             else {}
    dashboard    = prediction.get("dashboard")           if isinstance(prediction.get("dashboard"), dict)           else {}
    alert_doc    = prediction.get("alert")               if isinstance(prediction.get("alert"), dict)               else {}
    profile_info = prediction.get("profile_info")        if isinstance(prediction.get("profile_info"), dict)        else {}
    anomaly_det  = prediction.get("anomaly_detail")      if isinstance(prediction.get("anomaly_detail"), dict)      else {}

    # ── Schema detection: what kind of resource is this? ─────────────────────
    category   = str(prediction.get("category") or "").strip()
    provider   = str(prediction.get("provider") or "").strip()
    canon_type = str(prediction.get("canonical_resource_type") or "").strip()
    component  = str(prediction.get("component") or "").strip()
    feature    = str(forecast.get("canonical_feature") or "").strip()
    metric_nm  = forecast.get("metric_name")

    is_pct     = _is_pct_metric(feature) if feature else False
    is_inv     = feature in _INVERTED_METRICS
    unit       = _METRIC_UNITS.get(feature, "") if feature else ""
    thresholds = _PCT_THRESHOLDS.get(feature, (None, None))
    warn_t, crit_t = thresholds

    current    = _safe_float(forecast.get("current_value"))

    ctx: dict = {}

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 1 — Current State Assessment
    # ═══════════════════════════════════════════════════════════════════════════
    state: dict = {}

    if current is not None:
        state["current_value"] = round(current, 2)
        state["unit"]          = unit

        health_sev = _classify_severity(current, feature if feature else None)
        state["health_status"] = health_sev

        if warn_t is not None and crit_t is not None:
            state["warning_threshold"]  = warn_t
            state["critical_threshold"] = crit_t

            at_warn = (current <= warn_t) if is_inv else (current >= warn_t)
            at_crit = (current <= crit_t) if is_inv else (current >= crit_t)
            state["currently_at_warning"]  = at_warn
            state["currently_at_critical"] = at_crit

            if not at_warn:
                # How much headroom before warning?
                gap = (current - warn_t) if is_inv else (warn_t - current)
                state["gap_to_warning_threshold"] = round(gap, 2)
                state["gap_to_warning_unit"]       = unit
                prox = round((current / warn_t) * 100, 1) if (is_inv and warn_t != 0) else (
                    round((current / warn_t) * 100, 1) if warn_t != 0 else None
                )
                if prox is not None:
                    state["pct_of_warning_threshold"] = prox
            else:
                excess = (warn_t - current) if is_inv else (current - warn_t)
                state["threshold_exceeded_by"] = round(excess, 2)
                state["threshold_exceeded_unit"] = unit

        if is_pct:
            if not is_inv:
                state["remaining_headroom_pct"] = round(100.0 - current, 2)
                state["consumed_pct"]           = round(current, 2)
            else:
                state["remaining_headroom_pct"] = round(current, 2)
                state["consumed_pct"]           = round(100.0 - current, 2)

    # Readable health label in plain English
    health_notes = {
        "HEALTHY":  "operating normally — within safe thresholds",
        "WARNING":  "approaching a threshold that requires attention",
        "CRITICAL": "at or past a critical threshold — immediate action may be required",
        "UNKNOWN":  "health status cannot be determined from available data",
    }
    health_key = state.get("health_status", "UNKNOWN")
    state["health_interpretation"] = health_notes.get(health_key, "")

    if state:
        ctx["current_state"] = state

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 2 — Trend Analysis
    # ═══════════════════════════════════════════════════════════════════════════
    trend: dict = {}

    trend_dir = forecast.get("trend_direction")
    if trend_dir:
        trend["direction"] = trend_dir

    if current is not None:
        f7d  = _safe_float(forecast.get("forecast_7d"))
        f24h = _safe_float(forecast.get("forecast_24h"))
        f30d = _safe_float(forecast.get("forecast_30d"))

        daily_rate: float | None = None
        if f7d is not None:
            daily_rate = (f7d - current) / 7.0
        elif f24h is not None:
            daily_rate = f24h - current

        if daily_rate is not None:
            trend["daily_change_rate"]   = round(daily_rate, 4)
            trend["daily_change_unit"]   = unit
            trend["weekly_change"]       = round(daily_rate * 7, 3)
            trend["hourly_change_rate"]  = round(daily_rate / 24.0, 6)
            lbl = _growth_rate_label(daily_rate, current, feature if feature else None)
            if lbl:
                trend["growth_rate_label"] = lbl

        if f30d is not None:
            trend["value_in_30_days"] = round(f30d, 2)

        # Signal volatility from the 24h CI width
        lo_24 = _safe_float(forecast.get("forecast_24h_lower"))
        hi_24 = _safe_float(forecast.get("forecast_24h_upper"))
        if lo_24 is not None and hi_24 is not None:
            ci_w = hi_24 - lo_24
            trend["ci_width_24h"]     = round(ci_w, 3)
            vlbl = _ci_volatility_label(ci_w, current)
            if vlbl:
                trend["signal_volatility"] = vlbl

    # Regime change
    if profile_info.get("regime_changed"):
        trend["regime_change_detected"] = True
        trend["regime_change_note"] = (
            "A sustained shift from this resource type's historical baseline was detected. "
            "If this reflects an intentional change (deployment, scaling event, traffic growth), "
            "the baseline will re-settle automatically. If unexpected, investigate the root cause."
        )

    if trend:
        ctx["trend_analysis"] = trend

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 3 — Multi-horizon Forecasts with Health Labels
    # ═══════════════════════════════════════════════════════════════════════════
    horizons: list[dict] = []
    HORIZON_DEFS = [
        ("forecast_1h",  "1 hour",   1),
        ("forecast_6h",  "6 hours",  6),
        ("forecast_24h", "24 hours", 24),
        ("forecast_7d",  "7 days",   168),
        ("forecast_30d", "30 days",  720),
    ]
    for hkey, hlabel, hhours in HORIZON_DEFS:
        val = _safe_float(forecast.get(hkey))
        if val is None:
            continue
        lo = _safe_float(forecast.get(f"{hkey}_lower"))
        hi = _safe_float(forecast.get(f"{hkey}_upper"))
        h: dict = {
            "horizon":            hlabel,
            "hours_ahead":        hhours,
            "predicted_value":    round(val, 2),
            "unit":               unit,
            "predicted_health":   _classify_severity(val, feature if feature else None),
        }
        if lo is not None:
            h["lower_bound"] = round(lo, 2)
        if hi is not None:
            h["upper_bound"] = round(hi, 2)
        if lo is not None and hi is not None:
            h["ci_width"] = round(hi - lo, 3)
        horizons.append(h)

    if horizons:
        ctx["forecast_horizons"] = horizons

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 4 — Breach Analysis
    # ═══════════════════════════════════════════════════════════════════════════
    breach: dict = {}
    daily_rate_b = _safe_float((ctx.get("trend_analysis") or {}).get("daily_change_rate"))

    if is_pct and current is not None and warn_t is not None and crit_t is not None:
        breach["warning_threshold"]  = warn_t
        breach["critical_threshold"] = crit_t

        if is_inv:
            breach["currently_breaching_warning"]  = current <= warn_t
            breach["currently_breaching_critical"] = current <= crit_t
            if daily_rate_b is not None and daily_rate_b < 0:
                for label, thresh in [("warning", warn_t), ("critical", crit_t)]:
                    if current > thresh:
                        days = (current - thresh) / abs(daily_rate_b)
                        if 0 < days < 3650:
                            breach[f"days_to_{label}_breach"]    = round(days, 1)
                            breach[f"{label}_breach_in"]         = _days_label(days)
                            breach[f"{label}_breach_date"]       = (
                                datetime.now(timezone.utc) + timedelta(days=days)
                            ).strftime("%Y-%m-%d")
                    elif current <= thresh:
                        breach[f"days_to_{label}_breach"]  = 0
                        breach[f"{label}_breach_note"]     = f"Already below the {thresh}{unit} {label} threshold."
        else:
            breach["currently_breaching_warning"]  = current >= warn_t
            breach["currently_breaching_critical"] = current >= crit_t
            if daily_rate_b is not None and daily_rate_b > 0:
                for label, thresh in [("warning", warn_t), ("critical", crit_t)]:
                    if current < thresh:
                        days = (thresh - current) / daily_rate_b
                        if 0 < days < 3650:
                            breach[f"days_to_{label}_breach"]    = round(days, 1)
                            breach[f"{label}_breach_in"]         = _days_label(days)
                            breach[f"{label}_breach_date"]       = (
                                datetime.now(timezone.utc) + timedelta(days=days)
                            ).strftime("%Y-%m-%d")
                    elif current >= thresh:
                        breach[f"days_to_{label}_breach"]  = 0
                        breach[f"{label}_breach_note"]     = f"Already at or past the {thresh}{unit} {label} threshold."

    # Category payload breach ETAs (from slope_forecast — complementary)
    for outer_key, inner_map in _PAYLOAD_BREACH_FLATTEN.items():
        sub = payload.get(outer_key)
        if not isinstance(sub, dict):
            continue
        for inner_key, dst_key in inner_map.items():
            val = sub.get(inner_key)
            if val is not None:
                breach[dst_key] = val

    if breach:
        ctx["breach_analysis"] = breach

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 5 — Dominant Metrics (cross-signal severity ranking)
    # ═══════════════════════════════════════════════════════════════════════════
    dominant: list[dict] = []

    # Primary metric
    if current is not None and feature:
        dominant.append({
            "metric_name":               _derive_metric_display_name(feature, forecast),
            "canonical_key":             feature,
            "current_value":             round(current, 2),
            "unit":                      unit,
            "severity":                  _classify_severity(current, feature),
            "is_primary_forecast_target": True,
        })

    # Sibling metrics from payload
    _populate_dominant_from_payload(payload, category, feature, dominant)

    # Sort: CRITICAL first, then WARNING, then HEALTHY; primary before siblings
    _SEV_ORDER = {"CRITICAL": 0, "WARNING": 1, "HEALTHY": 2, "UNKNOWN": 3}
    dominant.sort(key=lambda d: (
        _SEV_ORDER.get(d.get("severity", "UNKNOWN"), 3),
        0 if d.get("is_primary_forecast_target") else 1,
    ))

    if dominant:
        ctx["dominant_metrics"] = dominant[:6]

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 6 — Category Operational Signals
    # ═══════════════════════════════════════════════════════════════════════════
    cat_signals = _extract_category_signals(payload, category)
    if cat_signals:
        ctx["category_operational_signals"] = cat_signals

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 7 — Anomaly Context
    # ═══════════════════════════════════════════════════════════════════════════
    anomaly: dict = {}

    is_anomalous = prediction.get("is_anomalous")
    if is_anomalous is not None:
        anomaly["is_anomalous"] = is_anomalous

    anom_score = _safe_float(prediction.get("anomaly_score"))
    if anom_score is not None:
        anomaly["anomaly_score"] = round(anom_score, 3)

    # Internal anomaly signals — only include when they carry meaningful content
    det_error = anomaly_det.get("error")
    if det_error:
        if det_error == "no_telemetry":
            anomaly["data_issue"]      = "no_telemetry"
            anomaly["data_issue_note"] = "No metric data was available — verify the monitoring agent is reporting."
        elif det_error == "stale_resource":
            anomaly["data_issue"]      = "stale_resource"
            age = anomaly_det.get("age_minutes")
            anomaly["data_issue_note"] = (
                f"Resource has not reported metrics for {age:.0f} minutes — collector may be down."
                if age is not None
                else "Resource metrics are stale — collector may have stopped reporting."
            )
    elif anomaly_det:
        # Surface individual anomaly signals when they are elevated
        zscore = _safe_float(anomaly_det.get("residual_zscore_signal"))
        if zscore is not None and zscore >= 0.4:
            anomaly["zscore_signal"] = round(zscore, 3)
            anomaly["zscore_note"] = (
                "current value is far outside this resource's own historical pattern for this time period"
                if zscore >= 0.7 else
                "current value is moderately unusual compared to historical baseline"
            )

        burst = _safe_float(anomaly_det.get("burst_ratio_signal"))
        if burst is not None and burst >= 0.4:
            anomaly["burst_ratio_signal"] = round(burst, 3)
            anomaly["burst_note"] = "short-term spike detected within the reporting interval (max/avg ratio elevated)"

        iff = _safe_float(anomaly_det.get("isolation_forest_signal"))
        if iff is not None and iff >= 0.55:
            anomaly["isolation_forest_signal"] = round(iff, 3)
            anomaly["isolation_forest_note"] = "multi-signal isolation forest pattern flags this as a behavioural outlier"

        if anomaly_det.get("any_signal_dominant"):
            anomaly["any_signal_dominant"] = True

    if profile_info.get("regime_changed"):
        anomaly["regime_change_detected"] = True
        anomaly["regime_change_note"] = (
            "A sustained shift from this resource type's historical baseline was recently detected. "
            "This could reflect an intentional change (deployment, scaling, traffic growth) "
            "or an unexpected operational shift requiring investigation."
        )

    if anomaly:
        ctx["anomaly_context"] = anomaly

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 8 — Evaluation Quality
    # ═══════════════════════════════════════════════════════════════════════════
    eq: dict = {}

    eval_basis = forecast.get("evaluation_basis")
    if eval_basis:
        eq["evaluation_basis"] = eval_basis
        if eval_basis == "out_of_sample_backtest":
            eq["evaluation_note"] = (
                "Validated against a held-out test set — a real out-of-sample result "
                "is a stronger confidence signal than in-sample fit."
            )
        else:
            eq["evaluation_note"] = (
                "No held-out test was available — confidence is based on in-sample fit only. "
                "Treat with more caution than a backtested forecast."
            )

    bq = _safe_float(forecast.get("backtest_quality_score"))
    if bq is not None:
        eq["backtest_quality_score"] = round(bq, 1)
        eq["backtest_quality_grade"] = (
            "excellent" if bq >= 80 else ("acceptable" if bq >= 50 else "poor")
        )

    mase = _safe_float(forecast.get("backtest_mase"))
    if mase is not None:
        eq["backtest_mase"] = round(mase, 3)
        eq["mase_interpretation"] = (
            f"better than naive baseline by {(1 - mase) * 100:.0f}%"
            if mase < 1.0 else
            f"slightly worse than naive baseline (MASE={mase:.2f}) — forecast adds limited value"
        )

    cov = _safe_float(forecast.get("backtest_interval_coverage_pct"))
    if cov is not None:
        eq["backtest_interval_coverage_pct"] = round(cov, 1)

    conf_score = _safe_float(forecast.get("prediction_confidence_score"))
    if conf_score is not None:
        eq["prediction_confidence_score"] = round(conf_score, 1)

    reliability = forecast.get("forecast_reliability")
    if reliability:
        eq["forecast_reliability_label"] = reliability

    method = forecast.get("forecast_method")
    if method:
        eq["forecast_method"] = method
        _METHOD_DESC = {
            "stl_decomposition":             "STL trend decomposition (separates seasonal + trend components)",
            "linear_regression_bootstrap":   "Linear regression with bootstrap confidence intervals",
            "exponential_smoothing":         "Exponential smoothing",
            "shrunk_slope_xgboost_prior":    "Slope shrunk toward pooled XGBoost prior (thin history)",
            "pooled_type_prior":             "Pooled type-level prior (insufficient local history)",
            "flat_signal":                   "Flat signal — metric is not trending",
            "seasonal_naive_baseline":       "Seasonal-naive baseline (diurnal/weekly pattern)",
        }
        for k, v in _METHOD_DESC.items():
            if method == k or method.startswith(f"adaptive_{k}") or method.startswith(k):
                eq["forecast_method_description"] = v
                break

    pooling_tier = forecast.get("pooling_tier")
    if pooling_tier:
        eq["pooling_tier"] = pooling_tier

    n_pool = profile_info.get("n_resources_pooled")
    if n_pool is not None:
        eq["n_resources_in_pool"] = n_pool

    age_days = profile_info.get("profile_age_days")
    if age_days is not None:
        eq["profile_age_days"] = age_days

    prof_conf = _safe_float(profile_info.get("profile_confidence"))
    if prof_conf is not None:
        eq["profile_confidence"] = round(prof_conf, 2)

    completeness_raw = _safe_float(
        forecast.get("data_completeness_score") or prediction.get("data_completeness_score")
    )
    if completeness_raw is not None:
        # Field can be 0-1 or 0-100 depending on path
        pct = completeness_raw * 100 if completeness_raw <= 1.0 else completeness_raw
        eq["data_completeness_pct"] = round(pct, 1)
        if pct < 70:
            eq["data_completeness_warning"] = (
                f"Only {pct:.0f}% of expected metric slots have data — "
                f"missing telemetry reduces forecast accuracy."
            )

    data_note = forecast.get("data_note") or prediction.get("data_quality_note")
    if data_note and data_note not in ("ok", "ok+", ""):
        eq["data_quality_note"] = data_note

    # Plain-English pooling explanation
    pool_exp = _pooling_explanation(pooling_tier, n_pool, canon_type or None, age_days)
    if pool_exp:
        eq["pooling_explanation"] = pool_exp

    # Composite confidence summary sentence
    conf_parts: list[str] = []
    if eq.get("evaluation_basis") == "out_of_sample_backtest" and bq is not None:
        conf_parts.append(
            f"Backtested (quality score {bq:.0f}/100"
            + (f", MASE {mase:.2f}" if mase is not None else "")
            + (f", {cov:.0f}% interval coverage" if cov is not None else "")
            + ")."
        )
    elif eq.get("evaluation_basis"):
        conf_parts.append("In-sample fit only — no holdout backtest available.")
    if reliability and conf_score is not None:
        conf_parts.append(
            f"Reliability: {reliability} (confidence score {conf_score:.0f}/100)."
        )
    if pool_exp:
        conf_parts.append(pool_exp)
    if conf_parts:
        eq["confidence_summary"] = " ".join(conf_parts)

    if eq:
        ctx["evaluation_quality"] = eq

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 9 — Resource Metadata (from summary_details)
    # ═══════════════════════════════════════════════════════════════════════════
    summary_raw = prediction.get("summary_details")
    if summary_raw:
        metadata = _sanitize_summary_details(summary_raw)
        if metadata:
            ctx["resource_metadata"] = metadata

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 10 — Pipeline Signals
    # ═══════════════════════════════════════════════════════════════════════════
    recs = prediction.get("recommendations")
    if isinstance(recs, list) and recs:
        ctx["pipeline_recommendations"] = recs

    failures = prediction.get("failure_predictions")
    if isinstance(failures, list) and failures:
        ctx["detected_failure_risks"] = [
            {k: v for k, v in fp.items() if k != "_id"}
            for fp in failures
            if isinstance(fp, dict)
        ]

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 11 — Risk and Alert State
    # ═══════════════════════════════════════════════════════════════════════════
    risk_score_raw = dashboard.get("display_score")
    risk_score = _safe_float(risk_score_raw)
    if risk_score is not None:
        ctx["risk_score"] = int(round(risk_score))
        ctx["risk_label"] = _score_to_risk_label(risk_score)

    alert_state: dict = {}
    sev = alert_doc.get("severity")
    if sev:
        alert_state["severity"] = sev
    trigger = alert_doc.get("trigger")
    if trigger is not None:
        alert_state["alert_active"] = trigger
    if alert_state:
        ctx["alert_state"] = alert_state

    return ctx


# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────

RESOURCE_SYSTEM_PROMPT = """\
You are an experienced Cloud Engineer writing a per-resource health report for a software engineering team.
Your audience is software engineers who understand code but may not be cloud infrastructure experts.
Write as if a senior SRE colleague is explaining the situation in a detailed, jargon-free handover note.

You receive a structured "LLM Explanation Context" — all numbers are pre-computed. Do NOT perform any
arithmetic. Use every number exactly as given. Do NOT invent values that are absent from the context.

═══════════════════════════════════════════════════════════════
 WHAT TO WRITE IN EACH SECTION
═══════════════════════════════════════════════════════════════

resource_overview (3–4 sentences)
  • What is this resource? Name it fully: use resource_identity.resource_type_description which is
    dynamically built from the resource's own metadata (type, kind, provider, region). Add any
    specifics from resource_metadata (instance type, OS, owning team, environment, purpose).
  • Why does this resource exist? What workload does it serve? (use metadata / category_context)
  • Why does its health matter operationally? (cite category_context consequences)

current_state (3–4 sentences)
  • State the METRIC being monitored. Explain what it measures in plain English using
    metric_identity.metric_description. Name it as the provider names it (metric_identity.provider_metric_name).
  • Give the current value with its unit. Example: "CPU utilization is currently 82%, meaning the
    VM is using most of its available processing capacity."
  • Is the current value healthy, warning, or critical? How far from the warning threshold?
    Example: "The warning threshold is 80%, so this resource is already 2 percentage points past it."
  • If is_anomalous is true, explain WHY it is flagged as anomalous using zscore_note, burst_note,
    or isolation_forest_note. Translate the signal into plain English, not raw numbers.

trend_analysis (2–3 sentences)
  • Is the metric increasing, decreasing, or stable? Give the daily rate of change with units.
    Example: "CPU is rising at approximately 1.1% per day — at this rate the resource will consume
    an additional 7.7% over the next week."
  • Is the signal stable or volatile? Use signal_volatility to explain whether the forecast is
    precise or uncertain.
  • If regime_change_detected is true, explain what that means operationally.

forecast (4–6 sentences — one per available horizon)
  • For each horizon in forecast_horizons (1 hour, 24 hours, 7 days, 30 days), state the
    predicted value AND what it means operationally.
  • Example: "In 24 hours the model predicts CPU will reach 87.1% (WARNING level), meaning the
    server will be under sustained heavy load. By 7 days it is forecast to hit 91% (CRITICAL),
    at which point the system may begin dropping requests or becoming unresponsive."
  • If breach_analysis contains breach dates (warning_breach_date, critical_breach_date), cite
    the specific calendar date. Example: "At the current growth rate, CPU will breach the 90%
    critical threshold on approximately 2024-07-30."

evidence (2–3 sentences)
  • What data supports this prediction? Cite the forecast model (evaluation_quality.forecast_method)
    and whether it was backtested (evaluation_basis). Give the backtest quality score and MASE.
  • How complete is the telemetry? Cite data_completeness_pct if present.
  • Are there breach ETAs from a separate slope-based model (cpu_90pct_breach_eta, etc.)? Cite them.

operational_impact (3–5 sentences)
  • What SPECIFICALLY will happen if the trend continues? Be concrete — name the failure mode.
    "If CPU exceeds 90%, the application will begin experiencing request throttling, slower response
    times, and potentially OOM kills for memory-intensive processes."
  • Cite category_operational_signals: oom_probability, health_failure_probability_24h,
    burst_exhaustion_risk, ddos_probability, db_connections, unschedulable_pods, etc. — each with
    its value explained. Example: "There is a 23% probability of an OOM kill event in the next 24
    hours, meaning memory-intensive processes could be forcibly terminated."
  • If detected_failure_risks is present, explain each failure type in plain English with ETA.

recommended_actions (list of 3–5 strings, ordered most urgent first)
  • Each action must answer: WHAT to do, on WHICH resource (by name), WHY (cite the specific value
    that makes this urgent), and WHEN (immediately / within 24 hours / within 7 days / this week).
  • Use pipeline_recommendations as a starting point but expand them with timing and reasoning.
  • Example: "Immediately scale out the checkout-api auto-scaling group by 2 instances — CPU is
    at 83% and will breach the 90% critical threshold within ~6 days. Waiting risks request timeouts."
  • If the resource is stale (data_issue = stale_resource), the first action must be: verify the
    monitoring agent is running and report metrics are flowing before acting on the forecast.

confidence (2–3 sentences)
  • How reliable is this prediction? Use evaluation_quality.confidence_summary if present.
    Otherwise compose from: evaluation_basis (in-sample or out-of-sample backtest),
    backtest_quality_grade (excellent/acceptable/poor), backtest_mase (is it better than naive?),
    pooling_explanation (individual vs pooled), and profile_age_days.
  • Tell the engineer whether they should ACT on this prediction confidently or treat it cautiously.
    Example: "This forecast is backed by an out-of-sample backtest (quality score 74/100, MASE 0.61
    — better than naive baseline) using 45 days of individual history. Confidence is moderate;
    treat the exact breach date as an approximation within ±2–3 days."

═══════════════════════════════════════════════════════════════
 ABSOLUTE RULES
═══════════════════════════════════════════════════════════════
1. NEVER say "CPU is high" — always give the actual value: "CPU is at 83%".
2. NEVER say "the metric is increasing" without the rate: "rising at ~1.1%/day".
3. NEVER invent numbers — only use values from the context.
4. NEVER include account_id, resource_id, project_id, tenant_id, subscription_id,
   SSH keys, passwords, tokens, access keys, connection strings, or credentials.
5. ALWAYS explain what the metric measures in the first mention of it.
6. If a section has no data (e.g. no anomaly signals, no breach dates), say so briefly
   and don't pad with generic filler.
7. Convert bytes/s to KB/s or MB/s for readability. Convert very long timestamps to dates.
8. If anomaly_context.data_issue is "stale_resource", note prominently that telemetry
   has not been received recently and the forecast is based on stale data.

═══════════════════════════════════════════════════════════════
 RESPONSE FORMAT — valid JSON only, no markdown fences
═══════════════════════════════════════════════════════════════
{
  "resource_overview":   "string — 3-4 sentences",
  "current_state":       "string — 3-4 sentences explaining metric, value, and proximity to threshold",
  "trend_analysis":      "string — 2-3 sentences with rate and volatility",
  "forecast":            "string — 4-6 sentences, one per horizon, with operational meaning",
  "evidence":            "string — 2-3 sentences on model, backtest, completeness",
  "operational_impact":  "string — 3-5 sentences on specific failure modes and cited risk values",
  "recommended_actions": ["string (most urgent, with timing)", "string", "string"],
  "confidence":          "string — 2-3 sentences on reliability and how much to trust the forecast",
  "risk_level":          "LOW"
}
risk_level must be exactly one of: "LOW", "MEDIUM", "HIGH", "CRITICAL"
"""


# ─────────────────────────────────────────────────────────────────────────────
# Payload and prompt construction
# ─────────────────────────────────────────────────────────────────────────────

def build_resource_payload(prediction: dict) -> dict:
    """Build the complete LLM payload from a prediction document.

    Calls build_llm_explanation_context() for all derived facts, then wraps
    it with the resource identity section (type descriptions, category context).

    Sensitive fields are never included: account_id, subscription_id, tenant_id,
    raw resource_id, SSH config, access keys, connection strings, raw tags.
    """
    forecast   = prediction.get("behavioral_forecast") if isinstance(prediction.get("behavioral_forecast"), dict) else {}
    feature    = str(forecast.get("canonical_feature") or "").strip()
    canon_type = str(prediction.get("canonical_resource_type") or "").strip()
    category   = str(prediction.get("category") or "").strip()
    provider   = str(prediction.get("provider") or "").strip()

    # Build the rich explanation context (all arithmetic happens here)
    explanation_context = build_llm_explanation_context(prediction)

    # Resource identity section — built dynamically from available metadata.
    # No static lookup tables: description and category context are derived
    # from summary_details, component, service_name, provider, and location.
    resource_identity = {
        "resource_name":             prediction.get("resource_name") or "Unnamed resource",
        "resource_type":             _extract_resource_type_label(prediction),
        "resource_type_description": _build_resource_type_description(prediction),
        "category":                  category or "unknown",
        "category_context":          _build_category_context(category, prediction),
        "cloud_provider":            provider or "unknown",
        "region":                    prediction.get("location") or "unknown region",
        "component":                 prediction.get("component") or None,
        "service_name":              prediction.get("service_name") or None,
    }

    # Metric identity — which specific metric this forecast covers.
    # Prefer the provider's actual metric_name; derive display name and
    # description from the canonical feature key pattern if it is absent.
    metric_identity: dict = {}
    mn = forecast.get("metric_name")
    if mn:
        metric_identity["provider_metric_name"] = mn
    if feature:
        metric_identity["canonical_feature_key"] = feature
        metric_identity["display_name"]          = _derive_metric_display_name(feature, forecast)
        metric_identity["metric_description"]    = _derive_metric_description(feature, forecast, category)
        unit = _METRIC_UNITS.get(feature) or _derive_unit_from_feature(feature)
        if unit:
            metric_identity["unit"] = unit
        if feature in _INVERTED_METRICS:
            metric_identity["is_inverted_metric"] = True
            metric_identity["inverted_note"] = (
                "For this metric, LOWER values indicate worse health. "
                "The metric drains toward 0 (or a minimum) under stress, "
                "not rises toward 100."
            )
        thresholds = _PCT_THRESHOLDS.get(feature)
        if thresholds:
            metric_identity["warning_threshold"]  = thresholds[0]
            metric_identity["critical_threshold"] = thresholds[1]

    return {
        "resource_identity":    resource_identity,
        "metric_identity":      metric_identity if metric_identity else None,
        "explanation_context":  explanation_context,
    }


def build_resource_prompt(prediction: dict, breach_states: list | None = None) -> str:
    """Build the user-turn prompt from pre-computed facts."""
    payload = build_resource_payload(prediction)
    if breach_states:
        payload["threshold_breach_history"] = breach_states

    ri  = payload["resource_identity"]
    ec  = payload["explanation_context"]
    cs  = ec.get("current_state") or {}
    eq  = ec.get("evaluation_quality") or {}
    dm  = ec.get("dominant_metrics") or []

    # Compact header — gives the LLM the most critical facts at a glance
    # before it encounters the full JSON body.
    header_lines: list[str] = [
        f"Resource: {ri['resource_name']}",
        f"Type: {ri['resource_type']} ({ri['category']}) | "
        f"Provider: {ri['cloud_provider']} | Region: {ri['region']}",
        f"What it is: {ri['resource_type_description']}",
        f"Category context: {ri['category_context']}",
    ]

    cur_val = cs.get("current_value")
    cur_unit = cs.get("unit", "")
    health = cs.get("health_status", "UNKNOWN")
    if cur_val is not None:
        header_lines.append(
            f"Current metric: {cur_val}{cur_unit}  |  Health: {health}  |  "
            f"Risk: {ec.get('risk_label', 'UNKNOWN')} ({ec.get('risk_score', '?')}/100)"
        )

    # Surface the most critical non-primary signal so the LLM doesn't miss it
    critical_siblings = [d for d in dm if not d.get("is_primary_forecast_target") and d.get("severity") in ("CRITICAL", "WARNING")]
    if critical_siblings:
        top = critical_siblings[0]
        cv  = top.get("current_value") or top.get("classification", "")
        header_lines.append(
            f"Attention — sibling signal: {top['metric_name']} = {cv}{top.get('unit','')}"
            f" [{top['severity']}]"
        )

    reliability = eq.get("forecast_reliability_label", "low")
    eval_basis  = eq.get("evaluation_basis", "in_sample_fit_only")
    header_lines.append(
        f"Forecast reliability: {reliability}  |  Evaluation basis: {eval_basis}"
    )

    header = (
        "\n".join(header_lines)
        + "\n\nFull LLM Explanation Context below — use these pre-computed facts to write "
        "the report. Do NOT invent numbers or recalculate anything:\n"
    )
    return header + json.dumps(payload, indent=2, default=str)


# ─────────────────────────────────────────────────────────────────────────────
# Per-resource explanation and batch orchestration
# ─────────────────────────────────────────────────────────────────────────────

def explain_resource(
    client,
    prediction: dict,
    breach_states: list | None = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 2000,
) -> dict:
    """Call the LLM and return the structured 8-section explanation."""
    prompt = build_resource_prompt(prediction, breach_states)
    result = _call_llm(client, RESOURCE_SYSTEM_PROMPT, prompt, model=model, max_tokens=max_tokens)

    required_sections = (
        "resource_overview", "current_state", "trend_analysis", "forecast",
        "evidence", "operational_impact", "recommended_actions", "confidence",
    )
    # If the LLM returned a JSON-parse error, the narrative lives in "narrative"
    # or "explanation". Distribute it across required sections so the UI renders
    # something rather than empty strings.
    fallback_prose = result.get("explanation") or result.get("narrative") or ""
    for sec in required_sections:
        if sec not in result:
            result[sec] = [] if sec == "recommended_actions" else fallback_prose

    # Normalize risk_level: the LLM sometimes emits "HEALTHY", "None", or other
    # non-canonical values. Always resolve to LOW/MEDIUM/HIGH/CRITICAL.
    result["risk_level"] = _normalize_risk_level(result.get("risk_level"), prediction)

    # If the LLM response had a JSON parse error, mark it so the backend can
    # filter it out of the UI instead of showing the error prose as content.
    if result.get("_parse_error"):
        result["_error"] = True

    # Attach identifying metadata
    forecast = prediction.get("behavioral_forecast") if isinstance(prediction.get("behavioral_forecast"), dict) else {}
    result["resource_id"]    = prediction.get("resource_id")
    result["resource_name"]  = prediction.get("resource_name")
    result["category"]       = prediction.get("category")
    result["provider"]       = prediction.get("provider")
    result["project"]        = prediction.get("project")
    result["location"]       = prediction.get("location")
    result["metric_name"]    = forecast.get("metric_name")
    result["computed_stats"] = build_llm_explanation_context(prediction)
    result["thresholds"]     = breach_states or []
    result["generated_at"]   = datetime.now(timezone.utc).isoformat()
    result["model"]          = model
    return result


def explain_all_resources(
    predictions,
    breach_states_by_resource: dict | None = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 2000,
    client=None,
    provider=None,
    on_resource_done=None,
) -> list:
    """Generate one structured explanation per resource_id (latest prediction wins)."""
    client = client or _get_client(provider)
    breach_states_by_resource = breach_states_by_resource or {}

    latest_by_resource: dict = {}
    for p in predictions:
        rid = p.get("resource_id")
        if not rid:
            continue
        ts = p.get("prediction_timestamp") or ""
        if rid not in latest_by_resource or ts > (latest_by_resource[rid].get("prediction_timestamp") or ""):
            latest_by_resource[rid] = p

    explanations = []
    for resource_id, prediction in latest_by_resource.items():
        try:
            explanation = explain_resource(
                client, prediction,
                breach_states_by_resource.get(resource_id),
                model=model, max_tokens=max_tokens,
            )
        except Exception as exc:
            explanation = {
                "resource_id":        resource_id,
                "resource_name":      prediction.get("resource_name"),
                "category":           prediction.get("category"),
                "provider":           prediction.get("provider"),
                "project":            prediction.get("project"),
                "location":           prediction.get("location"),
                "resource_overview":  f"Explanation generation failed: {exc}",
                "current_state":      "",
                "trend_analysis":     "",
                "forecast":           "",
                "evidence":           "",
                "operational_impact": "",
                "recommended_actions": [],
                "confidence":         "",
                # Derive from the prediction's own score — UNKNOWN only for
                # _error docs so the backend can filter them from the UI.
                "risk_level":         _normalize_risk_level(None, prediction),
                "computed_stats":     {},
                "thresholds":         breach_states_by_resource.get(resource_id, []),
                "generated_at":       datetime.now(timezone.utc).isoformat(),
                "model":              model,
                "_error":             True,
            }
        explanations.append(explanation)
        if on_resource_done:
            on_resource_done(resource_id, explanation)

    return explanations


def write_resource_explanations(mongo_uri: str, db_name: str, explanations: list) -> int:
    """Upsert one document per resource_id — latest explanation replaces previous.

    Idempotent: re-running for the same resource_id overwrites the previous doc.
    Uses ordered=False so a single bad document doesn't block the rest of the batch.
    """
    from src.mongo_io import get_collection, _bson_safe
    from pymongo import UpdateOne
    from pymongo.errors import BulkWriteError

    if not explanations:
        return 0
    col = get_collection(mongo_uri, db_name, RESOURCE_EXPLANATION_COLLECTION)
    ops = []
    for e in explanations:
        if not e.get("resource_id"):
            continue
        safe = _bson_safe(e)
        ops.append(UpdateOne(
            {"resource_id": safe["resource_id"]},
            {"$set": safe},
            upsert=True,
        ))
    if not ops:
        return 0
    try:
        result = col.bulk_write(ops, ordered=False)
        return result.upserted_count + result.modified_count
    except BulkWriteError as bwe:
        # ordered=False: some writes may have succeeded before the error.
        # Log and return the partial success count rather than crashing.
        succeeded = bwe.details.get("nUpserted", 0) + bwe.details.get("nModified", 0)
        print(f"  [write_resource_explanations] partial write: {succeeded}/{len(ops)} succeeded. "
              f"Errors: {bwe.details.get('writeErrors', [])[:3]}")
        return succeeded


def ensure_resource_explanation_indexes(mongo_uri: str, db_name: str) -> str:
    from src.mongo_io import get_collection
    col = get_collection(mongo_uri, db_name, RESOURCE_EXPLANATION_COLLECTION)
    col.create_index("resource_id", unique=True)
    col.create_index([("project", 1), ("generated_at", -1)])
    return RESOURCE_EXPLANATION_COLLECTION


# ─────────────────────────────────────────────────────────────────────────────
# Command-line interface
# ─────────────────────────────────────────────────────────────────────────────

def _parser():
    import argparse
    p = argparse.ArgumentParser(
        description=(
            "Per-resource LLM explanation job. Reads predictions from MongoDB "
            "and writes a structured 8-section explanation per resource."
        )
    )
    p.add_argument("--mongo-uri", default="mongodb://localhost:27017")
    p.add_argument("--db", default="mydb",
                   help="Database to read predictions from (and write to unless --output-db is set).")
    p.add_argument("--output-db", default=None,
                   help="Write explanations to this DB; defaults to --db.")
    p.add_argument("--window-minutes", type=int, default=525600,
                   help="Lookback window for predictions in minutes "
                        "(default 525600 = 1 year, covers any static dataset). "
                        "Predictions are matched by prediction_timestamp >= now - window.")
    p.add_argument("--mode", default="all", choices=["priority", "all"],
                   help="'all' explains every resource in the window. "
                        "'priority' only explains resources that are anomalous or alerting.")
    p.add_argument("--provider", default=DEFAULT_PROVIDER, choices=["ollama", "anthropic"],
                   help="LLM provider. 'ollama' (default) runs fully local, no API key. "
                        "'anthropic' needs ANTHROPIC_API_KEY.")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--max-tokens", type=int, default=1200)
    p.add_argument("--compute-out", default="22prediction_logs_compute")
    p.add_argument("--out", default="22prediction_logs_storage",
                   help="Storage prediction collection name.")
    p.add_argument("--network-out", default="22prediction_logs_network")
    p.add_argument("--container-out", default="22prediction_logs_container")
    p.add_argument("--databases-out", default="22prediction_logs_databases")
    p.add_argument("--dry-run", action="store_true",
                   help="Explain the first resource only and print to stdout; do not write to MongoDB.")
    return p


def main():
    args = _parser().parse_args()
    output_db = args.output_db or args.db

    # Load predictions using the same loader as scripts/llm_project_summary.py.
    # Set COLLECTION_NAME_OVERRIDES before calling so the correct collection
    # names are used (mirrors what run_pipeline.py does).
    import scripts.llm_project_summary as llm_script
    llm_script.COLLECTION_NAME_OVERRIDES = {
        "Compute":   args.compute_out,
        "Storage":   args.out,
        "Network":   args.network_out,
        "Container": args.container_out,
        "Databases": args.databases_out,
    }

    print(f"Pulling predictions from last {args.window_minutes} minute(s) in {output_db}...")
    predictions = llm_script.load_window_predictions(args.mongo_uri, output_db, args.window_minutes)
    print(f"Total predictions in window: {len(predictions)}")

    if not predictions:
        print("Nothing to explain.")
        return 0

    if args.mode == "priority":
        to_explain = [
            p for p in predictions
            if p.get("is_anomalous") or (p.get("alert") or {}).get("severity", "INFO") != "INFO"
        ]
        print(f"Mode=priority: {len(to_explain)}/{len(predictions)} resource(s) meet criteria.")
    else:
        to_explain = predictions
        print(f"Mode=all: explaining all {len(to_explain)} resource(s).")

    if not to_explain:
        print("No resources meet the explain criteria.")
        return 0

    if args.dry_run:
        import json
        sample = explain_all_resources(
            to_explain[:1], model=args.model, max_tokens=args.max_tokens, provider=args.provider,
        )
        print(json.dumps(sample, indent=2, default=str))
        return 0

    ensure_resource_explanation_indexes(args.mongo_uri, output_db)
    written = [0]

    def _flush_one(resource_id, explanation):
        n = write_resource_explanations(args.mongo_uri, output_db, [explanation])
        written[0] += n
        print(f"  wrote explanation for {resource_id!r} (risk={explanation.get('risk_level')})")

    explain_all_resources(
        to_explain, model=args.model, max_tokens=args.max_tokens,
        provider=args.provider, on_resource_done=_flush_one,
    )
    print(f"Wrote {written[0]} explanation(s) -> {output_db}.{RESOURCE_EXPLANATION_COLLECTION}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
