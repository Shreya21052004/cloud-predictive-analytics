"""llm_summary.py — Per-project LLM aggregation layer.

Sits AFTER prediction generation, as a separate consumer, not inside
predict_from_df(). Reason: Compute/Storage/Network/Container/Databases
predictions each land in their own collection (see
config.CATEGORY_OUTPUT_COLLECTION) and are produced by potentially
different pipeline runs. A per-project, per-window summary has to read
across all of those collections for a time window, so it can't be a step
bolted onto a single category's predict() call -- it has to be its own
scheduled job. See scripts/llm_project_summary.py for the entrypoint that
is meant to run on a cron (every 1-2h, matching your window).

Design:
  1. Pull predictions from every category collection with
     prediction_timestamp inside [window_start, window_end).
  2. Group by `project` (never by resource_id -- one LLM call must be able
     to reason across categories on the same project, that's the point).
  3. Build a compact one-line digest per resource/category prediction
     instead of sending the full prediction document (payload, full
     forecast arrays, etc. are 90% dead weight for an LLM prompt).
  4. Budget by tokens, not resource count. If a project's digests fit in
     one call, summarize directly. If not, chunk (preferring category
     boundaries), summarize each chunk, then do one "reduce" call over the
     chunk summaries so the project still gets exactly ONE final summary
     document.
  5. Write one document per (project, window) to LLM_SUMMARY_COLLECTION.
"""
import json
import os
import time
from datetime import datetime, timezone

LLM_SUMMARY_COLLECTION = "22llm_project_summaries"

# Rough chars-per-token for English/JSON-ish text. Good enough for budgeting
# a prompt; we're not trying to match the real tokenizer exactly, just stay
# safely under the limit with margin.
CHARS_PER_TOKEN = 4

# ---------------------------------------------------------------------------
# LLM provider selection
# ---------------------------------------------------------------------------
# Ollama is the default: a local model server, no API key, no quota, no
# network egress. Point OLLAMA_HOST at a remote Ollama instance if you're not
# running it on the same box as the pipeline. Anthropic remains available as
# an opt-in cloud fallback (LLM_PROVIDER=anthropic + ANTHROPIC_API_KEY) for
# anyone who wants it, but it is no longer required for anything.
DEFAULT_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama").lower()

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")

_PROVIDER_DEFAULT_MODELS = {
    "ollama": "qwen3:8b",
    "anthropic": "claude-sonnet-4-6",
}

DEFAULT_MODEL = os.environ.get("LLM_MODEL", _PROVIDER_DEFAULT_MODELS.get(DEFAULT_PROVIDER, "qwen3:8b"))

# Leave headroom for the fixed prompt scaffolding + the model's own output.
DEFAULT_MAX_DIGEST_TOKENS = 4000


# ---------------------------------------------------------------------------
# 1. Metric knowledge — display names, units, thresholds, descriptions
# ---------------------------------------------------------------------------

def _derive_metric_info(prediction: dict) -> tuple[str, str, str]:
    """Derive (display_name, unit, description) from the prediction document.

    Uses the actual provider metric_name from behavioral_forecast.metric_name
    first, then falls back to pattern-matching on the canonical feature key.
    No static lookup table — works for any cloud provider or resource type.
    Returns (display_name, unit, description).
    """
    forecast = prediction.get("behavioral_forecast") if isinstance(prediction.get("behavioral_forecast"), dict) else {}
    feature  = str(forecast.get("canonical_feature") or "").strip()
    actual   = str(forecast.get("metric_name") or "").strip()
    category = str(prediction.get("category") or "").strip()

    # Display name: prefer provider's actual metric name
    if actual:
        display_name = actual
    elif feature:
        display_name = feature.replace("canonical_", "").replace("_", " ").title()
    else:
        display_name = "Unknown Metric"

    # Unit from feature key naming conventions
    f = feature.lower()
    if f.endswith("_pct") or "pct" in f:
        unit = "%"
    elif "_bytes" in f:
        unit = "bytes/s"
    elif "_iops" in f:
        unit = "IOPS"
    elif "_ms" in f:
        unit = "ms"
    elif "_connections" in f:
        unit = "connections"
    elif "dtu" in f:
        unit = "DTUs"
    elif "_pods" in f:
        unit = "pods"
    elif "_requests" in f:
        unit = "requests"
    elif "_nodes" in f:
        unit = "nodes"
    else:
        unit = ""

    # Plain-English description from feature key pattern
    provider_note = f" (provider metric: {actual})" if actual else ""
    if "cpu" in f and "idle" in f:
        desc = f"CPU idle time — 100% = fully idle, falling means increasing load{provider_note}"
    elif "cpu" in f:
        desc = f"percentage of total CPU capacity in use — higher = processor is busier{provider_note}"
    elif "mem" in f:
        desc = f"percentage of RAM in use — higher = less memory available for workloads{provider_note}"
    elif "burst" in f:
        desc = f"I/O burst credit balance — drains under sustained load; 0% = throttled{provider_note}"
    elif "storage" in f and "iops" in f:
        desc = f"storage I/O operations per second{provider_note}"
    elif "storage" in f and "bytes" in f:
        desc = f"storage throughput in bytes per second{provider_note}"
    elif "storage" in f:
        desc = f"percentage of disk/volume capacity consumed — 100% = full{provider_note}"
    elif "disk" in f and "bytes" in f:
        desc = f"disk I/O throughput in bytes per second{provider_note}"
    elif "net" in f and "in" in f:
        desc = f"inbound network throughput{provider_note}"
    elif "net" in f and "out" in f:
        desc = f"outbound network throughput{provider_note}"
    elif "health" in f:
        desc = f"% of backend targets passing health checks — lower = more backends failing{provider_note}"
    elif "availability" in f:
        desc = f"service availability % — below 99% = significant downtime{provider_note}"
    elif "subnet" in f:
        desc = f"% of subnet IPs allocated — exhaustion blocks new resources{provider_note}"
    elif "db" in f and "connection" in f:
        desc = f"number of active database connections{provider_note}"
    elif "db" in f and "cpu" in f:
        desc = f"CPU used by the database engine{provider_note}"
    elif "db" in f and "mem" in f:
        desc = f"memory used by the database engine{provider_note}"
    elif "db" in f and "storage" in f:
        desc = f"database storage space consumed{provider_note}"
    elif "dtu" in f:
        desc = f"Azure SQL Database Transaction Units consumed{provider_note}"
    elif "node" in f and "cpu" in f:
        desc = f"average CPU utilization across cluster nodes{provider_note}"
    elif "node" in f and "mem" in f:
        desc = f"average memory utilization across cluster nodes{provider_note}"
    elif "pod" in f or "unschedulable" in f:
        desc = f"pods unable to schedule due to node resource pressure{provider_note}"
    elif "latency" in f or "_ms" in f:
        desc = f"response latency in milliseconds{provider_note}"
    elif "iops" in f:
        desc = f"I/O operations per second{provider_note}"
    elif "saturation" in f:
        desc = f"resource saturation/queue depth as a percentage{provider_note}"
    elif "ddos" in f:
        desc = f"DDoS attack probability signal{provider_note}"
    elif "failure" in f:
        desc = f"execution failure rate as a percentage{provider_note}"
    elif "throttle" in f:
        desc = f"count of rate-limited/throttled requests{provider_note}"
    elif "connection" in f:
        desc = f"count of active network connections{provider_note}"
    else:
        desc = actual or f"{category.lower() or 'cloud'} infrastructure metric{provider_note}"

    return display_name, unit, desc

# (warning, critical) — for %-based metrics only.
# Inverted metrics: LOWER is worse (e.g. burst balance draining = bad).
_DIGEST_THRESHOLDS = {
    "canonical_cpu_pct":           (80,  90,  False),
    "canonical_mem_used_pct":      (80,  90,  False),
    "canonical_storage_used_pct":  (80,  90,  False),
    "canonical_node_cpu_pct":      (80,  90,  False),
    "canonical_node_mem_pct":      (80,  90,  False),
    "canonical_db_cpu_pct":        (80,  90,  False),
    "canonical_db_mem_pct":        (80,  90,  False),
    "canonical_db_storage_pct":    (85,  95,  False),
    "canonical_saturation":        (75,  90,  False),
    "canonical_subnet_util_pct":   (75,  90,  False),
    "canonical_burst_balance_pct": (20,  10,  True),   # inverted
    "canonical_health_pct":        (90,  80,  True),   # inverted
    "canonical_availability_pct":  (99,  95,  True),   # inverted
}


def _format_value(value, unit):
    """Format a numeric value with its unit for the digest."""
    if value is None:
        return "N/A"
    try:
        f = float(value)
        if unit == "%":
            return f"{f:.1f}%"
        if unit in ("bytes/s", "IOPS"):
            if f >= 1_073_741_824:
                return f"{f/1_073_741_824:.2f} GB/s"
            if f >= 1_048_576:
                return f"{f/1_048_576:.2f} MB/s"
            if f >= 1024:
                return f"{f/1024:.1f} KB/s"
            return f"{f:.0f} {unit}"
        if unit == "ms":
            return f"{f:.0f}ms"
        if unit in ("connections", "pods", "requests", "DTUs", "nodes"):
            return f"{f:.0f} {unit}"
        return f"{f:.3g} {unit}" if unit else f"{f:.3g}"
    except (TypeError, ValueError):
        return str(value)


def _severity_label(value, feature):
    """HEALTHY / WARNING / CRITICAL based on known thresholds."""
    if value is None or feature not in _DIGEST_THRESHOLDS:
        return "UNKNOWN"
    warn, crit, inverted = _DIGEST_THRESHOLDS[feature]
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "UNKNOWN"
    if inverted:
        if v <= crit: return "CRITICAL"
        if v <= warn: return "WARNING"
        return "HEALTHY"
    else:
        if v >= crit: return "CRITICAL"
        if v >= warn: return "WARNING"
        return "HEALTHY"


def _compact_tags(tags):
    if not tags:
        return None
    if isinstance(tags, dict):
        # Only meaningful, short tags — skip auto-generated cloud labels
        skip_prefixes = ("goog-", "aws:", "kubernetes.io/", "k8s.io/")
        items = [(k, v) for k, v in tags.items()
                 if not any(str(k).lower().startswith(p) for p in skip_prefixes)]
        s = ",".join(f"{k}={v}" for k, v in items[:6])
    elif isinstance(tags, (list, tuple)):
        s = ",".join(
            f"{t.get('key','?')}={t.get('value','?')}" if isinstance(t, dict) else str(t)
            for t in tags[:6]
        )
    else:
        s = str(tags)
    return (s[:120] + "…") if len(s) > 123 else s or None


def _safe_summary_snippet(summary_details):
    """Extract a short, human-readable context snippet from summary_details.
    Picks the most useful fields (name, type, OS, environment, team, purpose)
    and drops credential/URL values."""
    if not summary_details:
        return None
    SKIP_KEY_FRAGS = ("password", "key", "secret", "token", "cert", "url",
                      "uri", "ssh", "connection", "endpoint", "private", "access")
    if isinstance(summary_details, dict):
        useful = {}
        PREFER_KEYS = ("name", "type", "os", "environment", "env", "team", "purpose",
                       "application", "service", "role", "instance_type", "machine_type",
                       "size", "tier", "sku", "location", "region", "storageClass",
                       "locationType", "description")
        for k in PREFER_KEYS:
            v = summary_details.get(k)
            if v is None:
                # case-insensitive fallback
                for dk, dv in summary_details.items():
                    if str(dk).lower() == k:
                        v = dv
                        break
            if v and isinstance(v, str) and not any(f in str(k).lower() for f in SKIP_KEY_FRAGS):
                useful[k] = v[:60]
        if not useful:
            # fall back to first few safe string fields
            for k, v in summary_details.items():
                if isinstance(v, str) and not any(f in str(k).lower() for f in SKIP_KEY_FRAGS):
                    useful[k] = v[:60]
                if len(useful) >= 4:
                    break
        s = "; ".join(f"{k}={v}" for k, v in useful.items())
        return (s[:200] + "…") if len(s) > 200 else s or None
    s = str(summary_details).strip()
    return (s[:200] + "…") if len(s) > 200 else s or None


# ---------------------------------------------------------------------------
# 1. Digest construction — what actually goes into the prompt
# ---------------------------------------------------------------------------

def build_resource_digest(prediction):
    """Build a structured, metric-aware digest block for one prediction.

    Unlike the old one-liner, this produces a small structured block that
    tells the LLM:
      - What the resource IS (type, provider, region, purpose/team from summary_details)
      - What metric is being forecasted (human name, what it measures, unit)
      - The ACTUAL numbers with units and what they mean (current, 24h, 7d, 30d)
      - Whether the numbers are concerning (HEALTHY / WARNING / CRITICAL)
      - The trend in plain English and reliability of the forecast
      - Any anomalies or failure risks
    This costs a few more tokens per resource but removes the ambiguity that
    causes LLMs to produce vague or incorrect explanations.
    """
    forecast = prediction.get("behavioral_forecast") or {}
    dashboard = prediction.get("dashboard") or {}
    alert     = prediction.get("alert") or {}
    failures  = prediction.get("failure_predictions") or []
    payload   = prediction.get("payload") or {}

    display_name, unit, metric_desc = _derive_metric_info(prediction)
    feature   = str((forecast.get("canonical_feature") or "")).strip()

    current   = forecast.get("current_value")
    f24h      = forecast.get("forecast_24h")
    f7d       = forecast.get("forecast_7d")
    f30d      = forecast.get("forecast_30d")
    trend_dir = forecast.get("trend_direction") or dashboard.get("trend") or "unknown"
    severity  = _severity_label(current, feature)
    risk_score = dashboard.get("display_score", 0)

    # Build a daily growth rate in human terms
    growth_note = ""
    if current is not None and f7d is not None:
        try:
            daily = (float(f7d) - float(current)) / 7.0
            if unit == "%" and abs(daily) >= 0.01:
                direction = "rising" if daily > 0 else "falling"
                growth_note = f", {direction} ~{abs(daily):.2f}{unit}/day"
        except (TypeError, ValueError):
            pass

    # Resource context from summary_details
    context_snippet = _safe_summary_snippet(prediction.get("summary_details"))

    lines = [
        f"--- Resource: {prediction.get('resource_name') or prediction.get('resource_id', '?')} ---",
        f"  category       : {prediction.get('category', '?')} | type: {prediction.get('canonical_resource_type') or prediction.get('component', '?')}",
        f"  provider/region: {prediction.get('provider', '?')} / {prediction.get('location', '?')}",
    ]
    if context_snippet:
        lines.append(f"  context        : {context_snippet}")
    tags_str = _compact_tags(prediction.get("tags"))
    if tags_str:
        lines.append(f"  tags           : {tags_str}")

    lines += [
        f"  metric         : {display_name} — {metric_desc}" if metric_desc else f"  metric         : {display_name}",
        f"  current        : {_format_value(current, unit)} [{severity}]",
        f"  forecast 24h   : {_format_value(f24h, unit)}",
        f"  forecast 7d    : {_format_value(f7d, unit)}",
        f"  forecast 30d   : {_format_value(f30d, unit)}",
        f"  trend          : {trend_dir}{growth_note} | reliability: {forecast.get('forecast_reliability', '?')}",
        f"  risk score     : {risk_score}/100 | anomalous: {prediction.get('is_anomalous', False)} | alert: {alert.get('severity', 'INFO')}",
    ]

    # Threshold context so LLM knows when to worry
    if feature in _DIGEST_THRESHOLDS:
        warn, crit, inv = _DIGEST_THRESHOLDS[feature]
        inv_note = " (lower is worse — metric drains toward 0)" if inv else ""
        lines.append(f"  thresholds     : warn={warn}{unit}, critical={crit}{unit}{inv_note}")

    # Category-specific operational signals
    ops_items = []
    if payload.get("oom_probability") is not None:
        ops_items.append(f"OOM_risk={payload['oom_probability']*100:.0f}%")
    if payload.get("health_failure_probability_24h") is not None:
        ops_items.append(f"health_fail_24h={payload['health_failure_probability_24h']*100:.0f}%")
    if payload.get("memory_pressure_class") and payload["memory_pressure_class"] not in ("UNKNOWN", ""):
        ops_items.append(f"memory_pressure={payload['memory_pressure_class']}")
    if payload.get("burst_exhaustion_risk") is not None:
        ops_items.append(f"burst_exhaustion={payload['burst_exhaustion_risk']*100:.0f}%")
    if payload.get("ddos_probability") is not None:
        ops_items.append(f"ddos_risk={payload['ddos_probability']*100:.0f}%")
    if payload.get("db_connections") is not None:
        ops_items.append(f"db_connections={payload['db_connections']:.0f}")
    if payload.get("unschedulable_pods") is not None and payload["unschedulable_pods"] > 0:
        ops_items.append(f"unschedulable_pods={payload['unschedulable_pods']:.0f}")
    if ops_items:
        lines.append(f"  ops signals    : {', '.join(ops_items)}")

    # Failure risks and recommendations
    top_failure = failures[0] if failures else None
    if top_failure:
        label = top_failure.get("type") or top_failure.get("failure_type") or "risk"
        eta   = top_failure.get("eta") or top_failure.get("time_to_failure")
        lines.append(f"  failure risk   : {label}" + (f" (ETA: {eta})" if eta else ""))
    recs = prediction.get("recommendations") or []
    if recs:
        lines.append(f"  recommendations: {'; '.join(str(r) for r in recs[:2])}")

    return "\n".join(lines)


def estimate_tokens(text):
    return max(1, len(text) // CHARS_PER_TOKEN)


# ---------------------------------------------------------------------------
# 2. Grouping + chunking
# ---------------------------------------------------------------------------

def group_predictions_by_project(predictions):
    by_project = {}
    for p in predictions:
        project = p.get("project") or "UNASSIGNED"
        by_project.setdefault(project, []).append(p)
    return by_project


def build_resource_roster(predictions):
    """Deterministic (non-LLM) list of every distinct resourceId in this
    project's window, with its tags, category, forecasted metric values,
    and risk/anomaly state. Built directly from the predictions rather
    than asked of the LLM, so it's always complete and exact -- an LLM
    summarizing under a token budget can drop or merge resources it isn't
    the source of truth for the id/tag/forecast values.

    One entry per resource_id (latest prediction wins if a resource shows
    up more than once in the window, e.g. across several predict cycles).
    """
    roster = {}
    for p in predictions:
        rid = p.get("resource_id")
        if not rid:
            continue
        forecast = p.get("behavioral_forecast") or {}
        dashboard = p.get("dashboard") or {}
        roster[rid] = {
            "resource_id":   rid,
            "resource_name": p.get("resource_name"),
            "category":      p.get("category"),
            "tags":          p.get("tags"),
            "summary_details": p.get("summary_details"),
            "forecast": {
                "current_value":      forecast.get("current_value"),
                "forecast_1h":        forecast.get("forecast_1h"),
                "forecast_6h":        forecast.get("forecast_6h"),
                "forecast_24h":       forecast.get("forecast_24h"),
                "forecast_7d":        forecast.get("forecast_7d"),
                "forecast_30d":       forecast.get("forecast_30d"),
                "trend_direction":    forecast.get("trend_direction"),
                "forecast_reliability": forecast.get("forecast_reliability"),
            },
            "risk_score":    dashboard.get("display_score"),
            "is_anomalous":  p.get("is_anomalous", False),
            "recommendations": p.get("recommendations", []),
            "prediction_timestamp": p.get("prediction_timestamp"),
        }
    return sorted(roster.values(), key=lambda r: (r["category"] or "", r["resource_id"]))


def _sort_key(prediction):
    # Most important resources first, so if we ever DO have to truncate
    # (shouldn't happen with map-reduce, but as a hard safety cap) we drop
    # low-signal resources, not the CRITICAL ones.
    sev_rank = {"CRITICAL": 0, "HIGH": 1, "WARNING": 2, "INFO": 3}
    alert = prediction.get("alert") or {}
    return (
        sev_rank.get(alert.get("severity", "INFO"), 3),
        0 if prediction.get("is_anomalous") else 1,
    )


def chunk_by_token_budget(predictions, max_tokens=DEFAULT_MAX_DIGEST_TOKENS,
                           hard_cap_resources=800):
    """Group predictions into chunks whose digest text fits max_tokens.

    Prefers not to split a category across chunks when avoidable (keeps
    a chunk's context coherent for the map step), but will split within a
    category if that category alone exceeds the budget.
    """
    predictions = sorted(predictions, key=_sort_key)[:hard_cap_resources]
    by_category = {}
    for p in predictions:
        by_category.setdefault(p.get("category", "Unknown"), []).append(p)

    chunks = []
    current, current_tokens = [], 0
    for category, cat_preds in by_category.items():
        cat_digests = [(p, build_resource_digest(p)) for p in cat_preds]
        cat_tokens = sum(estimate_tokens(d) for _, d in cat_digests)

        if current and current_tokens + cat_tokens > max_tokens:
            chunks.append(current)
            current, current_tokens = [], 0

        if cat_tokens > max_tokens:
            # a single category alone blows the budget — split it directly
            for p, d in cat_digests:
                t = estimate_tokens(d)
                if current and current_tokens + t > max_tokens:
                    chunks.append(current)
                    current, current_tokens = [], 0
                current.append((p, d))
                current_tokens += t
        else:
            current.extend(cat_digests)
            current_tokens += cat_tokens

    if current:
        chunks.append(current)
    return chunks  # list[list[(prediction, digest_str)]]


# ---------------------------------------------------------------------------
# 3. Prompting
# ---------------------------------------------------------------------------

MAP_SYSTEM_PROMPT = """\
You are a senior cloud infrastructure SRE writing a structured health report for an engineering team.
You will receive metric prediction digests for resources in ONE project.
Each digest tells you: the resource name, type, provider/region, context (what it is for / which team owns it),
the METRIC being forecast (with a plain-English description of what that metric measures and its unit),
the current measured value and its severity (HEALTHY / WARNING / CRITICAL), forecast values for
24 hours, 7 days, and 30 days ahead, the trend direction, warning and critical thresholds, and any
operational signals such as OOM probability, burst credit exhaustion, DDoS risk, or DB connection counts.

Your job is to turn these facts into a complete, structured engineering report. Follow these rules:
1. Begin with a "project_overview": 2-3 sentences explaining what kind of project this appears to be
   based on the resources, provider, region, and tags. (e.g. "This is a GCP production project running
   several Storage buckets and Compute VMs in asia-south1, owned by the platform-infra team...")
2. For EACH resource, write a "resource_insights" entry that covers:
   a. What the resource IS and what it does (use context/tags/type fields)
   b. What metric is being monitored and what that metric measures (explain it plainly)
   c. The current value in plain English — is it healthy? How close to warning/critical thresholds?
   d. What the forecast says for 24h / 7d / 30d and what that means operationally
   e. Whether the trend is concerning and why (reference the growth rate if present)
   f. Any specific operational risks (OOM, burst exhaustion, DDoS, connection saturation, etc.)
3. "narrative": 3-4 sentences — executive summary of the entire project's health status.
4. "priority_alerts": list of urgent items needing attention NOW — be specific (name the resource,
   the metric, the current value, and the threshold it is approaching or has already breached).
5. "cross_category_correlations": observations about how resources are behaving relative to each other
   (e.g. CPU + memory both rising on the same VMs tagged production = scale-up needed; a storage
   bucket filling while DB connections rise = application writing more data than expected).
6. "recommended_action_order": ordered list of actions. Each action MUST say: what to do, on which
   resource (by name), why (cite the specific metric value and threshold), and when (immediately /
   within 24 hours / within 7 days).
7. "risk_level": the overall project risk — exactly one of LOW / MEDIUM / HIGH / CRITICAL.

CRITICAL RULES:
- Always quote the actual numbers from the digest — never say "CPU is high" without the value.
- When a metric is HEALTHY, still explain what the metric is and confirm it is within safe range.
- When a metric unit is bytes/s or IOPS, convert to KB/MB/GB so humans can understand the scale.
- Do NOT use vague language like "monitor closely" — say specifically WHAT to monitor and WHEN to act.
- Do NOT invent metrics or values that are not in the digest.

Respond ONLY with valid JSON (no markdown fences, no text before or after):
{
  "project_overview": "string",
  "narrative": "string",
  "risk_level": "LOW|MEDIUM|HIGH|CRITICAL",
  "resource_insights": [
    {
      "resource_name": "string",
      "category": "string",
      "what_this_resource_is": "string — explain the resource type and its purpose",
      "metric_being_monitored": "string — metric name and what it measures",
      "current_health": "string — current value, severity, and what it means",
      "forecast_summary": "string — 24h/7d/30d predictions and operational implications",
      "trend_assessment": "string — direction, rate, and whether it is concerning",
      "operational_risks": "string or null — specific risks like OOM, burst exhaustion, DDoS, etc."
    }
  ],
  "priority_alerts": ["string"],
  "cross_category_correlations": ["string"],
  "recommended_action_order": ["string"]
}
"""

REDUCE_SYSTEM_PROMPT = """\
You are a senior cloud infrastructure SRE. You will be given several partial JSON health reports,
each covering a subset of resources in ONE project (split because the full set didn't fit in one call).
Merge them into ONE final report, deduplicating overlapping points and re-deriving the overall
risk_level from the combined picture.

Combine all "resource_insights" arrays from the partial summaries (dedup by resource_name if a
resource appears in multiple partials). Merge "priority_alerts" and "recommended_action_order"
lists, removing exact duplicates and re-ordering by severity. Combine "cross_category_correlations"
lists. Write a unified "project_overview" and "narrative" that covers the full project.

Respond ONLY with valid JSON (no markdown fences):
{
  "project_overview": "string",
  "narrative": "string",
  "risk_level": "LOW|MEDIUM|HIGH|CRITICAL",
  "resource_insights": [ { same shape as MAP output } ],
  "priority_alerts": ["string"],
  "cross_category_correlations": ["string"],
  "recommended_action_order": ["string"]
}
"""


def build_map_prompt(project, chunk):
    # Derive helpful context from the predictions in this chunk
    predictions = [p for p, _ in chunk]
    providers   = sorted({p.get("provider", "") for p in predictions if p.get("provider")})
    categories  = sorted({p.get("category",  "") for p in predictions if p.get("category")})
    locations   = sorted({p.get("location",  "") for p in predictions if p.get("location")})[:3]

    header = (
        f"Project ID   : {project}\n"
        f"Providers    : {', '.join(providers) or 'unknown'}\n"
        f"Categories   : {', '.join(categories) or 'unknown'}\n"
        f"Regions      : {', '.join(locations) or 'unknown'}\n"
        f"Resources in this batch: {len(chunk)}\n"
    )
    digests = "\n\n".join(d for _, d in chunk)
    return header + "\n--- Resource Digests ---\n\n" + digests


def build_reduce_prompt(project, partial_summaries):
    return (
        f"Project: {project}\n\n"
        f"Partial summaries to merge:\n"
        f"{json.dumps(partial_summaries, indent=2)}"
    )


# ---------------------------------------------------------------------------
# 4. LLM calls — provider abstraction
# ---------------------------------------------------------------------------
# Every provider plugs in as a pair of functions: _get_client_<name>(timeout)
# and _call_<name>(client, system_prompt, user_prompt, model, max_tokens).
# Adding a new provider later means adding one entry to _PROVIDER_IMPLS, not
# touching any calling code (summarize_project, explain_resource, etc. only
# ever see the (provider_name, client) tuple returned by _get_client).

# Per-request timeout (seconds) for the LLM call. A slow/hung local model or
# an unreachable Ollama host must not block the calling thread indefinitely
# -- chained across many sequential map/reduce calls in a single cycle,
# that's how a whole summary cycle goes silent with no error and no output.
# Override with LLM_REQUEST_TIMEOUT_S.
DEFAULT_REQUEST_TIMEOUT_S = float(os.environ.get("LLM_REQUEST_TIMEOUT_S", "180"))

# Retries for transient failures (connection refused while Ollama is still
# starting, a momentary timeout, etc.). Kept small -- this delays an entire
# map/reduce chain, not a single call.
DEFAULT_MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", "2"))
_RETRY_BACKOFF_S = 2.0


def _get_client_ollama(timeout):
    import requests
    session = requests.Session()
    return {"session": session, "host": OLLAMA_HOST, "timeout": timeout}


def _get_client_anthropic(timeout):
    import anthropic
    return anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY"), timeout=timeout, max_retries=1,
    )


def _call_ollama(client, system_prompt, user_prompt, model, max_tokens):
    import requests
    url = f"{client['host']}/api/chat"
    body = {
        "model": model,
        "stream": False,
        # Qwen3 is a hybrid "thinking" model; turn thinking off so the
        # response is the JSON object directly, not a <think>...</think>
        # preamble followed by the JSON (Ollama >=0.9 supports this flag,
        # older servers just ignore the unknown field).
        "think": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "options": {"num_predict": max_tokens},
    }
    try:
        resp = client["session"].post(url, json=body, timeout=client["timeout"])
        resp.raise_for_status()
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(
            f"Could not reach Ollama at {client['host']}. Is it running? "
            f"Start it with `ollama serve` (and `ollama pull {model}` if you "
            f"haven't already). ({e})"
        ) from e
    except requests.exceptions.Timeout as e:
        raise RuntimeError(
            f"Ollama request timed out after {client['timeout']}s (model={model}). "
            f"Raise LLM_REQUEST_TIMEOUT_S if the model is just slow on this hardware."
        ) from e
    data = resp.json()
    return (data.get("message") or {}).get("content") or ""


def _call_anthropic(client, system_prompt, user_prompt, model, max_tokens):
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return "".join(
        block.text for block in resp.content if getattr(block, "type", None) == "text"
    )


_PROVIDER_IMPLS = {
    "ollama":     {"get_client": _get_client_ollama,     "call": _call_ollama},
    "anthropic":  {"get_client": _get_client_anthropic,  "call": _call_anthropic},
}


def _get_client(provider=None, timeout=DEFAULT_REQUEST_TIMEOUT_S):
    """Returns (provider_name, client). Ollama is the default and needs no
    API key -- just a running `ollama serve` and the model pulled locally.
    Anthropic remains available as an opt-in (LLM_PROVIDER=anthropic) for
    anyone who wants a cloud model instead.
    """
    provider = (provider or DEFAULT_PROVIDER).lower()
    impl = _PROVIDER_IMPLS.get(provider)
    if not impl:
        raise ValueError(
            f"Unknown LLM_PROVIDER {provider!r}; expected one of {sorted(_PROVIDER_IMPLS)}."
        )
    return (provider, impl["get_client"](timeout))


def _strip_think_tags(text):
    # Defense in depth: even with think=False, some Ollama/Qwen3 builds may
    # still emit a <think>...</think> block. Strip it so JSON parsing below
    # sees only the actual answer.
    if "<think>" in text and "</think>" in text:
        before, _, after = text.partition("<think>")
        _, _, after = after.partition("</think>")
        text = before + after
    return text


def _call_llm(client, system_prompt, user_prompt, model=DEFAULT_MODEL, max_tokens=1000):
    provider, c = client
    impl = _PROVIDER_IMPLS.get(provider)
    if not impl:
        raise ValueError(f"Unknown LLM provider {provider!r}.")

    last_err = None
    for attempt in range(DEFAULT_MAX_RETRIES + 1):
        try:
            text = impl["call"](c, system_prompt, user_prompt, model, max_tokens)
            break
        except Exception as e:
            last_err = e
            if attempt < DEFAULT_MAX_RETRIES:
                time.sleep(_RETRY_BACKOFF_S * (attempt + 1))
            else:
                raise
    else:
        raise last_err  # pragma: no cover — loop always breaks or raises

    text = _strip_think_tags(text).strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[4:] if text.lower().startswith("json") else text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Never let a malformed LLM response crash the batch job — degrade
        # to a raw-text summary rather than losing the whole project.
        return {
            "narrative": text,
            "risk_level": "UNKNOWN",
            "priority_alerts": [],
            "cross_category_correlations": [],
            "recommended_action_order": [],
            "_parse_error": True,
        }


# ---------------------------------------------------------------------------
# 5. Orchestration: one project -> one final summary (map-reduce as needed)
# ---------------------------------------------------------------------------

def summarize_project(client, project, predictions, model=DEFAULT_MODEL,
                       max_tokens=DEFAULT_MAX_DIGEST_TOKENS):
    chunks = chunk_by_token_budget(predictions, max_tokens=max_tokens)

    if not chunks:
        return None

    # Use a higher token limit for the LLM output to accommodate the richer
    # structured response (resource_insights array can be long for large projects)
    llm_output_tokens = max(2500, max_tokens // 2)

    if len(chunks) == 1:
        summary = _call_llm(
            client, MAP_SYSTEM_PROMPT,
            build_map_prompt(project, chunks[0]), model=model,
            max_tokens=llm_output_tokens,
        )
        summary["_map_reduce"] = False
    else:
        partials = [
            _call_llm(client, MAP_SYSTEM_PROMPT, build_map_prompt(project, c),
                      model=model, max_tokens=llm_output_tokens)
            for c in chunks
        ]
        summary = _call_llm(
            client, REDUCE_SYSTEM_PROMPT,
            build_reduce_prompt(project, partials), model=model,
            max_tokens=llm_output_tokens,
        )
        summary["_map_reduce"] = True
        summary["_chunk_count"] = len(chunks)

    # Ensure required keys exist even if LLM omitted them
    summary.setdefault("project_overview", "")
    summary.setdefault("resource_insights", [])
    summary.setdefault("cross_category_correlations", [])

    summary["project"] = project
    summary["resource_count"] = len(predictions)
    summary["categories"] = sorted({p.get("category", "Unknown") for p in predictions})
    # Deterministic roster built directly from predictions (not from LLM output)
    # so it's always complete regardless of chunking / token budget.
    summary["resources"] = build_resource_roster(predictions)
    summary["generated_at"] = datetime.now(timezone.utc).isoformat()
    summary["model"] = model
    return summary


def summarize_all_projects(predictions, model=DEFAULT_MODEL,
                            max_tokens=DEFAULT_MAX_DIGEST_TOKENS, client=None,
                            provider=None, on_project_done=None):
    """predictions: flat list pulled from Mongo across all category
    collections for the window. Returns list of project summary docs.

    on_project_done, if given, is called with (project, summary) right
    after each project finishes (success or per-project error) -- BEFORE
    moving on to the next project. This lets the caller persist results
    incrementally instead of waiting for every project in the window to
    finish before anything is written. Without it, one slow/rate-limited
    project anywhere in the list holds back every summary that already
    finished, and nothing shows up downstream until the whole batch is
    done -- which, with many projects and free-tier rate limits, can be a
    very long time.
    """
    client = client or _get_client(provider)
    by_project = group_predictions_by_project(predictions)

    summaries = []
    for project, project_preds in by_project.items():
        try:
            summary = summarize_project(client, project, project_preds, model=model, max_tokens=max_tokens)
        except Exception as e:
            summary = {
                "project": project,
                "resource_count": len(project_preds),
                "narrative": f"LLM summarization failed: {e}",
                "risk_level": "UNKNOWN",
                "priority_alerts": [],
                "cross_category_correlations": [],
                "recommended_action_order": [],
                "resources": build_resource_roster(project_preds),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "model": model,
                "_error": True,
            }
        if summary:
            summaries.append(summary)
            if on_project_done:
                on_project_done(project, summary)
    return summaries
