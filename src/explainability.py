"""explainability.py — Plain-English explanation for every prediction/alert.

Enterprises don't act on a bare risk_score. This module turns the numeric
signals already computed by anomaly.py + profiler.py + prediction.py into a
short, human-readable narrative:

    "CPU has been trending up ~2.1%/hour for the last 48h. Residual z-score
     is 3.4 (unusual for this resource at this time of day). No burst spike
     detected. Profile confidence: 0.86 (24 days of history)."

No new ML — purely a templating layer over fields that already exist on the
prediction dict (anomaly_detail, behavioral_forecast, profile_info, payload).
Designed to be called once per prediction, after generate_predictions() has
built the rest of the document, and the output attached as `explanation`.
"""
from __future__ import annotations

from typing import Dict, List


def _fmt_pct(v, decimals=1):
    if v is None:
        return None
    try:
        return f"{float(v):.{decimals}f}%"
    except (TypeError, ValueError):
        return None


def _driver_sentences(anomaly_detail: Dict) -> List[str]:
    """Translate anomaly.py's internal signal scores into plain English."""
    sentences = []
    if not anomaly_detail or anomaly_detail.get("error"):
        return sentences

    z   = anomaly_detail.get("residual_zscore_signal", 0.0)
    br  = anomaly_detail.get("burst_ratio_signal", 0.0)
    iff = anomaly_detail.get("isolation_forest_signal", 0.0)

    if z >= 0.7:
        sentences.append("Current value is far outside this resource's own historical pattern for this time of day/week.")
    elif z >= 0.4:
        sentences.append("Current value is moderately unusual compared to this resource's own baseline.")

    if br >= 0.6:
        sentences.append("A sharp short-term spike was detected within the reporting interval (max/avg ratio elevated).")
    elif br >= 0.4:
        sentences.append("Some intra-hour burstiness observed, below alert threshold.")

    if iff >= 0.55:
        sentences.append("Multi-signal pattern (isolation forest) flags this as an outlier across the resource's recent behavior.")

    if not sentences:
        sentences.append("No individual signal is dominant; composite score reflects a blend of small deviations.")

    return sentences


def _trend_sentence(forecast: Dict) -> str | None:
    if not forecast:
        return None
    trend = forecast.get("trend_direction")
    slope = forecast.get("hourly_slope")
    feature = forecast.get("canonical_feature", "the primary metric")
    reliability = forecast.get("forecast_reliability", "low")

    if trend == "increasing" and slope:
        per_hr = abs(slope)
        return (f"{feature.replace('canonical_', '').replace('_', ' ')} has been trending up "
                f"~{per_hr:.2f}/hour (forecast reliability: {reliability}).")
    if trend == "decreasing" and slope:
        per_hr = abs(slope)
        return (f"{feature.replace('canonical_', '').replace('_', ' ')} has been trending down "
                f"~{per_hr:.2f}/hour (forecast reliability: {reliability}).")
    if trend == "stable":
        return "No significant trend — value has been stable relative to its own baseline."
    return None


def _profile_sentence(profile_info: Dict) -> str | None:
    if not profile_info:
        return None
    conf = profile_info.get("profile_confidence", 0.0)
    age = profile_info.get("profile_age_days", 0)
    regime = profile_info.get("regime_changed", False)

    base = f"Profile confidence: {conf:.2f} ({age}d of pooled history)."
    if regime:
        base += " NOTE: a sustained shift from this resource type's pooled historical baseline was detected recently — thresholds may need to re-settle."
    return base


def _top_contributors(payload: Dict, category: str, max_items: int = 3) -> List[str]:
    """Pull the 2-3 most relevant raw metric values into the explanation,
    so the alert isn't just a score — it shows the actual numbers.
    """
    out = []
    if not payload:
        return out
    metrics = payload.get("metrics") or payload.get("signals") or {}
    if isinstance(metrics, dict):
        # Prefer canonical_ keys with non-null values
        items = [(k, v) for k, v in metrics.items()
                 if k.startswith("canonical_") and v is not None]
        for k, v in items[:max_items]:
            label = k.replace("canonical_", "").replace("_", " ")
            try:
                out.append(f"{label}={float(v):.1f}")
            except (TypeError, ValueError):
                out.append(f"{label}={v}")
    return out


def build_explanation(prediction: Dict) -> Dict:
    """Build a structured + plain-text explanation for one prediction document.

    Returns:
        {
          "narrative": "<2-4 sentence plain English summary>",
          "drivers": [...],            # bullet list, same content as narrative
          "top_metrics": [...],        # raw values that informed the decision
          "confidence_note": str,
        }
    """
    anomaly_detail = prediction.get("anomaly_detail", {}) or {}
    forecast       = prediction.get("behavioral_forecast", {}) or {}
    profile_info   = prediction.get("profile_info", {}) or {}
    payload        = prediction.get("payload", {}) or {}
    category       = prediction.get("category", "")
    risk_score     = prediction.get("dashboard", {}).get("display_score")

    sentences = []

    if risk_score is not None:
        sentences.append(f"Risk score {risk_score}/100 for this {category} resource.")

    sentences.extend(_driver_sentences(anomaly_detail))

    trend_s = _trend_sentence(forecast)
    if trend_s:
        sentences.append(trend_s)

    profile_s = _profile_sentence(profile_info)
    if profile_s:
        sentences.append(profile_s)

    top_metrics = _top_contributors(payload, category)

    return {
        "narrative":       " ".join(sentences),
        "drivers":         _driver_sentences(anomaly_detail),
        "top_metrics":     top_metrics,
        "confidence_note": profile_s,
    }


def attach_explanations(predictions: List[Dict]) -> List[Dict]:
    """Mutate predictions in-place, adding an `explanation` field to each.
    Safe to call unconditionally — never raises on malformed input.
    """
    for p in predictions:
        try:
            p["explanation"] = build_explanation(p)
        except Exception as e:
            p["explanation"] = {"narrative": "Explanation unavailable.", "error": str(e)}
    return predictions
