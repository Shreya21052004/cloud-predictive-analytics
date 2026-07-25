"""ui/backend.py — read-only FastAPI backend for the Cloud Analytics Platform.

Reads from MongoDB `optimised_kafkav2` (default). Serves index.html at /.
Does NOT write anything.

Collections read:
  22prediction_logs_compute / storage / network / container / databases
  22llm_project_summaries
  22llm_resource_explanations  (may be empty — fallback to prediction docs)

Start:
    uvicorn ui.backend:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from scripts.project_dashboard import get_project_predictions, get_project_latest_summary
from scripts.resource_dashboard import find_resources_by_tag
from src.mongo_io import _bson_safe, get_collection

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
DB_NAME   = os.environ.get("MONGO_DB",  "optimised_kafkav2")


def _risk_level_from_score(score) -> str:
    """Map a numeric display_score (0–100) to a canonical risk level string.

    Returns HEALTHY for score=0 (resource is idle/healthy, not unknown).
    Returns UNKNOWN only when the score itself is absent (None).
    """
    if score is None:
        return "UNKNOWN"
    if score >= 90:
        return "CRITICAL"
    if score >= 75:
        return "HIGH"
    if score >= 50:
        return "MEDIUM"
    if score > 0:
        return "LOW"
    return "HEALTHY"

CATEGORY_COLLECTIONS = {
    "Compute":   "22prediction_logs_compute",
    "Storage":   "22prediction_logs_storage",
    "Network":   "22prediction_logs_network",
    "Container": "22prediction_logs_container",
    "Databases": "22prediction_logs_databases",
}
LLM_SUMMARY_COLLECTION       = "22llm_project_summaries"
RESOURCE_EXPLANATION_COLLECTION = "22llm_resource_explanations"

# Fields to suppress from API responses (raw payload is large and not useful in UI)
_STRIP_FIELDS = {"payload", "_idempotent_key"}

# Sections produced by the improved LLM explanation pipeline
_LLM_EXP_SECTIONS = (
    "resource_overview", "current_state", "trend_analysis", "forecast",
    "evidence", "operational_impact", "recommended_actions", "confidence",
)

app = FastAPI(title="Cloud Analytics Platform API", version="2.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

static_dir = Path(__file__).resolve().parent
NO_CACHE = {"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0", "Pragma": "no-cache"}


# ── Helpers ────────────────────────────────────────────────────────────────

def _strip(doc: dict) -> dict:
    return {k: v for k, v in doc.items() if k not in _STRIP_FIELDS}


def _latest_per_resource(predictions: list) -> list:
    """Keep only the most-recent doc per (resource_id, category)."""
    seen: set = set()
    out: list = []
    for p in predictions:
        key = (p.get("resource_id"), p.get("category"))
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def _to_card(p: dict) -> dict:
    """Flatten a prediction doc into a UI-ready resource card dict.

    Includes explanation.narrative so the frontend can display it
    without a second round-trip.
    """
    f    = p.get("behavioral_forecast") or {}
    dash = p.get("dashboard") or {}
    exp  = p.get("explanation") or {}
    return {
        "resource_id":           p.get("resource_id"),
        "resource_name":         p.get("resource_name"),
        "category":              p.get("category"),
        "provider":              p.get("provider"),
        "location":              p.get("location"),
        "canonical_resource_type": p.get("canonical_resource_type"),
        "service_family":        p.get("service_family"),
        "tags":                  p.get("tags") or [],
        "project":               p.get("project"),
        "summary_details":       p.get("summary_details"),

        # Anomaly
        "is_anomalous":          p.get("is_anomalous", False),
        "anomaly_score":         p.get("anomaly_score"),

        # Risk / alert
        "risk_score":            dash.get("display_score"),
        "risk_level":            _risk_level_from_score(dash.get("display_score")),
        "risk_label":            dash.get("display_label"),
        "dashboard_trend":       dash.get("trend"),
        "dashboard_summary":     dash.get("summary_text"),
        "alert_severity":        (p.get("alert") or {}).get("severity", "INFO"),
        "alert_trigger":         (p.get("alert") or {}).get("trigger", False),

        # Forecast
        "metric_name":           f.get("metric_name"),
        "current_value":         f.get("current_value"),
        "trend_direction":       f.get("trend_direction"),
        "forecast_reliability":  f.get("forecast_reliability"),
        "forecast_method":       f.get("forecast_method"),
        "forecast_1h":           f.get("forecast_1h"),
        "forecast_6h":           f.get("forecast_6h"),
        "forecast_24h":          f.get("forecast_24h"),
        "forecast_7d":           f.get("forecast_7d"),
        "forecast_30d":          f.get("forecast_30d"),
        "forecast_uncertainty_pct": f.get("forecast_uncertainty_pct"),
        "prediction_confidence_score": f.get("prediction_confidence_score"),

        # AI explanation (inline)
        "explanation_narrative":  exp.get("narrative") or "",
        "explanation_confidence": exp.get("confidence_note") or "",
        "explanation_drivers":    exp.get("drivers") or [],

        # Recommendations
        "recommendations":        p.get("recommendations") or [],
        "failure_predictions":    p.get("failure_predictions") or [],

        # Pipeline metadata
        "prediction_timestamp":   p.get("prediction_timestamp"),
        "data_quality_note":      p.get("data_quality_note"),
        "data_completeness_score": p.get("data_completeness_score"),
    }


def _fetch_resource_from_predictions(resource_id: str) -> dict | None:
    """Search every prediction collection for the most-recent doc matching resource_id."""
    best = None
    for col_name in CATEGORY_COLLECTIONS.values():
        col = get_collection(MONGO_URI, DB_NAME, col_name)
        doc = col.find_one({"resource_id": resource_id}, sort=[("prediction_timestamp", -1)])
        if doc:
            ts = doc.get("prediction_timestamp") or ""
            if best is None or ts > (best.get("prediction_timestamp") or ""):
                best = doc
    return best


# ── Static ─────────────────────────────────────────────────────────────────

@app.get("/")
def serve_index():
    return FileResponse(str(static_dir / "index.html"), headers=NO_CACHE)


# ── Projects ───────────────────────────────────────────────────────────────

@app.get("/api/projects")
def list_projects():
    projects: set = set()
    for col_name in CATEGORY_COLLECTIONS.values():
        col = get_collection(MONGO_URI, DB_NAME, col_name)
        projects.update(col.distinct("project"))
    return _bson_safe({"projects": sorted(p for p in projects if p)})


@app.get("/api/project/{project_id}")
def project_dashboard(project_id: str):
    predictions = get_project_predictions(MONGO_URI, DB_NAME, project_id, limit=2000)
    summary     = get_project_latest_summary(MONGO_URI, DB_NAME, project_id)

    # Discard error-flagged summaries (written when the LLM timed out during
    # a pipeline run) so the UI never surfaces "LLM summarization failed…" as content.
    if summary and summary.get("_error"):
        summary = None

    if not predictions and not summary:
        raise HTTPException(404, f"No data for project '{project_id}'.")

    latest = _latest_per_resource(predictions)

    # Enrich each resource card with its LLM explanation if available
    exp_col = get_collection(MONGO_URI, DB_NAME, RESOURCE_EXPLANATION_COLLECTION)
    exp_by_rid = {}
    rids = [p.get("resource_id") for p in latest if p.get("resource_id")]
    if rids:
        for exp_doc in exp_col.find({"resource_id": {"$in": rids}, "_error": {"$ne": True}}, {"_id": 0}):
            exp_by_rid[exp_doc.get("resource_id")] = {
                sec: exp_doc.get(sec, "") for sec in _LLM_EXP_SECTIONS
            } | {
                "risk_level":    exp_doc.get("risk_level", ""),
                "computed_stats": exp_doc.get("computed_stats") or {},
                "generated_at":  exp_doc.get("generated_at"),
                "model":         exp_doc.get("model"),
                "metric_name":   exp_doc.get("metric_name"),
            }

    cards = []
    for p in latest:
        card = _to_card(p)
        rid  = p.get("resource_id")
        if rid and rid in exp_by_rid:
            card["llm_explanation"] = exp_by_rid[rid]
        cards.append(card)

    return _bson_safe({
        "project":        project_id,
        "summary":        summary,
        "resources":      cards,
        "resource_count": len(cards),
    })


@app.get("/api/resources/all")
def all_resources(
    limit:  int = Query(200, le=1000),
    offset: int = Query(0, ge=0),
):
    """Return all latest-per-resource docs across every category collection."""
    all_docs: list = []
    for col_name in CATEGORY_COLLECTIONS.values():
        col  = get_collection(MONGO_URI, DB_NAME, col_name)
        docs = list(col.find({}, {
            "resource_id": 1, "resource_name": 1, "category": 1,
            "provider": 1, "location": 1, "project": 1,
            "is_anomalous": 1, "dashboard": 1, "alert": 1,
            "behavioral_forecast": 1, "prediction_timestamp": 1,
            "recommendations": 1, "data_quality_note": 1,
        }).sort("prediction_timestamp", -1).limit(5000))
        all_docs.extend(docs)

    latest = _latest_per_resource(all_docs)
    latest.sort(key=lambda p: (p.get("dashboard") or {}).get("display_score") or 0, reverse=True)
    total = len(latest)
    page  = latest[offset: offset + limit]
    return _bson_safe({"total": total, "offset": offset, "limit": limit,
                       "resources": [_to_card(p) for p in page]})


# ── Platform stats ─────────────────────────────────────────────────────────

@app.get("/api/stats")
def global_stats():
    all_docs: list = []
    for col_name in CATEGORY_COLLECTIONS.values():
        col  = get_collection(MONGO_URI, DB_NAME, col_name)
        docs = list(col.find({}, {
            "resource_id": 1, "category": 1, "provider": 1, "project": 1,
            "is_anomalous": 1, "anomaly_score": 1, "dashboard": 1,
            "behavioral_forecast": 1, "alert": 1, "prediction_timestamp": 1,
        }).sort("prediction_timestamp", -1).limit(5000))
        all_docs.extend(docs)

    latest = _latest_per_resource(all_docs)

    providers:  dict = {}
    categories: dict = {}
    projects:   set  = set()
    anomalous       = 0
    alert_active    = 0
    risk_dist = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "HEALTHY": 0, "UNKNOWN": 0}

    for p in latest:
        prov = p.get("provider") or "Unknown"
        cat  = p.get("category") or "Unknown"
        proj = p.get("project") or ""
        providers[prov]   = providers.get(prov, 0) + 1
        categories[cat]   = categories.get(cat, 0) + 1
        if proj: projects.add(proj)
        if p.get("is_anomalous"): anomalous += 1
        if (p.get("alert") or {}).get("trigger"): alert_active += 1
        score = (p.get("dashboard") or {}).get("display_score")
        rl = _risk_level_from_score(score)
        risk_dist[rl] = risk_dist.get(rl, 0) + 1

    summary_col = get_collection(MONGO_URI, DB_NAME, LLM_SUMMARY_COLLECTION)
    return _bson_safe({
        "total_resources":   len(latest),
        "total_anomalous":   anomalous,
        "total_alerts":      alert_active,
        "total_projects":    len(projects),
        "providers":         providers,
        "categories":        categories,
        "risk_distribution": risk_dist,
        "llm_summaries":     summary_col.count_documents({}),
    })


# ── Resource search / filter ───────────────────────────────────────────────

@app.get("/api/resources/search")
def search_resources(
    project:       Optional[str] = Query(None),
    provider:      Optional[str] = Query(None),
    category:      Optional[str] = Query(None),
    resource_id:   Optional[str] = Query(None),
    metric_name:   Optional[str] = Query(None),
    service_name:  Optional[str] = Query(None),
    tag_key:       Optional[str] = Query(None),
    tag_value:     Optional[str] = Query(None),
    anomalous_only: bool         = Query(False),
    alert_only:     bool         = Query(False),
    risk_min:      Optional[int] = Query(None, ge=0),
    q:             Optional[str] = Query(None),   # free-text across id/name/narrative
    limit:         int           = Query(200, le=1000),
    offset:        int           = Query(0, ge=0),
):
    """Multi-filter resource search. Returns deduplicated latest-per-resource results."""
    base_query: dict = {}
    if project:        base_query["project"]    = project
    if anomalous_only: base_query["is_anomalous"] = True
    if alert_only:     base_query["alert.trigger"] = True
    if provider:       base_query["provider"]   = {"$regex": provider,  "$options": "i"}
    if resource_id:    base_query["resource_id"] = {"$regex": resource_id, "$options": "i"}

    if tag_key and tag_value:
        base_query["$or"] = [
            {f"tags.{tag_key}": tag_value},
            {"tags": {"$elemMatch": {"key": tag_key, "value": tag_value}}},
        ]
    elif tag_key:
        base_query["$or"] = [
            {f"tags.{tag_key}": {"$exists": True}},
            {"tags": {"$elemMatch": {"key": tag_key}}},
        ]

    target_cols = (
        [CATEGORY_COLLECTIONS[category]]
        if category and category in CATEGORY_COLLECTIONS
        else list(CATEGORY_COLLECTIONS.values())
    )

    all_docs: list = []
    for col_name in target_cols:
        col       = get_collection(MONGO_URI, DB_NAME, col_name)
        col_query = dict(base_query)
        if metric_name:  col_query["behavioral_forecast.metric_name"]  = {"$regex": metric_name,  "$options": "i"}
        if service_name: col_query["service_family"]                   = {"$regex": service_name, "$options": "i"}
        if risk_min is not None: col_query["dashboard.display_score"]  = {"$gte": risk_min}
        # Free-text: match against resource_id, explanation narrative, or summary_details
        if q:
            col_query["$and"] = col_query.get("$and", []) + [{
                "$or": [
                    {"resource_id":               {"$regex": q, "$options": "i"}},
                    {"resource_name":             {"$regex": q, "$options": "i"}},
                    {"explanation.narrative":     {"$regex": q, "$options": "i"}},
                    {"summary_details":           {"$regex": q, "$options": "i"}},
                ]
            }]
        docs = list(col.find(col_query).sort("prediction_timestamp", -1).limit(limit * 3))
        all_docs.extend(docs)

    latest = _latest_per_resource(all_docs)
    latest.sort(key=lambda p: (p.get("dashboard") or {}).get("display_score") or 0, reverse=True)

    total  = len(latest)
    page   = latest[offset: offset + limit]
    return _bson_safe({
        "total":     total,
        "offset":    offset,
        "limit":     limit,
        "resources": [_to_card(p) for p in page],
    })


# ── Resource detail ─────────────────────────────────────────────────────────

def _fetch_llm_explanation(resource_id: str) -> dict | None:
    """Return the structured 8-section LLM explanation for a resource, or None.

    Error-flagged docs (_error: True, written when the LLM timed out or returned
    invalid JSON) are excluded — the UI should fall back to scoreToRisk() rather
    than showing 'UNKNOWN' from a failed generation attempt.
    """
    exp_col = get_collection(MONGO_URI, DB_NAME, RESOURCE_EXPLANATION_COLLECTION)
    doc = exp_col.find_one({"resource_id": resource_id, "_error": {"$ne": True}})
    if not doc:
        return None
    doc.pop("_id", None)
    # Return only the sections that are useful for the UI — no raw metrics blob.
    return {
        **{sec: doc.get(sec, "") for sec in _LLM_EXP_SECTIONS},
        "risk_level":    doc.get("risk_level", ""),
        "computed_stats": doc.get("computed_stats") or {},
        "generated_at":  doc.get("generated_at"),
        "model":         doc.get("model"),
    }


@app.get("/api/resource/detail")
def resource_detail(resource_id: str = Query(...)):
    """Full prediction doc for one resource.

    Uses a query-parameter for resource_id so that Azure/GCP resource IDs
    containing slashes are passed safely (e.g. ?resource_id=/subscriptions/...).
    """
    doc = _fetch_resource_from_predictions(resource_id)

    if doc:
        doc.pop("_id", None)
        clean = _strip(doc)
        llm_exp = _fetch_llm_explanation(resource_id)
        if llm_exp:
            clean["llm_explanation"] = llm_exp
        return _bson_safe(clean)

    llm_exp = _fetch_llm_explanation(resource_id)
    if llm_exp:
        return _bson_safe(llm_exp)

    raise HTTPException(404, f"Resource '{resource_id}' not found.")


@app.get("/api/resource/history")
def resource_history(resource_id: str = Query(...), limit: int = Query(50, le=200)):
    """Time-series of predictions for one resource (for sparkline/trend data)."""
    results: list = []
    for col_name in CATEGORY_COLLECTIONS.values():
        col  = get_collection(MONGO_URI, DB_NAME, col_name)
        docs = list(col.find(
            {"resource_id": resource_id},
            {
                "_id": 0, "payload": 0, "_idempotent_key": 0,
                "summary_details": 0, "anomaly_detail": 0,
            }
        ).sort("prediction_timestamp", -1).limit(limit))
        results.extend(docs)
    results.sort(key=lambda p: p.get("prediction_timestamp", ""), reverse=True)
    return _bson_safe({"resource_id": resource_id, "history": results[:limit]})


# ── LLM summaries ──────────────────────────────────────────────────────────

@app.get("/api/summaries")
def list_summaries(limit: int = Query(50, le=200)):
    col  = get_collection(MONGO_URI, DB_NAME, LLM_SUMMARY_COLLECTION)
    # Exclude error-flagged summaries (LLM timed out during pipeline run)
    docs = list(col.find(
        {"_error": {"$ne": True}},
        {"_id": 0, "payload": 0}
    ).sort("generated_at", -1).limit(limit))
    return _bson_safe({"summaries": docs, "count": len(docs)})


# ── Tag search ─────────────────────────────────────────────────────────────

@app.get("/api/resources/by-tag")
def resources_by_tag(tag_key: str, tag_value: Optional[str] = None):
    docs = find_resources_by_tag(MONGO_URI, DB_NAME, tag_key, tag_value)
    return _bson_safe({"tag_key": tag_key, "tag_value": tag_value, "count": len(docs), "resources": docs})


# ── Unified search ─────────────────────────────────────────────────────────

@app.get("/api/search")
def unified_search(q: str = Query(..., min_length=1)):
    """Search across projects (by id) and resources (by id, name, narrative)."""
    q_stripped = q.strip()

    # 1. Project match
    project_hits: list = []
    sum_col = get_collection(MONGO_URI, DB_NAME, LLM_SUMMARY_COLLECTION)
    proj_docs = list(sum_col.find(
        {"$or": [
            {"project": {"$regex": q_stripped, "$options": "i"}},
            {"narrative": {"$regex": q_stripped, "$options": "i"}},
        ]},
        {"_id": 0}
    ).limit(10))
    project_hits = proj_docs

    # 2. Resource match across all prediction collections
    resource_hits: list = []
    for col_name in CATEGORY_COLLECTIONS.values():
        col  = get_collection(MONGO_URI, DB_NAME, col_name)
        docs = list(col.find(
            {"$or": [
                {"resource_id":           {"$regex": q_stripped, "$options": "i"}},
                {"resource_name":         {"$regex": q_stripped, "$options": "i"}},
                {"provider":              {"$regex": q_stripped, "$options": "i"}},
                {"category":              {"$regex": q_stripped, "$options": "i"}},
                {"service_family":        {"$regex": q_stripped, "$options": "i"}},
                {"explanation.narrative": {"$regex": q_stripped, "$options": "i"}},
            ]},
        ).sort("prediction_timestamp", -1).limit(50))
        resource_hits.extend(docs)

    resource_hits = _latest_per_resource(resource_hits)
    resource_hits.sort(
        key=lambda p: (p.get("dashboard") or {}).get("display_score") or 0, reverse=True
    )

    return _bson_safe({
        "query":          q_stripped,
        "projects":       project_hits[:10],
        "resources":      [_to_card(p) for p in resource_hits[:50]],
        "project_count":  len(project_hits),
        "resource_count": len(resource_hits),
    })
