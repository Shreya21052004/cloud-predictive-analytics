"""scripts/project_dashboard.py — read-side API backing the per-project UI.

No new collections. Every category still writes to ONE shared collection
(22prediction_logs_compute, _storage, etc. — see pipeline.py::_collection_for)
and llm_project_summary.py writes ONE shared 22llm_project_summaries
collection. Routing to "a project's view" happens here, entirely at query
time, by filtering on the `project` field that's already stamped on every
prediction and every summary doc. This module is the only place project
routing logic should live — don't duplicate it in the frontend or in
pipeline.py.

Cardinality this module enforces (per the source data model):
  - resourceId -> exactly one project
  - project -> many accounts
  - account -> exactly one project
So authorization is "does this account's project match the requested
project", not "does this account own this resource" (accounts don't own
resources directly — resources belong to the project the account is in).
"""
import sys
from pathlib import Path

# Allow running as `python scripts/project_dashboard.py` without installing
# the package — matches the convention used by scripts/backfill_project_collections.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CATEGORY_OUTPUT_COLLECTION, TARGET_CATEGORIES
from src.mongo_io import get_collection
from src.llm_summary import LLM_SUMMARY_COLLECTION

# Same collection-name convention as pipeline.py's --*-out CLI args.
DEFAULT_CATEGORY_COLLECTIONS = {
    "Compute":   "22prediction_logs_compute",
    "Storage":   "22prediction_logs_storage",
    "Network":   "22prediction_logs_network",
    "Container": "22prediction_logs_container",
    "Databases": "22prediction_logs_databases",
}


def ensure_project_indexes(mongo_uri, db_name, category_collections=None):
    """Run once (and after any collection rename). Without this, every
    dashboard load below does a full collection scan on potentially
    millions of prediction documents.
    """
    category_collections = category_collections or DEFAULT_CATEGORY_COLLECTIONS
    created = []
    for category, col_name in category_collections.items():
        col = get_collection(mongo_uri, db_name, col_name)
        col.create_index([("project", 1), ("prediction_timestamp", -1)])
        created.append(col_name)

    summary_col = get_collection(mongo_uri, db_name, LLM_SUMMARY_COLLECTION)
    summary_col.create_index([("project", 1), ("generated_at", -1)])
    created.append(LLM_SUMMARY_COLLECTION)

    return created


def account_can_access_project(mongo_uri, db_name, account_id, requested_project_id,
                                category_collections=None):
    """Resolve account_id -> its actual project from the data itself (never
    trust a project_id passed straight from the client) and compare.

    account_id -> project is 1:1 (one account belongs to a unique project),
    so any collection with a matching account_id document is a valid source
    of truth; we just need one hit.
    """
    category_collections = category_collections or DEFAULT_CATEGORY_COLLECTIONS
    for col_name in category_collections.values():
        col = get_collection(mongo_uri, db_name, col_name)
        doc = col.find_one({"account_id": account_id}, {"project": 1})
        if doc:
            return doc.get("project") == requested_project_id
    return False  # unknown account -> deny by default


def get_project_predictions(mongo_uri, db_name, project_id, category=None,
                             limit=500, category_collections=None):
    """All predictions for a project, optionally filtered to one category.
    This is the "project sees all resourceIds/predictions in that project"
    view — one query per category collection, filtered by `project`, merged.
    """
    category_collections = category_collections or DEFAULT_CATEGORY_COLLECTIONS
    categories = [category] if category else list(TARGET_CATEGORIES)

    predictions = []
    for cat in categories:
        col_name = category_collections.get(cat)
        if not col_name:
            continue
        col = get_collection(mongo_uri, db_name, col_name)
        cursor = (
            col.find({"project": project_id})
            .sort("prediction_timestamp", -1)
            .limit(limit)
        )
        predictions.extend(cursor)
    predictions.sort(key=lambda p: p.get("prediction_timestamp", ""), reverse=True)
    return predictions[:limit]


def get_project_latest_summary(mongo_uri, db_name, project_id):
    """Most recent LLM-generated narrative/recommendation for the project."""
    col = get_collection(mongo_uri, db_name, LLM_SUMMARY_COLLECTION)
    return col.find_one({"project": project_id}, sort=[("generated_at", -1)])


def get_project_dashboard(mongo_uri, db_name, account_id, project_id,
                           category=None, prediction_limit=500):
    """Single entrypoint the UI backend calls per page load.

    Enforces account->project authorization before returning anything.
    """
    if not account_can_access_project(mongo_uri, db_name, account_id, project_id):
        raise PermissionError(
            f"account_id={account_id!r} is not authorized for project={project_id!r}"
        )

    return {
        "project": project_id,
        "predictions": get_project_predictions(
            mongo_uri, db_name, project_id, category=category, limit=prediction_limit,
        ),
        "llm_summary": get_project_latest_summary(mongo_uri, db_name, project_id),
    }


if __name__ == "__main__":
    import argparse
    import json

    from src.mongo_io import _bson_safe

    p = argparse.ArgumentParser(description="Fetch or index the per-project dashboard view")
    p.add_argument("--mongo-uri", default="mongodb://localhost:27017")
    p.add_argument("--db", default="mydb")
    sub = p.add_subparsers(dest="cmd", required=True)

    idx = sub.add_parser("create-indexes")

    fetch = sub.add_parser("fetch")
    fetch.add_argument("--account-id", required=True)
    fetch.add_argument("--project-id", required=True)
    fetch.add_argument("--category", default=None)

    args = p.parse_args()

    if args.cmd == "create-indexes":
        created = ensure_project_indexes(args.mongo_uri, args.db)
        print(f"Indexed 'project' on: {', '.join(created)}")
    elif args.cmd == "fetch":
        try:
            result = get_project_dashboard(
                args.mongo_uri, args.db, args.account_id, args.project_id, category=args.category,
            )
            print(json.dumps(_bson_safe(result), indent=2, default=str))
        except PermissionError as e:
            print(f"DENIED: {e}")
            raise SystemExit(1)
