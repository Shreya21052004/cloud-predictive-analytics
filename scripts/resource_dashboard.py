"""scripts/resource_dashboard.py — read-side query module for per-resource
LLM explanations, and a CLI for setting/listing the manual thresholds that
feed them (src/thresholds.py).

Two query shapes, both backed entirely by
22llm_resource_explanations (src/llm_resource_explain.py):

    get_resource_explanation(uri, db, resource_id)   -> one resource, exact id
    find_resources_by_tag(uri, db, key, value)        -> many resources, by tag
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mongo_io import get_collection
from src.llm_resource_explain import RESOURCE_EXPLANATION_COLLECTION
from src.thresholds import (
    set_threshold, delete_threshold, get_thresholds_for_resource,
    list_thresholds, get_breach_state_for_resource,
)


def get_resource_explanation(mongo_uri, db_name, resource_id):
    """The current (latest) LLM explanation + full metrics + threshold
    state for one resource. Returns None if it hasn't been explained yet
    (e.g. --resource-explain-mode priority and it's never been anomalous/
    alerting/breaching)."""
    col = get_collection(mongo_uri, db_name, RESOURCE_EXPLANATION_COLLECTION)
    return col.find_one({"resource_id": resource_id})


def find_resources_by_tag(mongo_uri, db_name, tag_key, tag_value=None, limit=200):
    """All resources (with an explanation on file) carrying a given tag.
    tag_value=None matches any resource that HAS that tag key, regardless
    of value."""
    col = get_collection(mongo_uri, db_name, RESOURCE_EXPLANATION_COLLECTION)
    field = f"tags.{tag_key}"
    query = {field: {"$exists": True}} if tag_value is None else {field: tag_value}
    return list(col.find(query).limit(limit))


def find_resources_by_project(mongo_uri, db_name, project, limit=500):
    col = get_collection(mongo_uri, db_name, RESOURCE_EXPLANATION_COLLECTION)
    return list(col.find({"project": project}).limit(limit))


if __name__ == "__main__":
    import argparse
    import json

    from src.mongo_io import _bson_safe

    p = argparse.ArgumentParser(description="Query per-resource explanations / manage thresholds")
    p.add_argument("--mongo-uri", default="mongodb://localhost:27017")
    p.add_argument("--db", default="mydb")
    sub = p.add_subparsers(dest="cmd", required=True)

    get_p = sub.add_parser("get", help="Fetch one resource's explanation by resource_id")
    get_p.add_argument("--resource-id", required=True)

    tag_p = sub.add_parser("by-tag", help="List resources by tag key[/value]")
    tag_p.add_argument("--key", required=True)
    tag_p.add_argument("--value", default=None)

    set_p = sub.add_parser("set-threshold", help="Set (or replace) the threshold for a resource+metric")
    set_p.add_argument("--resource-id", required=True)
    set_p.add_argument("--metric-name", required=True)
    set_p.add_argument("--operator", required=True, choices=["gt", "gte", "lt", "lte"])
    set_p.add_argument("--value", required=True, type=float)
    set_p.add_argument("--category", default=None)
    set_p.add_argument("--project", default=None)
    set_p.add_argument("--set-by", default=None)

    del_p = sub.add_parser("delete-threshold")
    del_p.add_argument("--resource-id", required=True)
    del_p.add_argument("--metric-name", required=True)

    list_p = sub.add_parser("list-thresholds")
    list_p.add_argument("--project", default=None)
    list_p.add_argument("--category", default=None)

    state_p = sub.add_parser("breach-state")
    state_p.add_argument("--resource-id", required=True)

    args = p.parse_args()

    if args.cmd == "get":
        result = get_resource_explanation(args.mongo_uri, args.db, args.resource_id)
        print(json.dumps(_bson_safe(result), indent=2, default=str) if result else "null")
    elif args.cmd == "by-tag":
        result = find_resources_by_tag(args.mongo_uri, args.db, args.key, args.value)
        print(json.dumps(_bson_safe(result), indent=2, default=str))
    elif args.cmd == "set-threshold":
        doc = set_threshold(
            args.mongo_uri, args.db, args.resource_id, args.metric_name, args.operator, args.value,
            category=args.category, project=args.project, set_by=args.set_by,
        )
        print(json.dumps(_bson_safe(doc), indent=2, default=str))
    elif args.cmd == "delete-threshold":
        n = delete_threshold(args.mongo_uri, args.db, args.resource_id, args.metric_name)
        print(f"deleted {n} threshold rule(s).")
    elif args.cmd == "list-thresholds":
        result = list_thresholds(args.mongo_uri, args.db, project=args.project, category=args.category)
        print(json.dumps(_bson_safe(result), indent=2, default=str))
    elif args.cmd == "breach-state":
        result = get_breach_state_for_resource(args.mongo_uri, args.db, args.resource_id)
        print(json.dumps(_bson_safe(result), indent=2, default=str))
