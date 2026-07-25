"""ingestion/realtime_stream_consumer.py — production real-time worker.

Implements Section 5 of resource_metadata_enrichment_pipeline.py as a
standalone, always-running process (rather than a notebook cell), reading
from the two live source streams:

    db.cloud_account_data_collection_daily_updated  (Account Metrics Collection)
    db.service_resource_inventory_v2_updated         (Service Resource Inventory)

Flow (matches the flowchart):

    Service Resource Inventory --> MongoDB Change Stream --> fan out to:
        1) Inventory Lookup Cache (element_id -> account_id -> latest doc)
        2) Data Lake collection (immutable, append-only history)

    Account Metrics Collection --> MongoDB Change Stream --> Stream Consumer
        --> lookup(resourceId, account_id) in cache --> Metadata Enrichment
        --> upsert into enriched_data_lake (the JSR pipeline's --source)

Two background threads run concurrently against one shared in-memory cache,
bootstrapped once from a batch pass over the inventory collection before
either change stream starts (so lookups succeed immediately instead of
only for documents that change after startup).

Requirements:
  - MongoDB must be running as a replica set (Atlas clusters have this by
    default; a local single-node mongod needs `rs.initiate()` once).
  - pip install pymongo

Resume tokens are persisted to small local JSON files so a restart resumes
each stream from where it left off instead of missing changes that
happened while the process was down, or reprocessing the whole history.
"""
import argparse
import json
import signal
import threading
from datetime import datetime, timezone
from pathlib import Path

from bson import ObjectId
from pymongo import MongoClient
from pymongo.errors import PyMongoError

shutdown_event = threading.Event()


def normalize_id(value):
    """ObjectId('aaa') != 'aaa' in Python even though they're the same id,
    so every join key is coerced to a plain string before comparison."""
    if value is None:
        return None
    if isinstance(value, ObjectId):
        return str(value)
    return str(value).strip()


# def build_inventory_lookup_cache(inventory_coll, key_field, account_field, updated_field):
#     """Bootstrap pass: cache[element_id][account_id] = latest_doc."""
#     cache = {}
#     skipped = 0
#     for doc in inventory_coll.find():
#         elem_key = normalize_id(doc.get(key_field))
#         acct_key = normalize_id(doc.get(account_field))
#         if elem_key is None or acct_key is None:
#             skipped += 1
#             continue
#         bucket = cache.setdefault(elem_key, {})
#         existing = bucket.get(acct_key)
#         if existing is None:
#             bucket[acct_key] = doc
#             continue
#         new_ts, existing_ts = doc.get(updated_field), existing.get(updated_field)
#         if new_ts is not None and (existing_ts is None or new_ts > existing_ts):
#             bucket[acct_key] = doc
#     total = sum(len(b) for b in cache.values())
#     print(f"[bootstrap] cache: {len(cache)} bucket(s), {total} (element_id, account_id) entries, {skipped} skipped.")
#     return cache

def build_inventory_lookup_cache(inventory_coll, key_field, account_field, updated_field):
    """Lazy cache: start empty, fill on demand instead of preloading 10M docs."""
    inventory_coll.create_index([(key_field, 1), (account_field, 1)])
    print("[bootstrap] lazy inventory cache ready (index ensured, cache starts empty).")
    return {}


# def lookup_inventory(cache, resource_id, account_id):
#     if resource_id is None or account_id is None:
#         return None
#     bucket = cache.get(resource_id)
#     return bucket.get(account_id) if bucket else None

def lookup_inventory(cache, resource_id, account_id, inventory_coll=None,
                      key_field=None, account_field=None):
    if resource_id is None or account_id is None:
        return None
    bucket = cache.get(resource_id)
    if bucket is not None and account_id in bucket:
        return bucket[account_id]
    if inventory_coll is None:
        return None
    doc = inventory_coll.find_one({key_field: resource_id, account_field: account_id})
    if doc is not None:
        cache.setdefault(resource_id, {})[account_id] = doc
    return doc

def merge_documents(account_doc, inventory_doc, account_precedence=True,
                     preserve_account_id=True, authoritative_fields=("project", "category")):
    """See resource_metadata_enrichment_pipeline.py::merge_documents for the
    full rationale. `project` and `category` are inventory-authoritative
    regardless of general precedence: a resource belongs to exactly one
    project and one category by definition (per the Service Resource
    Inventory), so a stale value on the account doc must never win the merge.
    """
    base, overlay = (inventory_doc, account_doc) if account_precedence else (account_doc, inventory_doc)
    merged = {}
    for d in (base, overlay):
        for k, v in d.items():
            if k != "_id":
                merged[k] = v
    for k in authoritative_fields:
        if k in inventory_doc:
            merged[k] = inventory_doc[k]
    merged["account_doc_id"] = account_doc.get("_id")
    merged["inventory_doc_id"] = inventory_doc.get("_id")
    merged["merged_at"] = datetime.now(timezone.utc)
    if preserve_account_id:
        merged["_id"] = account_doc["_id"]
    return merged


# ---------------------------------------------------------------------------
# Resume token persistence — so a restart doesn't miss changes or replay
# the whole collection from scratch.
# ---------------------------------------------------------------------------

def _token_path(state_dir, name):
    return Path(state_dir) / f"resume_token_{name}.json"


def load_resume_token(state_dir, name):
    path = _token_path(state_dir, name)
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return None
    return None


def save_resume_token(state_dir, name, token):
    path = _token_path(state_dir, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(token, default=str))


# ---------------------------------------------------------------------------
# Stream consumers
# ---------------------------------------------------------------------------

NOT_REPLICA_SET_CODE = 40573
_replica_set_hint_shown = threading.Event()


def _explain_and_backoff(name, e):
    """On the 'not a replica set' error, change streams can never succeed
    until the server is reconfigured — retrying every 5s just floods the
    log. Print the fix once, then back off slower for that specific error.
    Any other PyMongoError keeps the original fast 5s retry (transient
    network blips, etc.).
    """
    code = getattr(e, "code", None)
    if code is None and hasattr(e, "details"):
        details = getattr(e, "details", None) or {}
        code = details.get("code")
    if code == NOT_REPLICA_SET_CODE or "only supported on replica sets" in str(e):
        if not _replica_set_hint_shown.is_set():
            _replica_set_hint_shown.set()
            print(
                f"[{name}] MongoDB change streams need a replica set — this server is "
                "running as a standalone mongod. One-time fix for local dev:\n"
                "    mongosh --eval 'rs.initiate()'\n"
                "  (Atlas clusters already are replica sets, so this only applies locally.) "
                "Retrying every 30s in case this gets fixed while running..."
            )
        shutdown_event.wait(30)
    else:
        print(f"[{name}] stream error, retrying in 5s: {e}")
        shutdown_event.wait(5)


def watch_inventory_changes(inventory_coll, data_lake_coll, cache, cfg, state_dir):
    resume_token = load_resume_token(state_dir, "inventory")
    watch_kwargs = dict(full_document="updateLookup", full_document_before_change="whenAvailable")
    if resume_token:
        watch_kwargs["resume_after"] = resume_token
        print("[inventory] resuming from saved token.")

    while not shutdown_event.is_set():
        try:
            with inventory_coll.watch(**watch_kwargs) as stream:
                while not shutdown_event.is_set():
                    change = stream.try_next()
                    if change is None:
                        continue
                    op_type = change["operationType"]
                    doc = change.get("fullDocument")
                    pre_image = change.get("fullDocumentBeforeChange")

                    source_doc = doc if doc is not None else pre_image
                    record = dict(source_doc) if source_doc else {}
                    record.pop("_id", None)
                    record["source_id"] = change.get("documentKey", {}).get("_id")
                    record["operation_type"] = op_type
                    record["captured_at"] = datetime.now(timezone.utc)
                    data_lake_coll.insert_one(record)

                    if op_type != "delete" and doc:
                        elem_key = normalize_id(doc.get(cfg.element_id_field))
                        acct_key = normalize_id(doc.get(cfg.account_id_field_inventory))
                        if elem_key is not None and acct_key is not None:
                            bucket = cache.setdefault(elem_key, {})
                            existing = bucket.get(acct_key)
                            new_ts = doc.get(cfg.updated_at_field)
                            existing_ts = existing.get(cfg.updated_at_field) if existing else None
                            if existing is None or new_ts is None or existing_ts is None or new_ts > existing_ts:
                                bucket[acct_key] = doc
                                print(f"[inventory] cache updated: element_id={elem_key!r} account_id={acct_key!r}")

                    save_resume_token(state_dir, "inventory", stream.resume_token)
        except PyMongoError as e:
            if shutdown_event.is_set():
                break
            _explain_and_backoff("inventory", e)


# def watch_account_changes(account_coll, output_coll, cache, cfg, state_dir , inventory_coll):
def watch_account_changes(account_coll, output_coll, inventory_coll, cache, cfg, state_dir,):
    resume_token = load_resume_token(state_dir, "account")
    watch_kwargs = dict(full_document="updateLookup")
    if resume_token:
        watch_kwargs["resume_after"] = resume_token
        print("[account] resuming from saved token.")

    while not shutdown_event.is_set():
        try:
            with account_coll.watch(**watch_kwargs) as stream:
                while not shutdown_event.is_set():
                    change = stream.try_next()
                    if change is None:
                        continue
                    if change["operationType"] not in ("insert", "update", "replace"):
                        save_resume_token(state_dir, "account", stream.resume_token)
                        continue
                    account_doc = change.get("fullDocument")
                    if not account_doc:
                        save_resume_token(state_dir, "account", stream.resume_token)
                        continue

                    resource_id = normalize_id(account_doc.get(cfg.resource_id_field))
                    account_id = normalize_id(account_doc.get(cfg.account_id_field_account))
                    
                    inventory_doc = lookup_inventory(
                        cache,
                        resource_id,
                        account_id,
                        inventory_coll=inventory_coll,
                        key_field=cfg.element_id_field,
                        account_field=cfg.account_id_field_inventory,
                    )

                    if inventory_doc is None:
                        print(f"[account] no inventory match: resourceId={resource_id!r} account_id={account_id!r}, skipping")
                    else:
                        merged = merge_documents(
                            account_doc, inventory_doc,
                            account_precedence=cfg.account_fields_take_precedence,
                            preserve_account_id=cfg.preserve_account_id,
                        )
                        output_coll.update_one({"_id": merged["_id"]}, {"$set": merged}, upsert=True)
                        print(f"[account] merged resourceId={resource_id!r} account_id={account_id!r} -> {cfg.output_collection}")

                    save_resume_token(state_dir, "account", stream.resume_token)
        except PyMongoError as e:
            if shutdown_event.is_set():
                break
            _explain_and_backoff("account", e)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def parser():
    p = argparse.ArgumentParser(description="Real-time account+inventory stream merger")
    p.add_argument("--mongo-uri", default="mongodb://localhost:27017")
    p.add_argument("--db", default="mydb")
    p.add_argument("--account-collection", default="cloud_account_data_collection_daily_updated")
    p.add_argument("--inventory-collection", default="service_resource_inventory_v2_updated")
    p.add_argument("--output-collection", default="enriched_data_lake")
    p.add_argument("--data-lake-collection", default="service_resource_inventory_data_lake")
    p.add_argument("--resource-id-field", default="resourceId")
    p.add_argument("--element-id-field", default="element_id")
    p.add_argument("--account-id-field-account", default="account_id")
    p.add_argument("--account-id-field-inventory", default="account_id")
    p.add_argument("--updated-at-field", default="updatedAt")
    p.add_argument("--account-fields-take-precedence", action="store_true", default=True)
    p.add_argument("--preserve-account-id", action="store_true", default=True)
    p.add_argument("--state-dir", default=".stream_state",
                    help="Where resume tokens are persisted, so a restart doesn't "
                         "miss changes or replay the whole collection.")
    return p


def main():
    args = parser().parse_args()
    client = MongoClient(args.mongo_uri)
    db = client[args.db]

    account_coll = db[args.account_collection]
    inventory_coll = db[args.inventory_collection]
    output_coll = db[args.output_collection]
    data_lake_coll = db[args.data_lake_collection]

    print(f"Connected to {args.db}. Account: {args.account_collection} "
          f"({account_coll.estimated_document_count()} docs). "
          f"Inventory: {args.inventory_collection} ({inventory_coll.estimated_document_count()} docs).")

    # Pre/post images required for full delete history in the Data Lake (MongoDB 6.0+).
    try:
        db.command({"collMod": args.inventory_collection,
                     "changeStreamPreAndPostImages": {"enabled": True}})
        print(f"Pre/post images enabled on {args.inventory_collection}.")
    except PyMongoError as e:
        print(f"Could not enable pre/post images (check MongoDB version >= 6.0): {e}")

    print("Bootstrapping inventory lookup cache from current state...")
    cache = build_inventory_lookup_cache(
        inventory_coll, args.element_id_field, args.account_id_field_inventory, args.updated_at_field,
    )

    def handle_shutdown(signum, frame):
        print("\nShutdown signal received, stopping stream consumers...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    inv_thread = threading.Thread(
        target=watch_inventory_changes,
        args=(inventory_coll, data_lake_coll, cache, args, args.state_dir),
        daemon=True,
    )
    acct_thread = threading.Thread(
        target=watch_account_changes,
        args=(
            account_coll,
            output_coll,
            inventory_coll,
            cache,
            args,
            args.state_dir,
        ),
        daemon=True,
    )
    inv_thread.start()
    acct_thread.start()
    print("Change stream consumers running. Press Ctrl+C to stop.")

    while not shutdown_event.is_set():
        shutdown_event.wait(1)

    inv_thread.join(timeout=5)
    acct_thread.join(timeout=5)
    print("Stopped.")


if __name__ == "__main__":
    main()
