"""mongo_io.py — MongoDB I/O helpers.

Changes in v6:
- write_predictions now upserts on (resource_id, prediction_timestamp) for idempotency.
- Predictions are tagged with a deterministic idempotent_key to prevent double-writes.
- Output schema validation runs before every write; invalid predictions are logged and skipped.
"""
import math

import numpy as np
import pandas as pd
from pymongo import MongoClient, UpdateOne

from .data_quality import idempotent_key, validate_predictions


def get_collection(uri, db_name, collection_name):
    client = MongoClient(uri)
    return client[db_name][collection_name]


def list_collections_with_prefix(uri, db_name, prefix):
    """Return all collection names in db_name starting with `prefix`.

    Used to discover the set of per-project output collections that
    actually exist (e.g. `22prediction_logs_compute__proj_a`,
    `22prediction_logs_compute__proj_b`) without the caller needing to
    track project names separately.
    """
    client = MongoClient(uri)
    return sorted(
        name for name in client[db_name].list_collection_names()
        if name.startswith(prefix)
    )


def load_documents(uri, db_name, collection_name, limit=50000, categories=None, extra_query=None):
    """Load raw documents from Mongo.

    No field projection: we fetch the FULL document. Dedup hashes the
    entire raw document (see data_quality.dedup_documents), so trimming
    fields here would silently make two genuinely-different documents
    (e.g. differing only in `updated_at` or `summary_details`) look
    identical and get wrongly dropped as duplicates. normalize_documents()
    only reads a known subset of fields, so the extra fields are simply
    ignored downstream — no behavior change there, just more complete dedup.

    `extra_query` (optional dict) is merged into the Mongo filter as-is —
    e.g. {"merged_at": {"$gt": last_ts}} for incremental predict cycles, or
    {"$or": [{"resourceId": {"$in": ids}}, {"resource_id": {"$in": ids}}]}
    to pull history for a specific set of changed resources. Kept separate
    from the `categories` filter below so both can be combined.
    """
    collection = get_collection(uri, db_name, collection_name)
    query = {}
    if extra_query:
        query.update(extra_query)

# The enriched_data_lake contains provider-specific categories
# (Virtual_Machines, Storage_Accounts, etc.). They are normalized later,
# so don't filter them here.
    if categories and collection_name != "enriched_data_lake":
        query["category"] = {"$in": list(categories)}
    cursor = (
        collection.find(query)
        .sort([("from_date", -1), ("created_at", -1)])
        .batch_size(2000)
    )
    if limit and limit > 0:
        cursor = cursor.limit(limit)
    try:
        return list(cursor)
    finally:
        cursor.close()


def iter_document_batches(uri, db_name, collection_name, batch_size=20000, categories=None, extra_query=None):
    """Same query/sort/filter as load_documents(), but yields lists of up to
    `batch_size` raw documents at a time via the driver's own cursor batching,
    instead of materializing the entire result set into one Python list.

    Use this for any full-collection scan (training) where the source
    collection can grow arbitrarily large. Nothing about the query itself
    changes — same rows are visited, just in chunks and in cursor order.
    The caller is responsible for dropping its reference to each batch
    before requesting the next one, so peak memory stays at ~1 batch,
    not the whole collection.
    """
    collection = get_collection(uri, db_name, collection_name)
    query = dict(extra_query) if extra_query else {}
    if categories and collection_name != "enriched_data_lake":
        query["category"] = {"$in": list(categories)}
    cursor = (
        collection.find(query)
        .sort([("from_date", -1), ("created_at", -1)])
        .batch_size(batch_size)
    )
    try:
        batch = []
        for doc in cursor:
            batch.append(doc)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
    finally:
        cursor.close()


def _bson_safe(obj):
    """Recursively convert prediction dicts to BSON-safe types."""
    if obj is None:
        return None
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, str):
        return obj
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, np.ndarray):
        return [_bson_safe(v) for v in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): _bson_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_bson_safe(v) for v in obj]
    try:
        return str(obj)
    except Exception:
        return None


def write_predictions(uri, db_name, collection_name, predictions, verbose=True):
    """Upsert predictions idempotently.

    Uses (resource_id, metric_combination, prediction_timestamp) as the
    upsert key so re-running the pipeline after a crash does not produce
    duplicate documents, and so multiple predictions for the same resourceId
    (one per distinct metric combination it reports) don't overwrite each
    other.

    Also runs output schema validation before writing — invalid predictions are
    skipped and their errors printed rather than crashing the pipeline.

    Returns number of upserted/modified documents.
    """
    if not predictions:
        return 0

    # --- Schema validation ---
    valid, errors = validate_predictions(predictions)
    if errors:
        print(f"  [write_predictions] {len(errors)} validation error(s), skipping those predictions:")
        for e in errors[:10]:  # cap output
            print(f"    {e}")
    if not valid:
        return 0

    collection = get_collection(uri, db_name, collection_name)

    # Build upsert operations in batches of 500
    batch_size = 500
    upserted = 0

    for i in range(0, len(valid), batch_size):
        batch = valid[i: i + batch_size]
        ops = []
        for p in batch:
            safe = _bson_safe(p)
            rid = safe.get("resource_id", "")
            pts = safe.get("prediction_timestamp", "")
            mc  = safe.get("metric_combination", "")
            ikey = idempotent_key(rid, pts, mc)
            safe["_idempotent_key"] = ikey
            ops.append(UpdateOne(
                {"_idempotent_key": ikey},
                {"$set": safe},
                upsert=True,
            ))
        result = collection.bulk_write(ops, ordered=False)
        upserted += result.upserted_count + result.modified_count

    return upserted
