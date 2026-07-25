# %% [markdown]
# # Resource Metadata Enrichment Pipeline
#
# Implements the merge stage of the architecture:
#
# ```
# Service Resource Inventory ---> Inventory Stream ---> Inventory Lookup Cache (element_id -> metadata)
#                                                                  ^
#                                                                  | lookup(resourceId)
# Account Metrics Collection ---> Account Metrics Stream --------┘
#                                             |
#                                             v
#                                     Stream Consumer
#                                             |
#                                             v
#                                   Metadata Enrichment
#                                             |
#                                             v
#                               New MongoDB Collection (merged output)
# ```
#
# ...and the inventory-side ingestion architecture:
#
# ```
# Inventory Collection
#          |
#          v
# MongoDB Change Stream
#          |
#     ┌────┴────┐
#     v         v
# Update Lookup   Write to Data Lake
# Cache (latest)  (immutable history)
#     |
#     v
# Real-time Enrichment
#     |
#     v
# Enriched Output Collection
# ```
#
# What this script does:
#
# 1. **Bucketed hash lookup cache.** Every inventory doc's `element_id` is hashed by
#    Python's dict (an O(1) hash table) into a "bucket". Each bucket holds the distinct
#    `account_id`s seen for that resource: `cache[element_id][account_id] = latest_doc`.
#    This is exactly what lets the same resourceId map to different metadata per account.
# 2. **Composite join.** Account docs are matched on `(resourceId -> element_id)` to find
#    the bucket, then `(account_id -> account_id)` within that bucket to find the exact doc.
# 3. **Latest-wins in the cache.** Within a bucket, if multiple inventory docs share the same
#    account_id, only the one with the greatest `UPDATED_AT_FIELD` is kept.
# 4. **Dual-write on inventory change.** Every inventory change-stream event is written
#    unconditionally (as an immutable, append-only record) to the Data Lake collection,
#    AND used to conditionally update the in-memory lookup cache bucket (only if it's
#    newer than what's already cached). The Data Lake keeps full history; the cache keeps
#    only the current, latest version of each resource.
# 5. **Merge & write.** Matched account + inventory docs are merged and upserted into the
#    output collection.
#
# Section 3 is a batch/backfill pipeline (one-time or periodic pass).
# Section 5 is the real-time version using MongoDB Change Streams, including the
# Data Lake dual-write for the inventory side.
#
# In VS Code: install the "Jupyter" extension, then run cells individually with
# "Run Cell" (the # %% markers create Interactive Window cells), or run the whole
# file top to bottom with "Run Python File".

# %% [markdown]
# ## 1. Install dependencies
# Run once in your terminal (not required every time):
# `pip install pymongo`

# %% [markdown]
# ## 2. Configuration

# %%
import os
# --- Connection ---
# In Colab, prefer storing this as a secret (key icon in left sidebar) rather than hardcoding it.
MONGO_URI = 'mongodb://localhost:27017'
print(MONGO_URI)
DB_NAME = 'mydb'
print('DB_NAME:', DB_NAME)
# --- Source collections (per the diagram) ---
ACCOUNT_COLLECTION = 'cloud_account_data_collection_daily_updated'  # left branch: Account Metrics Collection
INVENTORY_COLLECTION = 'service_resource_inventory_v2_updated'      # right branch: Service Resource Inventory
# --- Output collection (new merged collection) ---
OUTPUT_COLLECTION = 'enriched_data_lake'
# --- Data Lake collection (immutable, append-only history of every inventory change) ---
DATA_LAKE_COLLECTION = 'service_resource_inventory_data_lake'
# --- Join keys ---
RESOURCE_ID_FIELD = 'resourceId'   # field on account documents
ELEMENT_ID_FIELD = 'element_id'    # field on inventory documents (the lookup key)
ACCOUNT_ID_FIELD_ACCOUNT = 'account_id'   # account_id field name on account documents
ACCOUNT_ID_FIELD_INVENTORY = 'account_id'  # account_id field name on inventory documents
# The join is now: (resourceId == element_id) AND (account_id == account_id)
# --- "Latest version" field ---
# When the inventory collection has multiple docs for the same (element_id, account_id),
# the cache keeps only the one with the greatest value in this field.
UPDATED_AT_FIELD = 'updatedAt'   # e.g. 'updatedAt', 'lastUpdated', 'modifiedDate' -- adjust to your schema
# --- Merge behavior ---
# If both docs have a field with the same name (other than the join keys / _id),
# the account doc's value wins by default. Set to False to let inventory win instead.
ACCOUNT_FIELDS_TAKE_PRECEDENCE = True
# Keep the original account _id, or let Mongo generate a fresh one for the merged doc?
PRESERVE_ACCOUNT_ID = True
# Batch size for bulk writes
BATCH_SIZE = 10
# --- Testing / limited run ---
# Set to an int (e.g. 500) to only process the first N records from each collection.
# Set to None to process the full collections.
LIMIT_RECORDS = None
print('Config loaded. DB:', DB_NAME, '| LIMIT_RECORDS:', LIMIT_RECORDS)

# %% [markdown]
# ## 3. Batch pipeline: build hash table -> lookup -> merge -> write

# %%
from pymongo import MongoClient, UpdateOne
from datetime import datetime, timezone

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

account_coll = db[ACCOUNT_COLLECTION]
inventory_coll = db[INVENTORY_COLLECTION]
output_coll = db[OUTPUT_COLLECTION]
data_lake_coll = db[DATA_LAKE_COLLECTION]

print('Connected. Collections:')
print(' -', ACCOUNT_COLLECTION, 'count:', account_coll.estimated_document_count())
print(' -', INVENTORY_COLLECTION, 'count:', inventory_coll.estimated_document_count())
print(' -', DATA_LAKE_COLLECTION, 'count:', data_lake_coll.estimated_document_count())

# %%
from bson import ObjectId

def normalize_id(value):
    """Coerces a join-key value to a plain string for comparison.

    Handles the common case where one collection stores an id as a MongoDB
    ObjectId and the other stores the "same" value as a plain string --
    ObjectId('aaa') != 'aaa' in Python even though they represent the same id,
    so without this normalization every lookup would silently miss.
    """
    if value is None:
        return None
    if isinstance(value, ObjectId):
        return str(value)
    return str(value).strip()


def build_inventory_lookup_cache(inventory_collection, key_field=ELEMENT_ID_FIELD,
                                  account_field=ACCOUNT_ID_FIELD_INVENTORY,
                                  updated_field=UPDATED_AT_FIELD, limit=None):
    """Builds the Inventory Lookup Cache as a bucketed hash table.

    Structure: cache[element_id][account_id] = latest_doc

    `element_id` is hashed (via Python's dict, an O(1) hash table) into a top-level
    "bucket". Each bucket stores the distinct account_ids seen for that resource --
    this is what lets the same resourceId/element_id map to different metadata for
    different accounts. Within a bucket, if multiple inventory docs share the same
    account_id, only the one with the greatest `updated_field` value is kept, so the
    cache always holds the latest-updated version of each (element_id, account_id) pair.

    If `limit` is set, only the first `limit` inventory documents are scanned
    (useful for quick tests; for correctness across the whole collection, leave
    this as None so every duplicate is actually compared).
    """
    cache = {}
    skipped = 0
    replaced_by_newer = 0
    kept_older_duplicate = 0

    cursor = inventory_collection.find()
    if limit is not None:
        cursor = cursor.limit(limit)

    for doc in cursor:
        elem_key = normalize_id(doc.get(key_field))
        acct_key = normalize_id(doc.get(account_field))
        if elem_key is None or acct_key is None:
            skipped += 1
            continue

        bucket = cache.setdefault(elem_key, {})
        existing = bucket.get(acct_key)

        if existing is None:
            bucket[acct_key] = doc
            continue

        new_ts = doc.get(updated_field)
        existing_ts = existing.get(updated_field)

        # Replace only if the new doc has a timestamp and it's newer (or the existing
        # entry has no timestamp at all, in which case any timestamped doc wins).
        if new_ts is not None and (existing_ts is None or new_ts > existing_ts):
            bucket[acct_key] = doc
            replaced_by_newer += 1
        else:
            kept_older_duplicate += 1

    total_entries = sum(len(bucket) for bucket in cache.values())
    print(f'Built inventory lookup cache with {len(cache)} bucket(s) '
          f'({key_field} values) holding {total_entries} (element_id, account_id) entries.')
    print(f'  {skipped} docs skipped (missing {key_field} or {account_field}).')
    print(f'  {replaced_by_newer} duplicate keys resolved by keeping the newer {updated_field}.')
    print(f'  {kept_older_duplicate} older duplicate docs discarded in favor of an existing newer entry.')
    return cache


def lookup_inventory(cache, resource_id, account_id):
    """Looks up an inventory doc: hash resourceId to its bucket, then find account_id within it."""
    if resource_id is None or account_id is None:
        return None
    bucket = cache.get(resource_id)
    if not bucket:
        return None
    return bucket.get(account_id)


inventory_cache = build_inventory_lookup_cache(inventory_coll, limit=LIMIT_RECORDS)

# %%
# Fields where the inventory side is always authoritative, regardless of
# ACCOUNT_FIELDS_TAKE_PRECEDENCE. A resource belongs to exactly one project
# (per the account/project/resourceId cardinality rules), and `project` is
# sourced from Service Resource Inventory, not Account Metrics -- if an
# account doc happens to carry a stale/incorrect `project` value, letting it
# win here would silently misroute every downstream prediction (and any
# per-project LLM aggregation) for that resource. `category` is the same
# story: the inventory record is the system of record for what a resource
# IS (Compute/Storage/Network/...), while account metric docs sometimes
# carry a looser/legacy category label on the raw metric payload -- that
# should never override the inventory's classification.
INVENTORY_AUTHORITATIVE_FIELDS = {'project', 'category'}


def merge_documents(account_doc, inventory_doc):
    """Merges fields from the account doc and the matched inventory doc into one output doc."""
    merged = {}

    if ACCOUNT_FIELDS_TAKE_PRECEDENCE:
        base, overlay = inventory_doc, account_doc
    else:
        base, overlay = account_doc, inventory_doc

    for d in (base, overlay):
        for k, v in d.items():
            if k == '_id':
                continue
            merged[k] = v

    # Inventory wins unconditionally for authoritative fields, overriding
    # whatever the precedence order above just decided.
    for k in INVENTORY_AUTHORITATIVE_FIELDS:
        if k in inventory_doc:
            merged[k] = inventory_doc[k]

    # Keep provenance of both source ids, since both _id fields get dropped above
    merged['account_doc_id'] = account_doc.get('_id')
    merged['inventory_doc_id'] = inventory_doc.get('_id')
    merged['merged_at'] = datetime.now(timezone.utc)

    if PRESERVE_ACCOUNT_ID:
        merged['_id'] = account_doc['_id']

    return merged


def run_batch_merge(limit=None):
    """If `limit` is set, only the first `limit` account documents are processed."""
    matched, unmatched = 0, 0
    ops = []

    def flush(ops):
        if not ops:
            return
        output_coll.bulk_write(ops, ordered=False)
        ops.clear()

    cursor = account_coll.find()
    if limit is not None:
        cursor = cursor.limit(limit)

    for account_doc in cursor:
        resource_id = normalize_id(account_doc.get(RESOURCE_ID_FIELD))
        account_id = normalize_id(account_doc.get(ACCOUNT_ID_FIELD_ACCOUNT))

        inventory_doc = lookup_inventory(inventory_cache, resource_id, account_id)

        if inventory_doc is None:
            unmatched += 1
            continue  # no matching (element_id, account_id) in the lookup cache; skip (or route to a dead-letter collection)

        merged_doc = merge_documents(account_doc, inventory_doc)
        ops.append(
            UpdateOne({'_id': merged_doc['_id']}, {'$set': merged_doc}, upsert=True)
        )
        matched += 1

        if len(ops) >= BATCH_SIZE:
            flush(ops)

    flush(ops)
    print(f'Done. Matched & merged: {matched}, unmatched (no inventory hit): {unmatched}')

run_batch_merge(limit=LIMIT_RECORDS)

# %% [markdown]
# ## 3b. Debug: inspect raw join-key values (run this if matched count is 0)
#
# Prints the actual field names and values from a few docs on both sides so you can
# visually compare formatting -- e.g. casing, whitespace, prefixes/suffixes, or a
# completely different field name than what's configured above.

# %%
print('--- Sample ACCOUNT docs (first 3) ---')
for doc in account_coll.find().limit(3):
    print({
        '_id': doc.get('_id'),
        RESOURCE_ID_FIELD: doc.get(RESOURCE_ID_FIELD),
        ACCOUNT_ID_FIELD_ACCOUNT: doc.get(ACCOUNT_ID_FIELD_ACCOUNT),
        'all_keys': list(doc.keys()),
    })

print()
print('--- Sample INVENTORY docs (first 3) ---')
for doc in inventory_coll.find().limit(3):
    print({
        '_id': doc.get('_id'),
        ELEMENT_ID_FIELD: doc.get(ELEMENT_ID_FIELD),
        ACCOUNT_ID_FIELD_INVENTORY: doc.get(ACCOUNT_ID_FIELD_INVENTORY),
        'all_keys': list(doc.keys()),
    })

# %%
# Show exactly which buckets/keys exist in the cache, and which account resourceId
# values (from a larger sample) come anywhere close to matching.
print('--- All (element_id -> account_ids) buckets currently in inventory_cache ---')
for elem_key, bucket in inventory_cache.items():
    print(' ', elem_key, '->', list(bucket.keys()))

print()
print('--- resourceId/account_id pairs from first 20 account docs (normalized) ---')
for doc in account_coll.find().limit(20):
    resource_id = normalize_id(doc.get(RESOURCE_ID_FIELD))
    account_id = normalize_id(doc.get(ACCOUNT_ID_FIELD_ACCOUNT))
    hit = lookup_inventory(inventory_cache, resource_id, account_id) is not None
    print(f'  ({resource_id!r}, {account_id!r})  {"<-- MATCH" if hit else ""}')

# %%
# Sanity check across a larger sample.
loose_matches = 0
for doc in account_coll.find().limit(200):
    resource_id = normalize_id(doc.get(RESOURCE_ID_FIELD))
    account_id = normalize_id(doc.get(ACCOUNT_ID_FIELD_ACCOUNT))
    if lookup_inventory(inventory_cache, resource_id, account_id) is not None:
        loose_matches += 1

print(f'Normalized matches in first 200 account docs: {loose_matches}')



# %%
for doc in output_coll.find().limit(3):
    print(doc)
    print('---')

# %% [markdown]
# ## 4b. (Optional) Seed the Data Lake with a snapshot of current inventory
#
# The Data Lake is normally populated by `watch_inventory_changes` as changes happen.
# Run this once if you also want an initial history record for every document that
# already exists in the inventory collection (so the Data Lake isn't empty until the
# next real change occurs).

# %%
def seed_data_lake(limit=None):
    cursor = inventory_coll.find()
    if limit is not None:
        cursor = cursor.limit(limit)

    records = []
    for doc in cursor:
        record = dict(doc)
        record.pop('_id', None)
        record['source_id'] = doc.get('_id')
        record['operation_type'] = 'snapshot'
        record['captured_at'] = datetime.now(timezone.utc)
        records.append(record)

    if records:
        data_lake_coll.insert_many(records)
    print(f'Seeded Data Lake with {len(records)} snapshot record(s).')

seed_data_lake(limit=LIMIT_RECORDS)

# %% [markdown]
# ## 4c. Enable pre-images on the inventory collection (required for full delete history)
#
# By default, MongoDB change streams don't include document content on deletes.
# `changeStreamPreAndPostImages` makes MongoDB retain a snapshot of the document as it
# was right before each change, so `watch_inventory_changes` can log the *complete*
# document into the Data Lake even for deletes -- not just the id and timestamp.
#
# Requirements: MongoDB 6.0+, and this only needs to be run once per collection (it's
# a collection-level setting, safe to re-run / idempotent).

# %%
try:
    db.command({
        'collMod': INVENTORY_COLLECTION,
        'changeStreamPreAndPostImages': {'enabled': True},
    })
    print(f'Pre/post images enabled on {INVENTORY_COLLECTION}.')
except Exception as e:
    print(f'Could not enable pre/post images (check MongoDB version >= 6.0): {e}')

# %% [markdown]
# ## 5. Real-time version using MongoDB Change Streams (incl. Data Lake dual-write)
#
# Implements both flowcharts together:
# - **Account side**: `watch_account_changes` is the Stream Consumer + Metadata Enrichment
#   step -- it enriches each incoming account doc using the lookup cache and writes to
#   the output collection.
# - **Inventory side**: `watch_inventory_changes` is the fan-out step -- every inventory
#   change is written to the Data Lake (immutable history, always) and used to update the
#   Lookup Cache bucket (only if newer, so the cache always holds the current state).
#
# Notes:
# - Change Streams require MongoDB to be running as a replica set (Atlas clusters have this by default).
# - This blocks indefinitely, like a real stream consumer. Interrupt (Ctrl+C) to stop.
# - For production use, run this as a standalone worker process, and persist a `resumeToken`
#   so the consumer can restart from where it left off after a restart.

# %%
import threading

def watch_inventory_changes(cache):
    """Inventory-side real-time ingestion, matching the flowchart:

        Inventory Collection -> MongoDB Change Stream -> (fan out to both:)
            1) Update Lookup Cache (latest resource state)
            2) Write to Data Lake (immutable history)

    Every change event is written to the Data Lake unconditionally (append-only,
    full history, never overwritten). For deletes, the pre-image (the document as it
    was right before deletion, enabled in Section 4c) is used so the Data Lake keeps
    the full document content even for deletes -- not just an id and timestamp.
    The same event only updates the in-memory lookup cache bucket if it's newer than
    what's already cached there -- so the cache always reflects just the current,
    latest version of each resource.
    """
    with inventory_coll.watch(
        full_document='updateLookup',
        full_document_before_change='whenAvailable',
    ) as stream:
        for change in stream:
            op_type = change['operationType']
            doc = change.get('fullDocument')
            pre_image = change.get('fullDocumentBeforeChange')

            # --- 1) Write to Data Lake: immutable, append-only history ---
            # Always insert a new record, even for updates/deletes -- the Data Lake
            # never overwrites, so the full change history is preserved. For deletes,
            # fall back to the pre-image so the deleted document's fields are captured.
            source_doc = doc if doc is not None else pre_image
            data_lake_record = dict(source_doc) if source_doc else {}
            data_lake_record.pop('_id', None)  # avoid _id collisions across repeated snapshots
            data_lake_record['source_id'] = change.get('documentKey', {}).get('_id')
            data_lake_record['operation_type'] = op_type
            data_lake_record['captured_at'] = datetime.now(timezone.utc)
            data_lake_coll.insert_one(data_lake_record)

            if op_type == 'delete' or not doc:
                # Delete history (with full pre-image content) was captured above, but
                # there's no current document to key a cache update off of -- skip the
                # cache side so it keeps serving the last-known state.
                continue

            # --- 2) Update Lookup Cache: only the latest version per bucket ---
            elem_key = normalize_id(doc.get(ELEMENT_ID_FIELD))
            acct_key = normalize_id(doc.get(ACCOUNT_ID_FIELD_INVENTORY))
            if elem_key is None or acct_key is None:
                continue

            bucket = cache.setdefault(elem_key, {})
            existing = bucket.get(acct_key)
            new_ts = doc.get(UPDATED_AT_FIELD)
            existing_ts = existing.get(UPDATED_AT_FIELD) if existing else None

            if existing is None or new_ts is None or existing_ts is None or new_ts > existing_ts:
                bucket[acct_key] = doc


def watch_account_changes(cache):
    """Stream Consumer + Metadata Enrichment: enriches each incoming account doc and writes it out."""
    with account_coll.watch(full_document='updateLookup') as stream:
        for change in stream:
            if change['operationType'] not in ('insert', 'update', 'replace'):
                continue
            account_doc = change.get('fullDocument')
            if not account_doc:
                continue

            resource_id = normalize_id(account_doc.get(RESOURCE_ID_FIELD))
            account_id = normalize_id(account_doc.get(ACCOUNT_ID_FIELD_ACCOUNT))
            inventory_doc = lookup_inventory(cache, resource_id, account_id)

            if inventory_doc is None:
                print(f'No inventory match for resourceId={resource_id!r}, account_id={account_id!r}, skipping')
                continue

            merged_doc = merge_documents(account_doc, inventory_doc)
            output_coll.update_one({'_id': merged_doc['_id']}, {'$set': merged_doc}, upsert=True)
            print(f'Merged & wrote resourceId={resource_id!r}, account_id={account_id!r} -> {OUTPUT_COLLECTION}')


# Uncomment to start the real-time consumers (blocking; run in separate threads/processes as needed):
# inventory_thread = threading.Thread(target=watch_inventory_changes, args=(inventory_cache,), daemon=True)
# account_thread = threading.Thread(target=watch_account_changes, args=(inventory_cache,), daemon=True)
# inventory_thread.start()
# account_thread.start()
# print('Change stream consumers running. Interrupt to stop.')

# %% [markdown]
# ## 6. Check for duplicate (element_id, account_id) combinations
#
# Your inventory collection may contain more than one document for the same
# resource (same `element_id` + `account_id`) -- e.g. successive snapshots over time.
# The cache now automatically keeps only the newest one by `UPDATED_AT_FIELD`
# (see Section 3), so this is now informational rather than a source of silent data loss --
# but it's still useful to confirm `UPDATED_AT_FIELD` is populated on these duplicates.

# %%
dup_pipeline = [
    {'$group': {
        '_id': {'element_id': f'${ELEMENT_ID_FIELD}', 'account_id': f'${ACCOUNT_ID_FIELD_INVENTORY}'},
        'count': {'$sum': 1},
        'missing_updated_at': {
            '$sum': {'$cond': [{'$eq': [{'$type': f'${UPDATED_AT_FIELD}'}, 'missing']}, 1, 0]}
        },
    }},
    {'$match': {'count': {'$gt': 1}}},
    {'$sort': {'count': -1}},
]
duplicates = list(inventory_coll.aggregate(dup_pipeline))
print(f'{len(duplicates)} distinct (element_id, account_id) combination(s) have duplicates.')
for d in duplicates[:10]:
    flag = '  <-- some docs missing ' + UPDATED_AT_FIELD if d['missing_updated_at'] else ''
    print(f"  {d['_id']}  -> {d['count']} docs{flag}")
if len(duplicates) > 10:
    print(f'  ... and {len(duplicates) - 10} more')

# %% [markdown]
# ## 7. Recommended indexes
# For production performance, make sure these indexes exist so lookups and upserts
# stay fast even outside the in-memory cache. Both are now compound indexes matching
# the (resourceId, account_id) / (element_id, account_id) join.
#
# Not marked `unique=True` because duplicates (e.g. historical snapshots) exist in the
# source inventory data -- the pipeline resolves those in-memory via `UPDATED_AT_FIELD`
# rather than relying on a DB-level uniqueness constraint.

# %%
account_coll.create_index([(RESOURCE_ID_FIELD, 1), (ACCOUNT_ID_FIELD_ACCOUNT, 1)])
inventory_coll.create_index([(ELEMENT_ID_FIELD, 1), (ACCOUNT_ID_FIELD_INVENTORY, 1)])
inventory_coll.create_index(UPDATED_AT_FIELD)  # speeds up any future "latest per key" aggregation queries
output_coll.create_index('account_doc_id')
data_lake_coll.create_index([(ELEMENT_ID_FIELD, 1), (ACCOUNT_ID_FIELD_INVENTORY, 1), ('captured_at', -1)])
data_lake_coll.create_index('source_id')
print('Indexes ensured.')
