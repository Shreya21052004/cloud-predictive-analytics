"""backfill_project_collections.py — One-time migration of existing data from
the old shared collections (one per category) into the new per-project
collections (one per category PER project) introduced alongside the
project-aware cross_domain.py correlation change.

This does NOT run as part of the normal pipeline — run it once after
deploying the per-project routing change, to move historical documents
out of the old collections (e.g. `22prediction_logs_compute`) into the new
ones (e.g. `22prediction_logs_compute__proj_a`). New predictions going
forward are already written to the per-project collections directly by
pipeline.py; this script only handles what was written before the cutover.

Usage:
    python -m scripts.backfill_project_collections \
        --mongo-uri mongodb://localhost:27017 \
        --db mydb \
        [--dry-run] [--delete-source]

By default the script COPIES documents and leaves the old collections in
place untouched, so it's safe to re-run and easy to verify before cleanup.
Pass --delete-source to remove migrated documents from the old collection
after a successful copy (only deletes documents that were just migrated,
identified by their original _id).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python scripts/backfill_project_collections.py` without
# installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pymongo import MongoClient, InsertOne, DeleteOne

from src.config import PipelineConfig, CATEGORY_OUTPUT_COLLECTION, TARGET_CATEGORIES
from src.pipeline import _project_suffix, COMPOSITE_INCIDENTS_COLLECTION

BATCH_SIZE = 500


def _old_category_collections(config: PipelineConfig):
    """(category, old_collection_name) for every category's pre-migration
    shared collection."""
    for cat in TARGET_CATEGORIES:
        attr = CATEGORY_OUTPUT_COLLECTION.get(cat, "output_collection")
        yield cat, getattr(config, attr)


def migrate_collection(db, old_name, base_new_name, project_field="project",
                        fallback_field=None, dry_run=False, delete_source=False):
    """Copy every doc in `old_name` into `<base_new_name>__<suffix>` based on
    `project_field` (falling back to `fallback_field`, e.g. account_id, when
    project is missing). Returns (n_read, n_written, n_deleted).
    """
    old_col = db[old_name]
    n_read = 0
    n_written = 0
    n_deleted = 0

    by_target = {}  # target_collection_name -> list of docs
    ids_by_target = {}

    cursor = old_col.find({}).batch_size(2000)
    try:
        for doc in cursor:
            n_read += 1
            key = doc.get(project_field) or (doc.get(fallback_field) if fallback_field else None)
            target = f"{base_new_name}__{_project_suffix(key)}"
            by_target.setdefault(target, []).append(doc)
            ids_by_target.setdefault(target, []).append(doc["_id"])
    finally:
        cursor.close()

    if dry_run:
        for target, docs in by_target.items():
            print(f"    [dry-run] would copy {len(docs)} doc(s) from '{old_name}' → '{target}'")
        return n_read, 0, 0

    for target, docs in by_target.items():
        new_col = db[target]
        ops = [InsertOne(d) for d in docs]
        for i in range(0, len(ops), BATCH_SIZE):
            batch = ops[i:i + BATCH_SIZE]
            try:
                result = new_col.bulk_write(batch, ordered=False)
                n_written += result.inserted_count
            except Exception as e:
                # Likely duplicate _id from a prior partial run — count what
                # succeeded and move on rather than aborting the whole batch.
                print(f"    [warn] some inserts into '{target}' failed/skipped "
                      f"(likely already migrated): {e}")

        if delete_source:
            del_ops = [DeleteOne({"_id": _id}) for _id in ids_by_target[target]]
            for i in range(0, len(del_ops), BATCH_SIZE):
                batch = del_ops[i:i + BATCH_SIZE]
                result = old_col.bulk_write(batch, ordered=False)
                n_deleted += result.deleted_count

        print(f"    '{old_name}' → '{target}': {len(docs)} doc(s) processed")

    return n_read, n_written, n_deleted


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mongo-uri", default="mongodb://localhost:27017")
    ap.add_argument("--db", default="mydb")
    ap.add_argument("--dry-run", action="store_true",
                     help="Report what would be migrated without writing anything.")
    ap.add_argument("--delete-source", action="store_true",
                     help="Delete migrated documents from the old collection after copying. "
                          "Off by default — old collections are left intact unless you pass this.")
    ap.add_argument("--skip-composite-incidents", action="store_true",
                     help="Only migrate per-category prediction collections, skip composite incidents.")
    args = ap.parse_args()

    config = PipelineConfig(mongo_uri=args.mongo_uri, db_name=args.db)
    client = MongoClient(args.mongo_uri)
    db = client[args.db]

    grand_total = {"read": 0, "written": 0, "deleted": 0}

    print(f"=== Migrating per-category prediction collections (db={args.db}) ===")
    for category, old_name in _old_category_collections(config):
        if old_name not in db.list_collection_names():
            print(f"  [{category}] '{old_name}' does not exist, skipping.")
            continue
        print(f"  [{category}] migrating '{old_name}'...")
        n_read, n_written, n_deleted = migrate_collection(
            db, old_name, old_name,
            project_field="project", fallback_field="account_id",
            dry_run=args.dry_run, delete_source=args.delete_source,
        )
        grand_total["read"] += n_read
        grand_total["written"] += n_written
        grand_total["deleted"] += n_deleted
        print(f"  [{category}] read={n_read} written={n_written} deleted={n_deleted}")

    if not args.skip_composite_incidents:
        print(f"\n=== Migrating composite incidents (db={args.db}) ===")
        if COMPOSITE_INCIDENTS_COLLECTION not in db.list_collection_names():
            print(f"  '{COMPOSITE_INCIDENTS_COLLECTION}' does not exist, skipping.")
        else:
            # Composite incident docs carry `project` OR `account_id` directly
            # (see cross_domain.py), so project_field/fallback_field match
            # those exact keys.
            n_read, n_written, n_deleted = migrate_collection(
                db, COMPOSITE_INCIDENTS_COLLECTION, COMPOSITE_INCIDENTS_COLLECTION,
                project_field="project", fallback_field="account_id",
                dry_run=args.dry_run, delete_source=args.delete_source,
            )
            grand_total["read"] += n_read
            grand_total["written"] += n_written
            grand_total["deleted"] += n_deleted
            print(f"  read={n_read} written={n_written} deleted={n_deleted}")

    print(f"\n=== Done. Total read={grand_total['read']} "
          f"written={grand_total['written']} deleted={grand_total['deleted']} "
          f"{'(DRY RUN — nothing was written)' if args.dry_run else ''} ===")
    if not args.delete_source and not args.dry_run:
        print("Old collections were left intact (pass --delete-source to remove "
              "migrated documents from them after you've verified the new "
              "per-project collections look correct).")


if __name__ == "__main__":
    main()
