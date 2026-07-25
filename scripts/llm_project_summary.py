"""scripts/llm_project_summary.py — scheduled job (cron every 1-2h).

Reads predictions produced by `pipeline.py predict` across ALL category
collections (Compute/Storage/Network/Container/Databases) for a fixed
lookback window, groups them by `project`, and writes one consolidated
LLM summary per project to the output DB.

This is intentionally separate from `pipeline.py predict` — categories
write to separate collections at separate times, so a per-project view
has to be assembled after the fact, on its own schedule, not inline in
a single category's prediction run.

Usage:
    python -m scripts.llm_project_summary \\
        --mongo-uri mongodb://localhost:27017 \\
        --db mydb --output-db mydb \\
        --window-minutes 90 \\
        --provider ollama --model qwen3:8b

Requires a local Ollama server (`ollama serve`) with the model pulled
(`ollama pull qwen3:8b`) — no API key needed. Set OLLAMA_HOST if it's not
running on localhost:11434. Pass --provider anthropic with ANTHROPIC_API_KEY
to use a cloud model instead.
"""
import argparse
import hashlib
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Allow running as `python scripts/llm_project_summary.py` without installing
# the package — matches the convention used by scripts/backfill_project_collections.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pymongo import UpdateOne

from src.config import CATEGORY_OUTPUT_COLLECTION, TARGET_CATEGORIES
from src.mongo_io import get_collection, _bson_safe
from src.llm_summary import summarize_all_projects, LLM_SUMMARY_COLLECTION, DEFAULT_MODEL, DEFAULT_PROVIDER


def write_project_summaries(mongo_uri, db_name, collection_name, summaries, window_minutes):
    """Upsert project summary docs, idempotent per (project, window bucket).

    NOT reusing mongo_io.write_predictions here on purpose: it runs
    validate_predictions(), which enforces the per-resource prediction
    schema (resource_id, prediction_type, etc.) that these project-level
    summary docs don't have and shouldn't be forced into.
    """
    if not summaries:
        return 0
    collection = get_collection(mongo_uri, db_name, collection_name)
    ops = []
    for s in summaries:
        safe = _bson_safe(s)
        # Bucket the idempotency key by window so re-running the job for
        # the same window updates the same doc instead of duplicating it,
        # while a later window still produces a new doc.
        window_bucket = safe["generated_at"][:13]  # hour-granularity bucket
        raw = f"{safe.get('project')}::{window_bucket}::{window_minutes}"
        safe["_idempotent_key"] = hashlib.sha256(raw.encode()).hexdigest()[:24]
        ops.append(UpdateOne(
            {"_idempotent_key": safe["_idempotent_key"]},
            {"$set": safe},
            upsert=True,
        ))
    result = collection.bulk_write(ops, ordered=False)
    return result.upserted_count + result.modified_count


def load_window_predictions(mongo_uri, db_name, window_minutes):
    """Pull predictions with prediction_timestamp inside the lookback
    window from every category's output collection.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(minutes=window_minutes)).isoformat()
    all_preds = []
    for category in TARGET_CATEGORIES:
        attr = CATEGORY_OUTPUT_COLLECTION.get(category)
        # attr is a config field name (e.g. "compute_output_collection");
        # scripts don't build a PipelineConfig here, so collection names
        # are passed explicitly instead — see parser() defaults below,
        # which mirror pipeline.py's --*-out defaults.
        col_name = COLLECTION_NAME_OVERRIDES.get(category)
        col = get_collection(mongo_uri, db_name, col_name)
        cursor = col.find({"prediction_timestamp": {"$gte": cutoff}})
        batch = list(cursor)
        print(f"  [{category}] {len(batch)} prediction(s) in last {window_minutes}min from {col_name}.")
        all_preds.extend(batch)
    return all_preds


def parser():
    p = argparse.ArgumentParser(description="Per-project LLM summarization job")
    p.add_argument("--mongo-uri", default="mongodb://localhost:27017")
    p.add_argument("--db", default="mydb")
    p.add_argument("--output-db", default=None,
                   help="Where to write project summaries; defaults to --db.")
    p.add_argument("--window-minutes", type=int, default=90,
                   help="Lookback window for predictions to aggregate (matches your "
                        "predict/feedback cadence — 60-120min is typical).")
    p.add_argument("--provider", default=DEFAULT_PROVIDER, choices=["ollama", "anthropic"],
                   help="LLM provider. 'ollama' (default) runs fully local, no API key. "
                        "'anthropic' is an opt-in cloud fallback needing ANTHROPIC_API_KEY.")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--max-tokens", type=int, default=4000,
                   help="Digest token budget per LLM call before map-reduce chunking kicks in.")
    p.add_argument("--compute-out", default="22prediction_logs_compute")
    p.add_argument("--out", default="22prediction_logs_storage")
    p.add_argument("--network-out", default="22prediction_logs_network")
    p.add_argument("--container-out", default="22prediction_logs_container")
    p.add_argument("--databases-out", default="22prediction_logs_databases")
    p.add_argument("--summary-collection", default=LLM_SUMMARY_COLLECTION)
    p.add_argument("--dry-run", action="store_true")
    return p


def main():
    global COLLECTION_NAME_OVERRIDES
    args = parser().parse_args()
    output_db = args.output_db or args.db

    COLLECTION_NAME_OVERRIDES = {
        "Compute":   args.compute_out,
        "Storage":   args.out,
        "Network":   args.network_out,
        "Container": args.container_out,
        "Databases": args.databases_out,
    }

    print(f"Pulling predictions from last {args.window_minutes} minute(s)...")
    predictions = load_window_predictions(args.mongo_uri, output_db, args.window_minutes)
    print(f"Total predictions in window: {len(predictions)}")

    if not predictions:
        print("Nothing to summarize.")
        return 0

    by_project_count = {}
    for p in predictions:
        by_project_count[p.get("project") or "UNASSIGNED"] = by_project_count.get(p.get("project") or "UNASSIGNED", 0) + 1
    print(f"Projects in window: {len(by_project_count)}")
    for project, n in sorted(by_project_count.items(), key=lambda kv: -kv[1])[:20]:
        print(f"  {project}: {n} prediction(s)")

    if args.dry_run:
        summaries = summarize_all_projects(
            predictions, model=args.model, max_tokens=args.max_tokens, provider=args.provider,
        )
        import json
        print(json.dumps(summaries[:3], indent=2, default=str))
        print(f"Dry run produced {len(summaries)} project summary(ies).")
        return 0

    written = [0]

    def _flush_one(project, summary):
        n = write_project_summaries(
            args.mongo_uri, output_db, args.summary_collection, [summary], args.window_minutes,
        )
        written[0] += n
        print(f"  wrote summary for project={project!r} "
              f"({summary.get('resource_count')} resources, risk={summary.get('risk_level')})")

    summarize_all_projects(
        predictions, model=args.model, max_tokens=args.max_tokens, provider=args.provider,
        on_project_done=_flush_one,
    )
    print(f"Wrote {written[0]} project summary(ies) -> {output_db}.{args.summary_collection}.")
    return 0


COLLECTION_NAME_OVERRIDES = {}

if __name__ == "__main__":
    raise SystemExit(main())
