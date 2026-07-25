"""run_pipeline.py — ONE command that runs the whole flowchart, continuously.

    MongoDB (Account Metrics Collection, Service Resource Inventory)
            |  change streams (Section 5 of ingestion/*)
            v
    Inventory Lookup Cache  --lookup(resourceId)-->  Metadata Enrichment
            |
            v
    enriched_data_lake (kept live by the change-stream threads)
            |
            v
    Existing Prediction Pipeline  (src/pipeline.py: train + predict, on a timer)
            |
            v
    Prediction + Alerts + Dashboard + MongoDB
            |
            v
    Local Ollama LLM project summaries (scripts/llm_project_summary.py, on a timer)

Everything above runs as background threads inside ONE process, started by
ONE command:

    python run_pipeline.py

Ctrl+C stops every thread cleanly. No manual step-by-step commands needed.

Configuration is via CLI flags or environment variables (see `--help` and
`.env.example`). The LLM step defaults to a local Ollama server (no API key,
no quota) — run `ollama serve` and `ollama pull qwen3:8b` first. Pass
--llm-provider anthropic with ANTHROPIC_API_KEY to use a cloud model instead.
"""
import argparse
import json
import os
import signal
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pymongo import MongoClient, UpdateOne
from pymongo.errors import PyMongoError

from ingestion.realtime_stream_consumer import (
    shutdown_event,
    normalize_id,
    build_inventory_lookup_cache,
    lookup_inventory,
    merge_documents,
    watch_inventory_changes,
    watch_account_changes,
)
from ingestion.kafka_stream_consumer import add_kafka_args, start_kafka_threads
from src.pipeline import make_config, cmd_train, cmd_predict, cmd_predict_incremental
from src.llm_summary import summarize_all_projects, LLM_SUMMARY_COLLECTION, DEFAULT_MODEL, DEFAULT_PROVIDER, OLLAMA_HOST
from scripts.llm_project_summary import load_window_predictions, write_project_summaries
from scripts.project_dashboard import ensure_project_indexes
from src.thresholds import evaluate_resource as evaluate_resource_thresholds
from src.llm_resource_explain import (
    explain_all_resources, write_resource_explanations,
    ensure_resource_explanation_indexes, RESOURCE_EXPLANATION_COLLECTION,
)


def log(tag, msg):
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] [{tag}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Thread 1: real-time ingestion / enrichment (change streams -> enriched_data_lake)
# ---------------------------------------------------------------------------

def backfill_enriched_data_lake(account_coll, output_coll, cache, args):
    """One-time batch pass so `enriched_data_lake` (and the data-lake history
    collection) aren't empty on a fresh setup. Change streams only capture
    documents that change AFTER the process starts; without this, a brand
    new deployment sits with zero rows until something happens to touch
    every existing account doc, which can be hours/days. Safe to re-run —
    every write is an upsert.
    """
    existing = output_coll.estimated_document_count()
    if existing > 0 and not args.force_backfill:
        log("backfill", f"{args.enriched_collection} already has {existing} doc(s); "
                         "skipping batch merge (pass --force-backfill to redo it).")
        return

    log("backfill", "enriched_data_lake looks empty — running one-time batch merge "
                     "over existing account docs (this can take a while on a large collection)...")
    matched, unmatched, ops = 0, 0, []
    cursor = account_coll.find()
    if args.backfill_limit:
        cursor = cursor.limit(args.backfill_limit)

    for account_doc in cursor:
        resource_id = normalize_id(account_doc.get(args.resource_id_field))
        account_id = normalize_id(account_doc.get(args.account_id_field_account))
        # inventory_doc = lookup_inventory(cache, resource_id, account_id)
        inventory_doc = lookup_inventory(
            cache, resource_id, account_id,
            inventory_coll=args._inventory_coll,   # see below
            key_field=args.element_id_field,
            account_field=args.account_id_field_inventory,
        )
        if inventory_doc is None:
            unmatched += 1
            continue
        merged = merge_documents(account_doc, inventory_doc)
        ops.append(UpdateOne({"_id": merged["_id"]}, {"$set": merged}, upsert=True))
        matched += 1
        if len(ops) >= 1000:
            output_coll.bulk_write(ops, ordered=False)
            ops.clear()
    if ops:
        output_coll.bulk_write(ops, ordered=False)

    log("backfill", f"batch merge done: matched={matched} unmatched={unmatched} "
                     f"-> {args.enriched_collection}.")
    if matched == 0 and unmatched > 0:
        log("backfill", "0 matched — every account doc failed the inventory lookup. "
                         "Check --resource-id-field/--element-id-field/--account-id-field-* "
                         "match your actual document field names.")


def seed_data_lake(inventory_coll, data_lake_coll, args):
    if data_lake_coll.estimated_document_count() > 0 and not args.force_backfill:
        return
    batch, total = [], 0
    for doc in inventory_coll.find().batch_size(2000):
        record = dict(doc)
        record.pop("_id", None)
        record["source_id"] = doc.get("_id")
        record["operation_type"] = "snapshot"
        record["captured_at"] = datetime.now(timezone.utc)
        batch.append(record)
        if len(batch) >= 2000:
            data_lake_coll.insert_many(batch, ordered=False)
            total += len(batch)
            batch.clear()
    if batch:
        data_lake_coll.insert_many(batch, ordered=False)
        total += len(batch)
    log("backfill", f"seeded {args.data_lake_collection} with {total} snapshot record(s).")


def start_ingestion_threads(args):
    client = MongoClient(args.mongo_uri)
    db = client[args.db]
    account_coll = db[args.account_collection]
    inventory_coll = db[args.inventory_collection]
    output_coll = db[args.enriched_collection]
    data_lake_coll = db[args.data_lake_collection]

    log("ingest", f"connected to {args.db}. account={args.account_collection} "
                  f"inventory={args.inventory_collection} -> {args.enriched_collection}")

    try:
        db.command({"collMod": args.inventory_collection,
                    "changeStreamPreAndPostImages": {"enabled": True}})
    except PyMongoError as e:
        log("ingest", f"could not enable pre/post images (needs MongoDB >= 6.0, replica set): {e}")

    log("ingest", "bootstrapping inventory lookup cache...")
    args._inventory_coll = inventory_coll
    cache = build_inventory_lookup_cache(
        inventory_coll, args.element_id_field, args.account_id_field_inventory, args.updated_at_field,
    )
    
    # backfill_enriched_data_lake(account_coll, output_coll, cache, args)
    if not args.skip_backfill:
        backfill_enriched_data_lake(account_coll, output_coll, cache, args)
        seed_data_lake(inventory_coll, data_lake_coll, args)

    cfg = SimpleNamespace(
        element_id_field=args.element_id_field,
        account_id_field_inventory=args.account_id_field_inventory,
        account_id_field_account=args.account_id_field_account,
        resource_id_field=args.resource_id_field,
        updated_at_field=args.updated_at_field,
        account_fields_take_precedence=True,
        preserve_account_id=True,
        output_collection=args.enriched_collection,
    )

    inv_thread = threading.Thread(
        target=watch_inventory_changes,
        args=(inventory_coll, data_lake_coll, cache, cfg, args.state_dir),
        name="ingest-inventory", daemon=True,
    )
    acct_thread = threading.Thread(
        target=watch_account_changes,
        args=(
            account_coll,
            output_coll,
            inventory_coll,
            cache,
            cfg,
            args.state_dir,
        ),
        name="ingest-account",
        daemon=True,
    )
    inv_thread.start()
    acct_thread.start()
    log("ingest", "change-stream consumers running (live).")

    threads = [inv_thread, acct_thread]
    if args.enable_kafka:
        log("ingest", f"Kafka ingestion enabled — bootstrap={args.kafka_bootstrap_servers} "
                       f"account_topic={args.kafka_account_topic} inventory_topic={args.kafka_inventory_topic} "
                       f"(feeding the same cache + {args.enriched_collection}).")
        kafka_inv_thread, kafka_acct_thread = start_kafka_threads(
            args, cache, output_coll, inventory_coll, data_lake_coll, cfg,
        )
        threads += [kafka_inv_thread, kafka_acct_thread]
        log("ingest", "Kafka stream consumers running (live).")
    return threads


# ---------------------------------------------------------------------------
# Thread 2: prediction loop — Existing Prediction Pipeline, on a timer
# ---------------------------------------------------------------------------

def _predict_cursor_path(args):
    return Path(args.state_dir) / "predict_cursor.json"


def _load_predict_cursor(args):
    path = _predict_cursor_path(args)
    if path.exists():
        try:
            return json.loads(path.read_text()).get("last_predicted_merged_at")
        except (json.JSONDecodeError, OSError):
            return None
    return None


def _save_predict_cursor(args, cursor_value):
    path = _predict_cursor_path(args)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"last_predicted_merged_at": cursor_value}))


def predict_loop(args):
    pipeline_args = SimpleNamespace(
        mongo_uri=args.mongo_uri, db=args.db, output_db=args.output_db,
        source=args.enriched_collection,
        out=args.storage_out, compute_out=args.compute_out, network_out=args.network_out,
        container_out=args.container_out, databases_out=args.databases_out,
        governance_out=args.governance_out, security_out=args.security_out,
        models_dir=args.models_dir, limit=args.predict_limit,
        # Pass through reference-time and freshness-max-age so static datasets
        # are not treated as fully stale relative to the wall clock.
        reference_time=getattr(args, "reference_time", None),
        freshness_max_age=getattr(args, "freshness_max_age", None),
    )
    config = make_config(pipeline_args)

    from src.models import load_models

    existing_models = load_models(args.models_dir)

    cycle = 0
    last_train = 0.0
    cursor = _load_predict_cursor(args)  # None on first-ever run
    while not shutdown_event.is_set():
        cycle += 1
        now = time.monotonic()
        cycle_started_at = datetime.now(timezone.utc).isoformat()
        try:
            # if now - last_train >= args.train_interval:
            #     log("predict", f"cycle {cycle}: (re)training models (full sweep)...")
            #     cmd_train(config, batch_size=args.train_batch_size)
            #     last_train = now
            # First startup
            if cycle == 1:
                if len(existing_models) == 5:
                    log("predict", f"cycle {cycle}: found existing models, skipping startup training.")
                    last_train = now
                else:
                    log("predict", f"cycle {cycle}: no models found, training...")
                    cmd_train(config, batch_size=args.train_batch_size)
                    last_train = now

            # Periodic retraining after startup
            elif now - last_train >= args.train_interval:
                log("predict", f"cycle {cycle}: retraining models...")
                cmd_train(config, batch_size=args.train_batch_size)
                last_train = now
            log("predict", f"cycle {cycle}: predicting resources changed since "
                           f"{cursor or 'the beginning'} (not the full {args.enriched_collection})...")
            n, new_cursor = cmd_predict_incremental(config, cursor, dry_run=False)
            # First run has no cursor to diff against, so cmd_predict_incremental
            # predicted everything once; from here on only diff forward.
            cursor = new_cursor if new_cursor is not None else cycle_started_at
            _save_predict_cursor(args, cursor)
            log("predict", f"cycle {cycle}: wrote {n} prediction(s); cursor now {cursor}.")
        except Exception as e:
            log("predict", f"cycle {cycle} failed: {e}")
            traceback.print_exc()
        shutdown_event.wait(args.predict_interval)


# ---------------------------------------------------------------------------
# Thread 3: LLM project-summary loop (local Ollama by default), on a timer
# ---------------------------------------------------------------------------

def summary_loop(args):
    output_db = args.output_db or args.db
    global_overrides = {
        "Compute":   args.compute_out,
        "Storage":   args.storage_out,
        "Network":   args.network_out,
        "Container": args.container_out,
        "Databases": args.databases_out,
    }
    import scripts.llm_project_summary as llm_script
    cycle = 0
    while not shutdown_event.is_set():
        cycle += 1
        llm_script.COLLECTION_NAME_OVERRIDES = global_overrides
        try:
            log("llm", f"cycle {cycle}: pulling last {args.summary_window_minutes}min of predictions...")
            predictions = load_window_predictions(args.mongo_uri, output_db, args.summary_window_minutes)
            if not predictions:
                log("llm", f"cycle {cycle}: nothing to summarize yet.")
            else:
                by_project = {}
                for p in predictions:
                    by_project[p.get("project") or "UNASSIGNED"] = by_project.get(
                        p.get("project") or "UNASSIGNED", 0) + 1
                log("llm", f"cycle {cycle}: {len(predictions)} prediction(s) across "
                           f"{len(by_project)} project(s); summarizing one at a time...")

                written_total = [0]

                def _flush_one(project, summary):
                    # Write+log as soon as each project is done, so a slow or
                    # rate-limited project later in the list can't hold back
                    # results that already finished -- and so the UI/Mongo
                    # collection shows progress mid-cycle instead of nothing
                    # until the entire window's projects are all done.
                    try:
                        n = write_project_summaries(
                            args.mongo_uri, output_db, LLM_SUMMARY_COLLECTION, [summary],
                            args.summary_window_minutes,
                        )
                        written_total[0] += n
                        log("llm", f"cycle {cycle}: wrote summary for project={project!r} "
                                   f"({summary.get('resource_count')} resources, "
                                   f"risk={summary.get('risk_level')}) -> "
                                   f"{output_db}.{LLM_SUMMARY_COLLECTION}.")
                    except Exception as e:
                        log("llm", f"cycle {cycle}: failed writing summary for "
                                   f"project={project!r}: {e}")

                summarize_all_projects(
                    predictions, model=args.llm_model, max_tokens=args.llm_max_tokens,
                    provider=args.llm_provider, on_project_done=_flush_one,
                )
                log("llm", f"cycle {cycle}: done — wrote {written_total[0]} project summary(ies) "
                           f"({len(predictions)} predictions, {args.llm_provider}/{args.llm_model}) "
                           f"-> {output_db}.{LLM_SUMMARY_COLLECTION}.")
        except Exception as e:
            log("llm", f"cycle {cycle} failed: {e}")
            traceback.print_exc()
        shutdown_event.wait(args.summary_interval)


# ---------------------------------------------------------------------------
# Thread 4: per-resource threshold breach tracking + LLM explanation loop
# ---------------------------------------------------------------------------

def resource_explain_loop(args):
    """For every resource with a fresh prediction in the window: check it
    against any user-defined thresholds (src/thresholds.py) to update
    breach count + predicted-next-breach, then get a full-metric LLM
    explanation for it (src/llm_resource_explain.py), written incrementally
    (same reasoning as summary_loop's per-project flush -- see that
    docstring/comment; the same all-or-nothing risk applies here, at
    resource scale instead of project scale, so it matters even more).

    --resource-explain-mode:
      priority (default) — only call the LLM for resources that are
        currently alerting, anomalous, or breaching a threshold. Explaining
        every resource on every cycle multiplies the per-project LLM call
        volume by (resource count / project count), which regresses right
        back into the rate-limit-driven stall the LLM summary step already
        hit once. "priority" keeps the expensive step scoped to what
        actually needs an engineer's attention.
      all — explain every resource in the window, no filtering. Do the
        math on your provider's rate limits first (or use --llm-provider
        anthropic for this step, or widen --resource-explain-interval).
    """
    output_db = args.output_db or args.db
    global_overrides = {
        "Compute":   args.compute_out,
        "Storage":   args.storage_out,
        "Network":   args.network_out,
        "Container": args.container_out,
        "Databases": args.databases_out,
    }
    import scripts.llm_project_summary as llm_script
    cycle = 0
    while not shutdown_event.is_set():
        cycle += 1
        llm_script.COLLECTION_NAME_OVERRIDES = global_overrides
        try:
            predictions = load_window_predictions(args.mongo_uri, output_db, args.resource_explain_window_minutes)
            if not predictions:
                log("resource-llm", f"cycle {cycle}: nothing to explain yet.")
            else:
                # threshold evaluation runs over every resource in the window
                # regardless of mode -- breach counting has to stay complete
                # even for resources that don't get an LLM explanation this
                # cycle, or the count/ETA drifts from reality.
                breach_states_by_resource = {}
                latest_by_resource = {}
                for p in predictions:
                    rid = p.get("resource_id")
                    if not rid:
                        continue
                    ts = p.get("prediction_timestamp") or ""
                    if rid not in latest_by_resource or ts > (latest_by_resource[rid].get("prediction_timestamp") or ""):
                        latest_by_resource[rid] = p
                for rid, p in latest_by_resource.items():
                    try:
                        states = evaluate_resource_thresholds(args.mongo_uri, output_db, p)
                        if states:
                            breach_states_by_resource[rid] = states
                    except Exception as e:
                        log("resource-llm", f"cycle {cycle}: threshold check failed for {rid}: {e}")

                if args.resource_explain_mode == "priority":
                    to_explain = [
                        p for rid, p in latest_by_resource.items()
                        if p.get("is_anomalous")
                        or (p.get("alert") or {}).get("severity", "INFO") != "INFO"
                        or breach_states_by_resource.get(rid)
                    ]
                else:
                    to_explain = list(latest_by_resource.values())

                if not to_explain:
                    log("resource-llm", f"cycle {cycle}: {len(latest_by_resource)} resource(s) in window, "
                                         f"none met --resource-explain-mode={args.resource_explain_mode} criteria.")
                else:
                    log("resource-llm", f"cycle {cycle}: explaining {len(to_explain)}/{len(latest_by_resource)} "
                                         f"resource(s) (mode={args.resource_explain_mode})...")
                    written_total = [0]

                    def _flush_one(resource_id, explanation):
                        try:
                            n = write_resource_explanations(args.mongo_uri, output_db, [explanation])
                            written_total[0] += n
                            log("resource-llm", f"cycle {cycle}: wrote explanation for "
                                                 f"resource_id={resource_id!r} (risk={explanation.get('risk_level')}) "
                                                 f"-> {output_db}.{RESOURCE_EXPLANATION_COLLECTION}.")
                        except Exception as e:
                            log("resource-llm", f"cycle {cycle}: failed writing explanation for "
                                                 f"resource_id={resource_id!r}: {e}")

                    explain_all_resources(
                        to_explain, breach_states_by_resource=breach_states_by_resource,
                        model=args.resource_explain_model, max_tokens=args.resource_explain_max_tokens,
                        provider=args.llm_provider, on_resource_done=_flush_one,
                    )
                    log("resource-llm", f"cycle {cycle}: done — wrote {written_total[0]} resource "
                                         f"explanation(s) -> {output_db}.{RESOURCE_EXPLANATION_COLLECTION}.")
        except Exception as e:
            log("resource-llm", f"cycle {cycle} failed: {e}")
            traceback.print_exc()
        shutdown_event.wait(args.resource_explain_interval)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def parser():
    p = argparse.ArgumentParser(
        description="Run the whole real-time pipeline (ingestion + predict + LLM summaries) as one process.",
    )
    # Mongo / collections
    p.add_argument("--mongo-uri", default=os.environ.get("MONGO_URI", "mongodb://localhost:27017"))
    p.add_argument("--db", default=os.environ.get("MONGO_DB", "mydb"))
    p.add_argument("--output-db", default=os.environ.get("MONGO_OUTPUT_DB"))
    p.add_argument("--account-collection", default="cloud_account_data_collection_daily_updated")
    p.add_argument("--inventory-collection", default="service_resource_inventory_v2_updated")
    p.add_argument("--enriched-collection", default="enriched_data_lake")
    p.add_argument("--data-lake-collection", default="service_resource_inventory_data_lake")
    p.add_argument("--resource-id-field", default="resourceId")
    p.add_argument("--element-id-field", default="element_id")
    p.add_argument("--account-id-field-account", default="account_id")
    p.add_argument("--account-id-field-inventory", default="account_id")
    p.add_argument("--updated-at-field", default="updatedAt")
    p.add_argument("--state-dir", default=".stream_state")
    p.add_argument("--skip-backfill", action="store_true",
                    help="Don't run the one-time batch merge of existing account/inventory "
                         "docs into enriched_data_lake on startup (only new changes will flow in).")
    p.add_argument("--force-backfill", action="store_true",
                    help="Re-run the batch merge/data-lake seed even if the target collections "
                         "already have documents.")
    p.add_argument("--backfill-limit", type=int, default=0,
                    help="Cap on account docs processed during the initial batch merge (0 = all).")

    # Kafka (optional second ingestion front door, alongside the Mongo change streams)
    add_kafka_args(p)

    # Prediction output collections (same defaults as src/pipeline.py)
    p.add_argument("--storage-out", default="22prediction_logs_storage")
    p.add_argument("--compute-out", default="22prediction_logs_compute")
    p.add_argument("--network-out", default="22prediction_logs_network")
    p.add_argument("--container-out", default="22prediction_logs_container")
    p.add_argument("--databases-out", default="22prediction_logs_databases")
    p.add_argument("--governance-out", default="22prediction_logs_governance")
    p.add_argument("--security-out", default="22prediction_logs_security")
    p.add_argument("--models-dir", default="models")
    p.add_argument("--predict-limit", type=int, default=50000)

    # Timers (seconds)
    p.add_argument("--predict-interval", type=int, default=300,
                    help="How often to run the prediction pipeline (default 5 min).")
    p.add_argument("--train-interval", type=int, default=3600,
                    help="How often to retrain models before predicting (default 1h).")
    p.add_argument("--train-batch-size", type=int, default=20000,
                    help="Mongo cursor batch size during training (memory-bounded "
                         "streaming load — see src/pipeline.py load_normalized_streaming).")
    p.add_argument("--summary-interval", type=int, default=5400,
                    help="How often to run the LLM project-summary pass (default 90 min).")
    p.add_argument("--summary-window-minutes", type=int, default=90,
                    help="Lookback window for the LLM summary pass.")

    # LLM
    p.add_argument("--llm-provider", default=DEFAULT_PROVIDER, choices=["ollama", "anthropic"],
                    help="'ollama' (default) runs fully local via OLLAMA_HOST, no API key. "
                         "'anthropic' is an opt-in cloud fallback needing ANTHROPIC_API_KEY.")
    p.add_argument("--llm-model", default=DEFAULT_MODEL)
    p.add_argument("--llm-max-tokens", type=int, default=4000)

    # Per-resource threshold breach tracking + LLM explanation
    p.add_argument("--resource-explain-interval", type=int, default=300,
                    help="How often to run the per-resource threshold-check + LLM-explanation pass (default 5 min).")
    p.add_argument("--resource-explain-window-minutes", type=int, default=10,
                    help="Lookback window of predictions to consider for per-resource explanation.")
    p.add_argument("--resource-explain-mode", default="priority", choices=["priority", "all"],
                    help="'priority' (default) only explains resources that are anomalous, alerting, or "
                         "breaching a threshold. 'all' explains every resource in the window every cycle "
                         "-- much higher LLM call volume, check your provider's rate limits first.")
    p.add_argument("--resource-explain-model", default=None,
                    help="Model for per-resource explanations; defaults to --llm-model if unset.")
    p.add_argument("--resource-explain-max-tokens", type=int, default=1200)
    p.add_argument("--skip-resource-explain", action="store_true",
                    help="Don't run the per-resource threshold/explanation thread at all.")

    p.add_argument("--skip-indexes", action="store_true",
                    help="Skip creating the (project, timestamp) indexes on startup.")

    # Static / historical dataset support
    p.add_argument("--reference-time", dest="reference_time", default=None,
                    metavar="DATE_OR_AUTO",
                    help="Anchor 'now' for staleness evaluation. Use 'auto' to detect "
                         "max(to_date) from the enriched collection (recommended for static "
                         "datasets). Use 'YYYY-MM-DD' for an explicit date. Omit for real-time "
                         "streaming data (defaults to wall-clock time).")
    p.add_argument("--freshness-max-age", dest="freshness_max_age", type=int, default=None,
                    metavar="MINUTES",
                    help="Override the freshness gate in minutes. Auto-detected from data "
                         "granularity when --reference-time is set.")
    return p


def main():
    args = parser().parse_args()
    if not args.resource_explain_model:
        args.resource_explain_model = args.llm_model

    if not args.skip_indexes:
        try:
            ensure_project_indexes(args.mongo_uri, args.output_db or args.db)
            ensure_resource_explanation_indexes(args.mongo_uri, args.output_db or args.db)
            log("startup", "project + resource-explanation indexes ensured.")
        except Exception as e:
            log("startup", f"could not ensure indexes (continuing anyway): {e}")

    if args.llm_provider == "ollama":
        try:
            import requests
            requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3).raise_for_status()
        except Exception as e:
            log("startup", f"WARNING: could not reach Ollama at {OLLAMA_HOST} ({e}). "
                            f"Start it with `ollama serve` and `ollama pull {args.llm_model}` "
                            f"before the LLM summary step runs.")
    if args.llm_provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        log("startup", "WARNING: ANTHROPIC_API_KEY not set — the LLM summary step will fail.")

    def handle_shutdown(signum, frame):
        log("startup", "shutdown signal received, stopping all threads...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    ingestion_threads = start_ingestion_threads(args)

    predict_thread = threading.Thread(target=predict_loop, args=(args,), name="predict-loop", daemon=True)
    summary_thread = threading.Thread(target=summary_loop, args=(args,), name="summary-loop", daemon=True)
    predict_thread.start()
    summary_thread.start()

    extra_threads = ()
    if not args.skip_resource_explain:
        resource_thread = threading.Thread(
            target=resource_explain_loop, args=(args,), name="resource-explain-loop", daemon=True,
        )
        resource_thread.start()
        extra_threads = (resource_thread,)
        log("startup", f"pipeline running end-to-end. predict every {args.predict_interval}s "
                       f"(train every {args.train_interval}s), LLM project summaries every "
                       f"{args.summary_interval}s, per-resource threshold/explanation every "
                       f"{args.resource_explain_interval}s (mode={args.resource_explain_mode}) "
                       f"via {args.llm_provider}. Press Ctrl+C to stop.")
    else:
        log("startup", f"pipeline running end-to-end. predict every {args.predict_interval}s "
                       f"(train every {args.train_interval}s), LLM summaries every "
                       f"{args.summary_interval}s via {args.llm_provider}/{args.llm_model}. "
                       f"Per-resource explanation thread skipped (--skip-resource-explain). "
                       f"Press Ctrl+C to stop.")

    while not shutdown_event.is_set():
        shutdown_event.wait(1)

    for t in (*ingestion_threads, predict_thread, summary_thread, *extra_threads):
        t.join(timeout=5)
    log("startup", "stopped.")


if __name__ == "__main__":
    main()
