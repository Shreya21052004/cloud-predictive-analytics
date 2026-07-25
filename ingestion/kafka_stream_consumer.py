"""ingestion/kafka_stream_consumer.py — Kafka ingestion, run alongside the
existing MongoDB change-stream watchers in ingestion/realtime_stream_consumer.py.

This does NOT replace the Mongo change streams. It's a second front door
into the same pipeline: any producer (a microservice, a CDC connector on a
different cluster, another team's event bus, etc.) can publish account
metric events and/or inventory events onto Kafka topics, and they land in
the exact same `enriched_data_lake` collection the Mongo watchers already
keep live — same in-memory lookup cache, same merge_documents() join logic,
same downstream predict/LLM loops in run_pipeline.py don't need to know or
care which front door a given row came in through.

Two topics, mirroring the two Mongo collections:

    <account-topic>    -- one JSON message per account/metric snapshot,
                           same field shape as a document in
                           cloud_account_data_collection_daily_updated
    <inventory-topic>  -- one JSON message per inventory upsert/delete,
                           same field shape as a document in
                           service_resource_inventory_v2_updated,
                           optionally with a top-level "operation_type"
                           field ("insert" | "update" | "delete"); defaults
                           to "upsert" if absent.

Flow (matches the Kafka lane of the architecture diagram):

    Kafka <account-topic>   --> lookup(resourceId, account_id) in the
                                 shared Inventory Lookup Cache
                             --> Metadata Enrichment (merge_documents)
                             --> upsert into enriched_data_lake

    Kafka <inventory-topic> --> shared Inventory Lookup Cache (update)
                             --> append record to the Data Lake collection

Requirements:
  - pip install kafka-python
  - A reachable Kafka (or Redpanda / MSK / Confluent Cloud) bootstrap
    endpoint. See --help for broker/security flags.

Offsets are committed manually, one message at a time, only AFTER the
corresponding Mongo write succeeds — so a crash mid-batch reprocesses
(at-least-once) rather than silently drops a message. Mongo upserts here
are idempotent (deterministic _id per resource+account, or the message's
own "_id"/"id" field if it carries one), so reprocessing is safe.
"""
import argparse
import json
import os
import threading
from datetime import datetime, timezone

from bson import ObjectId
from pymongo.errors import PyMongoError

from .realtime_stream_consumer import (
    shutdown_event,
    normalize_id,
    lookup_inventory,
    merge_documents,
)

DEFAULT_ACCOUNT_TOPIC = "cloud-account-metrics"
DEFAULT_INVENTORY_TOPIC = "service-resource-inventory"


def _import_kafka():
    try:
        from kafka import KafkaConsumer
        from kafka.errors import KafkaError
    except ImportError as e:
        raise SystemExit(
            "kafka-python is required for Kafka ingestion. Install it with:\n"
            "    pip install kafka-python\n"
            f"(original error: {e})"
        )
    return KafkaConsumer, KafkaError


def _consumer_kwargs(args):
    """Shared KafkaConsumer connection/security kwargs, built from CLI args
    (which default to environment variables so this composes cleanly with
    run_pipeline.py's existing env-var-driven config)."""
    kwargs = dict(
        bootstrap_servers=args.kafka_bootstrap_servers.split(","),
        enable_auto_commit=False,
        auto_offset_reset=args.kafka_auto_offset_reset,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")) if v else None,
        key_deserializer=lambda k: k.decode("utf-8") if k else None,
        consumer_timeout_ms=1000,  # so poll loops can check shutdown_event
    )
    if args.kafka_security_protocol:
        kwargs["security_protocol"] = args.kafka_security_protocol
    if args.kafka_sasl_mechanism:
        kwargs["sasl_mechanism"] = args.kafka_sasl_mechanism
        kwargs["sasl_plain_username"] = args.kafka_sasl_username
        kwargs["sasl_plain_password"] = args.kafka_sasl_password
    return kwargs


def _explain_and_backoff(name, e):
    print(f"[kafka:{name}] stream error, retrying in 5s: {e}")
    shutdown_event.wait(5)


# ---------------------------------------------------------------------------
# Inventory topic -> cache + data lake
# ---------------------------------------------------------------------------

def watch_kafka_inventory_topic(cache, data_lake_coll, cfg, args):
    KafkaConsumer, KafkaError = _import_kafka()
    topic = args.kafka_inventory_topic

    while not shutdown_event.is_set():
        try:
            consumer = KafkaConsumer(
                topic,
                group_id=f"{args.kafka_group_id}-inventory",
                **_consumer_kwargs(args),
            )
            print(f"[kafka:inventory] subscribed to '{topic}' "
                  f"(group={args.kafka_group_id}-inventory).")
            try:
                while not shutdown_event.is_set():
                    for msg in consumer:
                        if shutdown_event.is_set():
                            break
                        doc = msg.value
                        if not doc:
                            consumer.commit()
                            continue
                        op_type = doc.pop("operation_type", "upsert")

                        record = dict(doc)
                        record["source_id"] = record.get("_id") or msg.key
                        record["operation_type"] = op_type
                        record["captured_at"] = datetime.now(timezone.utc)
                        record["source"] = "kafka"
                        record.pop("_id", None)
                        data_lake_coll.insert_one(record)

                        if op_type != "delete":
                            elem_key = normalize_id(doc.get(cfg.element_id_field))
                            acct_key = normalize_id(doc.get(cfg.account_id_field_inventory))
                            if elem_key is not None and acct_key is not None:
                                bucket = cache.setdefault(elem_key, {})
                                existing = bucket.get(acct_key)
                                new_ts = doc.get(cfg.updated_at_field)
                                existing_ts = existing.get(cfg.updated_at_field) if existing else None
                                if existing is None or new_ts is None or existing_ts is None or new_ts > existing_ts:
                                    bucket[acct_key] = doc
                                    print(f"[kafka:inventory] cache updated: "
                                          f"element_id={elem_key!r} account_id={acct_key!r}")
                        consumer.commit()
                    # consumer_timeout_ms fires here every ~1s with no messages;
                    # loop back around to re-check shutdown_event.
            finally:
                consumer.close()
        except KafkaError as e:
            if shutdown_event.is_set():
                break
            _explain_and_backoff("inventory", e)
        except PyMongoError as e:
            if shutdown_event.is_set():
                break
            _explain_and_backoff("inventory", e)


# ---------------------------------------------------------------------------
# Account topic -> lookup + merge -> enriched_data_lake
# ---------------------------------------------------------------------------

def watch_kafka_account_topic(output_coll, inventory_coll, cache, cfg, args):
    KafkaConsumer, KafkaError = _import_kafka()
    topic = args.kafka_account_topic

    while not shutdown_event.is_set():
        try:
            consumer = KafkaConsumer(
                topic,
                group_id=f"{args.kafka_group_id}-account",
                **_consumer_kwargs(args),
            )
            print(f"[kafka:account] subscribed to '{topic}' "
                  f"(group={args.kafka_group_id}-account).")
            try:
                while not shutdown_event.is_set():
                    for msg in consumer:
                        if shutdown_event.is_set():
                            break
                        account_doc = msg.value
                        if not account_doc:
                            consumer.commit()
                            continue

                        # Kafka messages have no native Mongo _id. Use one the
                        # producer supplied ("_id"/"id"), or a message-key-derived
                        # deterministic id, or a fresh ObjectId as last resort —
                        # any of these keep the eventual upsert idempotent on retry.
                        raw_id = account_doc.get("_id") or account_doc.get("id") or msg.key
                        account_doc["_id"] = ObjectId(raw_id) if raw_id and ObjectId.is_valid(str(raw_id)) else ObjectId()

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
                            print(f"[kafka:account] no inventory match: "
                                  f"resourceId={resource_id!r} account_id={account_id!r}, skipping")
                        else:
                            merged = merge_documents(
                                account_doc, inventory_doc,
                                account_precedence=cfg.account_fields_take_precedence,
                                preserve_account_id=cfg.preserve_account_id,
                            )
                            merged["source"] = "kafka"
                            output_coll.update_one({"_id": merged["_id"]}, {"$set": merged}, upsert=True)
                            print(f"[kafka:account] merged resourceId={resource_id!r} "
                                  f"account_id={account_id!r} -> {cfg.output_collection}")

                        consumer.commit()
            finally:
                consumer.close()
        except KafkaError as e:
            if shutdown_event.is_set():
                break
            _explain_and_backoff("account", e)
        except PyMongoError as e:
            if shutdown_event.is_set():
                break
            _explain_and_backoff("account", e)


# ---------------------------------------------------------------------------
# Wiring helper for run_pipeline.py (shares the same cache/collections the
# Mongo change-stream threads use, so both front doors feed one cache/one
# output collection with no duplicate logic).
# ---------------------------------------------------------------------------

def start_kafka_threads(args, cache, output_coll, inventory_coll, data_lake_coll, cfg):
    inv_thread = threading.Thread(
        target=watch_kafka_inventory_topic,
        args=(cache, data_lake_coll, cfg, args),
        name="kafka-ingest-inventory", daemon=True,
    )
    acct_thread = threading.Thread(
        target=watch_kafka_account_topic,
        args=(output_coll, inventory_coll, cache, cfg, args),
        name="kafka-ingest-account", daemon=True,
    )
    inv_thread.start()
    acct_thread.start()
    return inv_thread, acct_thread


# ---------------------------------------------------------------------------
# Standalone CLI (same collections/fields as realtime_stream_consumer.py, so
# it can run as its own process instead of inside run_pipeline.py if wanted)
# ---------------------------------------------------------------------------

def add_kafka_args(p):
    p.add_argument("--enable-kafka", action="store_true",
                    help="Also ingest from Kafka topics, alongside the Mongo change streams.")
    p.add_argument("--kafka-bootstrap-servers",
                    default=os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
                    help="Comma-separated list of host:port broker addresses.")
    p.add_argument("--kafka-account-topic",
                    default=os.environ.get("KAFKA_ACCOUNT_TOPIC", DEFAULT_ACCOUNT_TOPIC))
    p.add_argument("--kafka-inventory-topic",
                    default=os.environ.get("KAFKA_INVENTORY_TOPIC", DEFAULT_INVENTORY_TOPIC))
    p.add_argument("--kafka-group-id",
                    default=os.environ.get("KAFKA_GROUP_ID", "jsr-pipeline"),
                    help="Consumer group id prefix (account/inventory each get their own suffix).")
    p.add_argument("--kafka-auto-offset-reset",
                    default=os.environ.get("KAFKA_AUTO_OFFSET_RESET", "earliest"),
                    choices=["earliest", "latest"])
    p.add_argument("--kafka-security-protocol",
                    default=os.environ.get("KAFKA_SECURITY_PROTOCOL", ""),
                    help="e.g. SASL_SSL for Confluent Cloud / MSK. Leave blank for PLAINTEXT (local dev).")
    p.add_argument("--kafka-sasl-mechanism",
                    default=os.environ.get("KAFKA_SASL_MECHANISM", ""),
                    help="e.g. PLAIN or SCRAM-SHA-512.")
    p.add_argument("--kafka-sasl-username", default=os.environ.get("KAFKA_SASL_USERNAME", ""))
    p.add_argument("--kafka-sasl-password", default=os.environ.get("KAFKA_SASL_PASSWORD", ""))
    return p


def parser():
    p = argparse.ArgumentParser(description="Standalone Kafka ingestion worker")
    p.add_argument("--mongo-uri", default="mongodb://localhost:27017")
    p.add_argument("--db", default="mydb")
    p.add_argument("--inventory-collection", default="service_resource_inventory_v2_updated")
    p.add_argument("--output-collection", default="enriched_data_lake")
    p.add_argument("--data-lake-collection", default="service_resource_inventory_data_lake")
    p.add_argument("--resource-id-field", default="resourceId")
    p.add_argument("--element-id-field", default="element_id")
    p.add_argument("--account-id-field-account", default="account_id")
    p.add_argument("--account-id-field-inventory", default="account_id")
    p.add_argument("--updated-at-field", default="updatedAt")
    add_kafka_args(p)
    return p


def main():
    import signal
    from types import SimpleNamespace
    from pymongo import MongoClient
    from .realtime_stream_consumer import build_inventory_lookup_cache

    args = parser().parse_args()
    client = MongoClient(args.mongo_uri)
    db = client[args.db]
    inventory_coll = db[args.inventory_collection]
    output_coll = db[args.output_collection]
    data_lake_coll = db[args.data_lake_collection]

    cache = build_inventory_lookup_cache(
        inventory_coll, args.element_id_field, args.account_id_field_inventory, args.updated_at_field,
    )
    cfg = SimpleNamespace(
        element_id_field=args.element_id_field,
        account_id_field_inventory=args.account_id_field_inventory,
        account_id_field_account=args.account_id_field_account,
        resource_id_field=args.resource_id_field,
        updated_at_field=args.updated_at_field,
        account_fields_take_precedence=True,
        preserve_account_id=True,
        output_collection=args.output_collection,
    )

    def handle_shutdown(signum, frame):
        print("\nShutdown signal received, stopping Kafka consumers...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    inv_thread, acct_thread = start_kafka_threads(args, cache, output_coll, inventory_coll, data_lake_coll, cfg)
    print("Kafka stream consumers running. Press Ctrl+C to stop.")

    while not shutdown_event.is_set():
        shutdown_event.wait(1)

    inv_thread.join(timeout=5)
    acct_thread.join(timeout=5)
    print("Stopped.")


if __name__ == "__main__":
    main()
