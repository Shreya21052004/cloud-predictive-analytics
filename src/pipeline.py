"""pipeline.py — Orchestrator (v6).

Changes from v5:
- 5 categories supported: Compute, Storage, Network, Container, Databases.
  (Governance and Security are intentionally excluded — see config.TARGET_CATEGORIES.)
- Deduplication of raw documents before normalization.
- Freshness gate and minimum history guard applied before prediction.
- Idempotent writes via mongo_io.write_predictions (upsert on idempotent_key).
- Model version tagging on every prediction (from models.py MODEL_VERSION).
- Drift detection / retraining trigger logged after each predict run.
- Feedback loop staleness check in cmd_feedback.
- Alert state-change dedup collection: 22alert_states.
- Dynamic output collection routing per category (CATEGORY_OUTPUT_COLLECTION).

v7: baseline profiles and the anomaly IsolationForest are pooled by
canonical_resource_type (component) instead of per resource_id — see
profiler.build_all_profiles and anomaly.train_type_isolation_forests. A
resource is only ever scored against its type's shared model; it never
trains/profiles anything from its own history alone.
"""
import argparse
from collections import Counter, defaultdict
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from .config import (
    PipelineConfig, TARGET_CATEGORIES, RAW_CATEGORY_FILTER,
    CATEGORY_OUTPUT_COLLECTION, PIPELINE_VERSION,
)
from .data_quality import (
    dedup_documents,
    dedup_documents_streaming,
    stale_resources,
    resource_history_hours,
    has_sufficient_history,
    check_drift_trigger,
    check_feedback_staleness,
)
from .feedback import run_feedback_pass, compute_performance_metrics, export_training_labels
from .metric_registry import REGISTRY
from .models import load_models, train_models
from .mongo_io import get_collection, load_documents, iter_document_batches, write_predictions
from .normalizer import (
    normalize_category, normalize_documents, normalize_provider,
    _accumulate_documents, _finalize_normalized,
)
from .prediction import generate_predictions
from .profiler import build_all_profiles, upsert_profile
from .validation import validate_normalized
from .explainability import attach_explanations
from .maintenance_windows import load_active_windows, apply_maintenance_suppression
from .cross_domain import detect_composite_incidents
from .anomaly_clustering import cluster_anomalies

PROFILES_COLLECTION  = "22resource_type_profiles"
ALERT_STATE_COLLECTION = "22alert_states"
MAINTENANCE_WINDOWS_COLLECTION = "22maintenance_windows"
COMPOSITE_INCIDENTS_COLLECTION = "22composite_incidents"
FLEET_ANOMALIES_COLLECTION     = "22fleet_anomalies"


def _detect_data_horizon(mongo_uri: str, db_name: str, source_collection: str) -> datetime:
    """Return max(to_date) from the source collection.

    Used when --reference-time auto is passed so the pipeline evaluates
    staleness relative to the latest data point rather than wall-clock time.
    This is the correct behaviour for static / historical datasets; for
    real-time streaming data the two are identical anyway.
    """
    col = get_collection(mongo_uri, db_name, source_collection)
    pipeline = [{"$group": {"_id": None, "max_to": {"$max": "$to_date"}}}]
    result = list(col.aggregate(pipeline, allowDiskUse=True))
    if result and result[0].get("max_to"):
        ts = result[0]["max_to"]
        if hasattr(ts, "tzinfo"):
            return ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts
        return datetime.fromisoformat(str(ts)).replace(tzinfo=timezone.utc)
    # Fallback: wall clock
    return datetime.now(timezone.utc)


def _parse_reference_time(value: str, mongo_uri: str, db_name: str, source_collection: str) -> datetime:
    """Parse the --reference-time CLI value.

    Accepts:
      auto           → detect from max(to_date) in source collection
      YYYY-MM-DD     → midnight UTC on that date
      YYYY-MM-DDTHH:MM:SS  → explicit datetime (UTC assumed if no tz)
    """
    if value.lower() == "auto":
        ts = _detect_data_horizon(mongo_uri, db_name, source_collection)
        print(f"  [reference-time] auto-detected data horizon: {ts.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        return ts
    try:
        ts = datetime.fromisoformat(value)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        print(f"  [reference-time] using explicit reference time: {ts.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        return ts
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--reference-time must be 'auto', 'YYYY-MM-DD', or 'YYYY-MM-DDTHH:MM:SS'. Got: {value!r}"
        )


def _detect_data_granularity_minutes(mongo_uri: str, db_name: str, source_collection: str) -> int:
    """Estimate the typical inter-sample gap in minutes by checking the median
    span of individual (from_date → to_date) windows in the source collection.

    Daily data (to - from ≈ 1440 min) → returns 1440.
    Hourly data → returns 60.  Used to set a sensible freshness gate.
    Falls back to 1440 (daily) if the collection is empty or fields missing.
    """
    col = get_collection(mongo_uri, db_name, source_collection)
    sample = list(col.find(
        {"from_date": {"$exists": True}, "to_date": {"$exists": True}},
        {"from_date": 1, "to_date": 1}
    ).limit(200))
    if not sample:
        return 1440
    spans = []
    for doc in sample:
        fd = doc.get("from_date")
        td = doc.get("to_date")
        if fd and td:
            try:
                diff = (td - fd).total_seconds() / 60.0
                if 0 < diff < 10080:  # ignore anything >7 days (bad data)
                    spans.append(diff)
            except Exception:
                pass
    if not spans:
        return 1440
    spans.sort()
    median_span = spans[len(spans) // 2]
    # Round up to nearest 60-min boundary, add 50% buffer for late delivery
    return max(60, int(median_span * 1.5))


def make_config(args):
    ref_time = None
    if getattr(args, "reference_time", None):
        ref_time = _parse_reference_time(
            args.reference_time,
            args.mongo_uri, args.db, args.source,
        )

    # When reference_time is set (static/historical dataset), freshness gate
    # must match the data's own granularity — not the 60-min real-time default.
    # Auto-detect the inter-sample period and use 1.5× as the staleness window,
    # unless the user has explicitly set --freshness-max-age.
    explicit_freshness = getattr(args, "freshness_max_age", None)
    if explicit_freshness is not None:
        freshness_minutes = int(explicit_freshness)
    elif ref_time is not None:
        freshness_minutes = _detect_data_granularity_minutes(args.mongo_uri, args.db, args.source)
        print(f"  [freshness] auto-set freshness_max_age_minutes={freshness_minutes} "
              f"(detected data granularity × 1.5)")
    else:
        freshness_minutes = 60  # real-time default

    return PipelineConfig(
        mongo_uri=args.mongo_uri,
        db_name=args.db,
        output_db_name=args.output_db or "",
        source_collection=args.source,
        output_collection=args.out,
        compute_output_collection=args.compute_out,
        network_output_collection=args.network_out,
        container_output_collection=args.container_out,
        databases_output_collection=args.databases_out,
        governance_output_collection=args.governance_out,
        security_output_collection=args.security_out,
        models_dir=Path(args.models_dir),
        limit=args.limit,
        reference_time=ref_time,
        freshness_max_age_minutes=freshness_minutes,
    )


def _collection_for(config, category, project=None):
    """Return the MongoDB collection name for a given category.

    All resources in a category write to one shared collection regardless of
    project — e.g. every Compute prediction goes to `22prediction_logs_compute`.
    `project` is accepted but no longer used; kept so callers don't need to
    change their call signature.
    """
    attr = CATEGORY_OUTPUT_COLLECTION.get(category, "output_collection")
    return getattr(config, attr)


def load_normalized(config):
    docs = load_documents(
        config.mongo_uri, config.db_name, config.source_collection,
        limit=config.limit, categories=RAW_CATEGORY_FILTER,
    )
    # --- Deduplication ---
    docs, n_dropped = dedup_documents(docs)
    if n_dropped:
        print(f"  [dedup] Dropped {n_dropped} duplicate raw documents.")

    df = normalize_documents(docs)
    limit_msg = "all documents" if config.limit == 0 else f"up to {config.limit} documents"
    print(
        f"Loaded {len(docs)} Mongo document(s) (after dedup), normalized to {len(df)} "
        f"resource-time row(s) from {limit_msg}. "
        f"Categories: {', '.join(TARGET_CATEGORIES)} "
        f"(raw query matched: {', '.join(RAW_CATEGORY_FILTER)})."
    )
    return df


def load_normalized_streaming(config, batch_size=20000, log_every=5):
    """Same result as load_normalized(), but reads the source collection
    through iter_document_batches() so at most one batch of raw documents
    is ever in memory at once, instead of the whole collection.

    Dedup and normalization still cover every document exactly as before —
    dedup_documents_streaming() shares one hash set across all batches (a
    set of digest strings, far smaller than the documents themselves), and
    _accumulate_documents() folds each batch into the same running
    (grouped, raw_metrics, resource_tags) accumulators that normalize_documents()
    would have built in one shot. _finalize_normalized() then runs once at
    the end, over the complete accumulated set — so sorting and gap-flag
    computation (which need a resource's whole timeline) are unaffected by
    batch boundaries and produce byte-identical output to load_normalized().

    Only the raw per-document Mongo payloads are ever discarded early; the
    accumulators end up the same size either way.

    `log_every`: print a progress line every N batches (and always on the
    first batch), so a long full-collection load isn't silent for 10+
    minutes with no visible sign it's still working.
    """
    grouped, raw_metrics, resource_tags = {}, defaultdict(dict), {}
    seen_hashes = set()
    total_docs, total_dropped, n_batches = 0, 0, 0
    t0 = time.monotonic()

    for batch in iter_document_batches(
        config.mongo_uri, config.db_name, config.source_collection,
        batch_size=batch_size, categories=RAW_CATEGORY_FILTER,
    ):
        n_batches += 1
        batch, n_dropped = dedup_documents_streaming(batch, seen_hashes)
        total_dropped += n_dropped
        total_docs += len(batch)
        _accumulate_documents(batch, grouped, raw_metrics, resource_tags)
        del batch  # drop this batch's raw documents before the next is fetched

        if n_batches == 1 or n_batches % log_every == 0:
            elapsed = time.monotonic() - t0
            rate = total_docs / elapsed if elapsed > 0 else 0
            print(f"  [stream] batch {n_batches}: {total_docs} doc(s) read so far, "
                  f"{len(grouped)} accumulated row-key(s), {total_dropped} dup(s) dropped, "
                  f"{elapsed:.0f}s elapsed ({rate:.0f} docs/s).")

    if total_dropped:
        print(f"  [dedup] Dropped {total_dropped} duplicate raw documents across {n_batches} batch(es).")

    load_elapsed = time.monotonic() - t0
    print(f"  [stream] read+accumulate done: {total_docs} doc(s), {n_batches} batch(es), "
          f"{len(grouped)} row-key(s) in {load_elapsed:.0f}s. Finalizing into a DataFrame "
          f"(sort + gap-flag pass over the full accumulated set)...")

    t1 = time.monotonic()
    df = _finalize_normalized(grouped, raw_metrics, resource_tags)
    del grouped, raw_metrics, resource_tags
    print(f"  [stream] finalize done in {time.monotonic() - t1:.0f}s.")

    limit_msg = f"{total_docs} documents across {n_batches} batch(es) of ~{batch_size}"
    print(
        f"Loaded {limit_msg} (streamed, after dedup), normalized to {len(df)} "
        f"resource-time row(s). Categories: {', '.join(TARGET_CATEGORIES)} "
        f"(raw query matched: {', '.join(RAW_CATEGORY_FILTER)})."
    )
    return df


def _parse_cursor(since_iso):
    """Cursor is persisted to disk as an ISO string (JSON-safe), but
    `merged_at` in Mongo is a BSON Date (see merge_documents()). Comparing
    a Date field against a bare string in a $gt query is a type mismatch
    that silently matches nothing — so always convert back to a real
    datetime before it touches a Mongo query."""
    if not since_iso:
        return None
    if isinstance(since_iso, datetime):
        return since_iso
    return datetime.fromisoformat(since_iso)


def _changed_resource_ids_since(config, since_iso, batch_size=20000):
    """Which raw docs landed in enriched_data_lake since `since_iso`?

    Streamed via iter_document_batches() instead of load_documents(): right
    after a full backfill, `merged_at` gets stamped "now" on every document
    in the collection, so an old/stale cursor here can match the ENTIRE
    collection, not just a small delta. The old load_documents() call would
    materialize that whole match into one Python list purely to read
    `resource_id` back out of it — batching keeps memory bounded and (via
    the progress line below) makes it visible when this is happening,
    instead of it looking hung.

    Returns (resource_ids, new_cursor). new_cursor is an ISO string of the
    max merged_at actually seen, or None if nothing changed (caller should
    leave the cursor untouched then).
    """
    since_dt = _parse_cursor(since_iso)
    ids = set()
    max_merged_at = since_dt
    n_docs, n_batches = 0, 0
    t0 = time.monotonic()

    for batch in iter_document_batches(
        config.mongo_uri, config.db_name, config.source_collection,
        batch_size=batch_size, categories=None,
        extra_query={"merged_at": {"$gt": since_dt}} if since_dt else None,
    ):
        n_batches += 1
        n_docs += len(batch)
        for doc in batch:
            rid = (doc.get("resourceId") or doc.get("resource_id")
                   or doc.get("element_id") or doc.get("_id"))
            if rid is not None:
                ids.add(str(rid))
            m = doc.get("merged_at")
            if m is not None and (max_merged_at is None or m > max_merged_at):
                max_merged_at = m
        del batch
        if n_batches == 1 or n_batches % 10 == 0:
            print(f"  [changed-scan] batch {n_batches}: {n_docs} doc(s) scanned, "
                  f"{len(ids)} distinct resource_id(s) so far, "
                  f"{time.monotonic() - t0:.0f}s elapsed.")

    if not ids:
        return set(), None
    print(f"  [changed-scan] done: {n_docs} doc(s) across {n_batches} batch(es) -> "
          f"{len(ids)} distinct resource_id(s) in {time.monotonic() - t0:.0f}s.")
    return ids, (max_merged_at.isoformat() if max_merged_at else None)


def load_normalized_for_resources(config, resource_ids, batch_size=20000, id_page_size=100):
    """Pull full history (not just the newest row) for a specific set of
    resources, so the forecast/pooling logic in prediction.py still has
    enough per-resource history to work with — we're narrowing *which*
    resources get predicted this cycle, not truncating their history.

    Streamed the same way as load_normalized_streaming(): resource_ids is
    paged (id_page_size at a time, since a single $in with hundreds of
    thousands of values is itself slow/heavy on Mongo — e.g. right after a
    full backfill, when nearly every resource can look "changed"), and each
    page's matching documents are read via iter_document_batches() and
    folded into the same accumulators, discarding raw docs as we go.
    _finalize_normalized() still runs once at the end over every resource
    in `resource_ids` together — pooling (build_all_profiles etc.) still
    sees the same combined set it always has, so prediction/pooling
    behavior is unchanged; only how the raw documents got loaded changed.
    """
    if not resource_ids:
        return None
    id_list = list(resource_ids)
    # grouped, raw_metrics, resource_tags = {}, defaultdict(dict), {}
    # seen_hashes = set()
    # total_docs, total_dropped = 0, 0
    t0 = time.monotonic()

    pages = [id_list[i:i + id_page_size] for i in range(0, len(id_list), id_page_size)]
    for p_idx, page in enumerate(pages, start=1):
        grouped = {}
        raw_metrics = defaultdict(dict)
        resource_tags = {}

        seen_hashes = set()

        total_docs = 0
        total_dropped = 0

        page_start = time.monotonic()
        extra_query = {"$or": [
            {"resourceId": {"$in": page}},
            {"resource_id": {"$in": page}},
            {"element_id": {"$in": page}},
        ]}
        for batch in iter_document_batches(
            config.mongo_uri, config.db_name, config.source_collection,
            batch_size=batch_size, categories=RAW_CATEGORY_FILTER, extra_query=extra_query,
        ):
            batch, n_dropped = dedup_documents_streaming(batch, seen_hashes)
            total_dropped += n_dropped
            total_docs += len(batch)
            _accumulate_documents(batch, grouped, raw_metrics, resource_tags)
            del batch

        if p_idx == 1 or p_idx % 5 == 0 or p_idx == len(pages):
            print(f"  [predict-load] resource-id page {p_idx}/{len(pages)}: "
                  f"{total_docs} doc(s) so far, {len(grouped)} row-key(s), "
                  f"{time.monotonic() - t0:.0f}s elapsed.")
        if total_dropped:
            print(f"  [dedup] Dropped {total_dropped} duplicate raw documents.")

        df = _finalize_normalized(grouped, raw_metrics, resource_tags)

        print(
            f"  [predict-load] page {p_idx}/{len(pages)} "
            f"-> {len(df)} normalized rows "
            f"in {time.monotonic() - page_start:.0f}s."
        )

        yield df

        del df
        del grouped
        del raw_metrics
        del resource_tags

    # if total_dropped:
    #     print(f"  [dedup] Dropped {total_dropped} duplicate raw documents.")
    # df = _finalize_normalized(grouped, raw_metrics, resource_tags)
    # del grouped
    # del raw_metrics
    # del resource_tags
    # print(f"  [predict-load] done: {total_docs} doc(s) for {len(id_list)} resource(s) "
    #       f"-> {len(df)} normalized row(s) in {time.monotonic() - t0:.0f}s.")
    # return df


def cmd_predict_incremental(config, since_iso, dry_run=False):
    """Predict only for resources that changed since the last cycle,
    instead of re-scanning the whole source collection every time.

    Returns (n_predictions, new_cursor). Call with `since_iso=None` on the
    very first run — treat that as "everyone changed" so history exists
    for the initial pass.

    Note: right after a full backfill, every document gets `merged_at` =
    now, so if `since_iso` predates the backfill, this first cycle can
    legitimately match nearly the whole resource population — that's
    correct (it needs one full pass before it can go incremental), just
    slow. The batching in load_normalized_for_resources() bounds memory for
    that case; the timing logs below make its progress visible instead of
    it looking hung.
    """
    if since_iso is None:
        resource_ids, new_cursor = _changed_resource_ids_since(config, None)

        page_size = 100
        total_predictions = 0

        for i in range(0, len(resource_ids), page_size):
            page = list(resource_ids)[i:i+page_size]

            for df in load_normalized_for_resources(config, page):

                if df is None or df.empty:
                    continue

                total_predictions += predict_from_df(
                    config,
                    df,
                    dry_run=dry_run,
                    return_count=True,
                )

        return total_predictions, new_cursor# caller sets the cursor from wall-clock time after this

    FULL_SWEEP_RESOURCE_THRESHOLD = 50000
    t0 = time.monotonic()
    resource_ids, new_cursor = _changed_resource_ids_since(config, since_iso)
    if not resource_ids:
        print(f"  [predict] no new/changed docs since {since_iso} — skipping cycle.")
        return 0, since_iso

    if len(resource_ids) >= FULL_SWEEP_RESOURCE_THRESHOLD:
        # A stale/old cursor (e.g. predating a fresh backfill, which stamps
        # merged_at="now" on every document) can make nearly the whole
        # collection look "changed". Paging that many resource_ids through
        # load_normalized_for_resources() means an $in query per page that,
        # combined, rescans almost the entire collection a SECOND time (the
        # first being _changed_resource_ids_since's scan above). Above this
        # threshold it's cheaper to do one full streaming pass instead.
        print(f"  [predict] {len(resource_ids)} resource(s) changed (>= "
              f"{FULL_SWEEP_RESOURCE_THRESHOLD}) — that's most of the collection, so "
              f"doing one full streaming pass instead of paging per-resource queries.")
        df = load_normalized_streaming(config)
    else:
        print(f"  [predict] {len(resource_ids)} resource(s) changed since {since_iso} "
            f"— predicting only those (not the full collection).")

        n = 0

        for df in load_normalized_for_resources(config, resource_ids):

            if df is None or df.empty:
                continue

            n += predict_from_df(
                config,
                df,
                dry_run=dry_run,
                return_count=True,
            )

    print(f"  [predict] incremental cycle done in {time.monotonic() - t0:.0f}s "
        f"({n} prediction(s) written).")
    return n, new_cursor


def cmd_validate(config):
    df = load_normalized(config)
    issues = validate_normalized(df)
    if issues:
        print("Validation issues:")
        for issue in issues:
            print(f"- {issue}")
        return 1
    print(f"Validation passed for {len(df)} normalized resource-time rows.")
    return 0


def metric_is_mapped(category, provider, metric):
    for provider_map in REGISTRY.get(category, {}).values():
        if metric in provider_map.get(provider, {}):
            return True
    return False


def cmd_inspect(config):
    docs = load_documents(
        config.mongo_uri, config.db_name, config.source_collection,
        limit=config.limit, categories=RAW_CATEGORY_FILTER,
    )
    docs, n_dropped = dedup_documents(docs)
    print(f"  [dedup] Dropped {n_dropped} duplicate raw documents during inspect.")

    raw = Counter()
    mapped = Counter()
    unmapped = Counter()
    for doc in docs:
        category = normalize_category(doc.get("category"))
        prov = normalize_provider(doc.get("service_name") or doc.get("provider"))
        metric = doc.get("metric")
        if not category or not prov or not metric:
            continue
        key = (category, prov, metric)
        raw[key] += 1
        if metric_is_mapped(category, prov, metric):
            mapped[key] += 1
        else:
            unmapped[key] += 1

    df = normalize_documents(docs)
    available_cols = [col for col in df.columns if col.endswith("_available")] if not df.empty else []
    total_slots = len(df) * len(available_cols)
    available_slots = int(df[available_cols].astype(bool).to_numpy().sum()) if available_cols else 0
    mapped_docs = sum(mapped.values())
    unmapped_docs = sum(unmapped.values())
    raw_metric_docs = mapped_docs + unmapped_docs
    mapping_pct = (mapped_docs / raw_metric_docs * 100) if raw_metric_docs else 0
    density_pct = (available_slots / total_slots * 100) if total_slots else 0

    print(f"Inspected {len(docs)} Mongo document(s).")
    print(f"Normalized rows: {len(df)}")
    print(f"Raw metric mapping coverage: {mapped_docs}/{raw_metric_docs} ({mapping_pct:.2f}%)")
    print(f"Canonical feature density: {available_slots}/{total_slots} ({density_pct:.2f}%)")
    print("\nTop mapped raw metrics:")
    for (category, prov, metric), count in mapped.most_common(25):
        print(f"- {category} / {prov} / {metric}: {count}")
    print("\nTop unmapped raw metrics:")
    for (category, prov, metric), count in unmapped.most_common(50):
        print(f"- {category} / {prov} / {metric}: {count}")
    return 0


def cmd_train(config, batch_size=20000):
    df = load_normalized_streaming(config, batch_size=batch_size)
    issues = validate_normalized(df)
    if issues:
        print("Validation issues found. Training stopped:")
        for issue in issues:
            print(f"- {issue}")
        return 1
    trained = train_models(df, config.models_dir)
    print(f"Trained {len(trained)} category model(s): {', '.join(trained.keys()) or 'none'}")
    for category, artifact in trained.items():
        eval_metrics = artifact.get("eval_metrics")
        if eval_metrics:
            print(f"  [{category}] kind={artifact.get('kind','?')} eval={json.dumps(eval_metrics)}")
    return 0


def _build_and_persist_profiles(df, config):
    """Build one pooled profile per (canonical_resource_type, category) —
    never one per resource_id. Every resource sharing a canonical type
    (e.g. every EC2/Compute_Engine/Virtual_Machines resource, all mapped to
    `vm_instance`) contributes to and looks up the same profile.
    """
    profiles = build_all_profiles(df)
    if not profiles:
        return profiles
    try:
        for (ctype, cat), profile in profiles.items():
            upsert_profile(config.mongo_uri, config.effective_output_db, PROFILES_COLLECTION, profile)
        print(f"  Upserted {len(profiles)} resource-type profile(s) (pooled across all resources "
              f"of each type) → {config.effective_output_db}.{PROFILES_COLLECTION}.")
    except Exception as e:
        print(f"  Warning: profile persistence failed ({e}). Profiles built in-memory only.")
    return profiles


def _apply_data_quality_filters(df, config):
    """Log stale resources and resources with insufficient history.

    Does NOT drop them from df — the prediction loop handles suppression
    per-resource so the data is still available for profile building.
    """
    stale = stale_resources(
        df,
        max_age_minutes=config.freshness_max_age_minutes,
        reference_time=config.reference_time,
    )
    ref_label = (
        config.reference_time.strftime("%Y-%m-%d %H:%M UTC")
        if config.reference_time else "wall clock"
    )
    if stale:
        print(f"  [freshness] {len(stale)} stale resource(s) (>{config.freshness_max_age_minutes}min since last metric, ref={ref_label}):")
        for (rid, cat), age in list(stale.items())[:10]:
            print(f"    {cat}/{rid}: {age}min ago")
        if len(stale) > 10:
            print(f"    ... and {len(stale)-10} more.")

    history_map = resource_history_hours(df)
    thin = {k: v for k, v in history_map.items() if v < config.min_history_hours}
    if thin:
        print(f"  [min-history] {len(thin)} resource(s) with <{config.min_history_hours}h of data "
              f"(predictions suppressed).")

    return stale, history_map


def cmd_predict(config, dry_run=False):
    df = load_normalized(config)
    return predict_from_df(config, df, dry_run=dry_run)


def predict_from_df(config, df, dry_run=False, return_count=False):
    if df.empty or "category" not in df.columns:
        print("No normalized rows to predict on (source collection empty or not yet "
              "populated) — skipping this cycle.")
        return 0

    models = load_models(config.models_dir)
    if not models:
        print("No model artifacts found. Training models first.")
        models = train_models(df, config.models_dir)

    stale, history_map = _apply_data_quality_filters(df, config)

    predictions = generate_predictions(
        df, models,
        target_categories=TARGET_CATEGORIES,
        stale_resources=stale,
        history_map=history_map,
        min_history_hours=config.min_history_hours,
        pipeline_version=PIPELINE_VERSION,
    )

    # --- Explainability: attach plain-English narrative to every prediction ---
    attach_explanations(predictions)

    # --- Maintenance window suppression (skip if collection unreachable) ---
    try:
        windows_col = get_collection(config.mongo_uri, config.db_name, MAINTENANCE_WINDOWS_COLLECTION)
        active_windows = load_active_windows(windows_col)
        if active_windows:
            apply_maintenance_suppression(predictions, active_windows)
            print(f"  [maintenance] Suppressed alerts for {len(active_windows)} active window(s).")
    except Exception as e:
        print(f"  [maintenance] Skipped (collection unavailable: {e}).")

    # --- Cross-domain correlation: composite incidents across categories ---
    composite_incidents = detect_composite_incidents(predictions)
    if composite_incidents:
        print(f"  [cross-domain] Detected {len(composite_incidents)} composite incident(s) "
              f"spanning multiple categories on shared infrastructure.")

    # --- Anomaly clustering: fleet/blast-radius detection ---
    fleet_anomalies = cluster_anomalies(predictions)
    if fleet_anomalies:
        print(f"  [clustering] Detected {len(fleet_anomalies)} fleet-wide anomaly cluster(s) "
              f"(likely platform/region-level cause, not independent incidents).")

    # --- Drift detection ---
    drift = check_drift_trigger(predictions)
    retrain_cats = [c for c, d in drift.items() if d["retrain_flag"]]
    if retrain_cats:
        print(f"  [drift] Retraining recommended for: {retrain_cats}")
        for cat in retrain_cats:
            d = drift[cat]
            print(f"    {cat}: anomaly_rate={d['anomaly_rate']:.0%} ({d['anomalous']}/{d['n']})")

    if dry_run:
        sample = predictions[:10]
        print(json.dumps(sample, indent=2, default=str))
        print(f"Dry run produced {len(predictions)} prediction(s), "
              f"{len(composite_incidents)} composite incident(s), "
              f"{len(fleet_anomalies)} fleet anomaly cluster(s).")
        return 0

    # --- Route predictions to per-category collections (shared across projects) ---
    by_category = {}
    for p in predictions:
        by_category.setdefault(p.get("category", "Unknown"), []).append(p)

    total_inserted = 0
    for category, preds in by_category.items():
        col_name = _collection_for(config, category)
        inserted = write_predictions(config.mongo_uri, config.effective_output_db, col_name, preds)
        print(f"Wrote {inserted} {category} prediction(s) → {config.effective_output_db}.{col_name}.")
        total_inserted += inserted

    # --- Write composite incidents and fleet anomalies to their own collections ---
    if composite_incidents:
        n = write_predictions(config.mongo_uri, config.effective_output_db, COMPOSITE_INCIDENTS_COLLECTION, composite_incidents)
        print(f"Wrote {n} composite incident(s) → {config.effective_output_db}.{COMPOSITE_INCIDENTS_COLLECTION}.")

    # Fleet anomaly clusters are intentionally NOT split per project — they
    # group by provider/location/category to catch platform- or region-wide
    # events that can span multiple projects at once. Splitting them per
    # project would hide cross-project blast radius, which is the point of
    # this detector.
    if fleet_anomalies:
        n = write_predictions(config.mongo_uri, config.effective_output_db, FLEET_ANOMALIES_COLLECTION, fleet_anomalies)
        print(f"Wrote {n} fleet anomaly cluster(s) → {config.effective_output_db}.{FLEET_ANOMALIES_COLLECTION}.")

    print(f"Total predictions written: {total_inserted}  [pipeline v{PIPELINE_VERSION}]")
    return total_inserted if return_count else 0


def cmd_run(config, dry_run=False):
    df = load_normalized(config)
    issues = validate_normalized(df)
    if issues:
        print("Validation issues:")
        for issue in issues:
            print(f"- {issue}")
        return 1
    print(f"Validation passed for {len(df)} normalized resource-time rows.")

    print("Building resource-type profiles (pooled across all resources of each type)...")
    _build_and_persist_profiles(df, config)

    trained = train_models(df, config.models_dir)
    print(f"Trained {len(trained)} category model(s): {', '.join(trained.keys()) or 'none'}")
    for category, artifact in trained.items():
        eval_metrics = artifact.get("eval_metrics")
        if eval_metrics:
            print(f"  [{category}] kind={artifact.get('kind','?')} eval={json.dumps(eval_metrics)}")

    return predict_from_df(config, df, dry_run=dry_run)


def cmd_feedback(config, incidents_collection_name=None, export_path=None):
    """Run nightly feedback pass + print performance metrics + staleness check."""
    # Staleness check — one shared collection per category now (no per-project split).
    storage_col_name = config.output_collection
    try:
        storage_col = get_collection(config.mongo_uri, config.effective_output_db, storage_col_name)
        staleness = check_feedback_staleness(storage_col)
        print(f"  [feedback-staleness] {storage_col_name}: "
              f"{'WARNING stale' if staleness['stale'] else 'OK'} "
              f"(last feedback {staleness.get('last_feedback_at', 'never')}).")
    except Exception as e:
        print(f"  [feedback-staleness] WARNING: could not read '{storage_col_name}' in "
              f"{config.effective_output_db} ({e}) — has `predict` been run yet?")

    incidents_col = None
    if incidents_collection_name:
        incidents_col = get_collection(config.mongo_uri, config.effective_output_db, incidents_collection_name)

    # One collection per category, shared across all projects.
    all_category_cols = []
    for cat in TARGET_CATEGORIES:
        attr = CATEGORY_OUTPUT_COLLECTION.get(cat, "output_collection")
        col_name = getattr(config, attr)
        col = get_collection(config.mongo_uri, config.effective_output_db, col_name)
        all_category_cols.append((col, cat))

    total_summary = {"labelled": 0, "true_positives": 0, "false_positives": 0}
    for col, label in all_category_cols:
        summary = run_feedback_pass(col, incidents_col=incidents_col)
        print(f"[{label}] Feedback pass: {json.dumps(summary)}")
        for k in ("labelled", "true_positives", "false_positives"):
            total_summary[k] += summary.get(k, 0)

    print(f"\nTotal: {json.dumps(total_summary)}")
    print("\n--- Performance metrics (last 30d) ---")
    for col, label in all_category_cols:
        perf = compute_performance_metrics(col)
        print(f"[{label}]: {json.dumps(perf, indent=2)}")

    if export_path:
        for col, label in all_category_cols:
            out = f"{export_path}_{label.lower()}_labels.csv"
            n = export_training_labels(col, out)
            print(f"Exported {n} labelled rows → {out}")

    return 0


def parser():
    p = argparse.ArgumentParser(description=f"Cloud predictive analytics pipeline v{PIPELINE_VERSION}")
    p.add_argument("command", choices=["inspect", "validate", "train", "predict", "run", "feedback"])
    p.add_argument("--mongo-uri",      default="mongodb://localhost:27017")
    p.add_argument("--db",             default="mydb")
    p.add_argument("--output-db",      default=None,
                   help="Write all pipeline-generated collections (predictions, composite "
                        "incidents, fleet anomalies, resource profiles) to this database "
                        "instead of --db. Source data and maintenance windows are still "
                        "read from --db. Defaults to --db if omitted.")
    p.add_argument("--source",         default="22predictive_analytics_dataset_sorted")
    p.add_argument("--out",            default="22prediction_logs_storage")
    p.add_argument("--compute-out",    default="22prediction_logs_compute")
    p.add_argument("--network-out",    default="22prediction_logs_network")
    p.add_argument("--container-out",  default="22prediction_logs_container")
    p.add_argument("--databases-out",  default="22prediction_logs_databases")
    p.add_argument("--governance-out", default="22prediction_logs_governance")
    p.add_argument("--security-out",   default="22prediction_logs_security")
    p.add_argument("--models-dir",     default="models")
    p.add_argument("--limit",          type=int, default=0)
    p.add_argument("--train-batch-size", type=int, default=20000,
                   help="Documents fetched per Mongo cursor batch during training "
                        "(cmd_train). Lower it if memory is still tight; the full "
                        "collection is still processed either way, just in chunks.")
    p.add_argument("--dry-run",        action="store_true")
    p.add_argument("--incidents-collection", default=None)
    p.add_argument("--export-labels",  default=None)
    p.add_argument(
        "--freshness-max-age", dest="freshness_max_age", type=int, default=None,
        metavar="MINUTES",
        help=(
            "Override the freshness gate (default 60 min for real-time, auto-detected "
            "for --reference-time). Set to e.g. 2880 for daily-granularity data."
        ),
    )
    p.add_argument(
        "--reference-time", dest="reference_time", default=None, metavar="DATE_OR_AUTO",
        help=(
            "Set the 'current moment' the pipeline evaluates staleness and forecasts against. "
            "Use 'auto' to detect max(to_date) from the source collection (recommended for "
            "static/historical datasets). Use 'YYYY-MM-DD' or 'YYYY-MM-DDTHH:MM:SS' for an "
            "explicit date. Omit for real-time streaming data (defaults to wall-clock time)."
        ),
    )
    return p


def main():
    args = parser().parse_args()
    config = make_config(args)
    if args.command == "validate":
        raise SystemExit(cmd_validate(config))
    if args.command == "inspect":
        raise SystemExit(cmd_inspect(config))
    if args.command == "train":
        raise SystemExit(cmd_train(config, batch_size=args.train_batch_size))
    if args.command == "predict":
        raise SystemExit(cmd_predict(config, dry_run=args.dry_run))
    if args.command == "run":
        raise SystemExit(cmd_run(config, dry_run=args.dry_run))
    if args.command == "feedback":
        raise SystemExit(cmd_feedback(
            config,
            incidents_collection_name=args.incidents_collection,
            export_path=args.export_labels,
        ))


if __name__ == "__main__":
    main()
