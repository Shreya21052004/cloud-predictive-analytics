"""resource_types.py — Canonical resource-type mapping for pooled forecasting.

`component` (see normalizer.base_context) is a raw, provider-specific string
-- EC2, Virtual_Machines, and Compute_Engine are three different spellings of
the same real-world resource type (a VM). Pooling forecasts by raw component
under-pools: a sparse Compute_Engine resource can't borrow strength from the
much larger EC2 population even though they behave the same way.

This module maps raw component -> canonical_resource_type, per the mapping
agreed in the handoff doc (section 4), plus the follow-up review of the
components that were previously falling into other_<category> (Accounts,
Container_Registry, Azure_Private_Link, Recovery_Services,
Application_Gateway, Load_Balancers, Azure_Kubernetes_Service). A component
only gets merged into an existing bucket when it's genuinely the same kind
of thing behaviorally (e.g. Load_Balancers + Application_Gateway); a
component with no real behavioral peer gets its own dedicated canonical
type instead of being force-merged just to hit a volume target. Anything
still unseen falls back to an explicit `other_<category>` bucket. Nothing
is ever silently dropped, and unmapped values are logged once each so
new/unexpected component strings surface for review instead of
disappearing into the fallback unnoticed.

ElastiCache is deliberately NOT merged into the NoSQL/KV bucket: it's an
in-memory cache (hit rate / eviction / memory pressure) rather than a
persistent KV store (read/write capacity / throttling), and its trend
behavior is different enough that merging risks blurring both. It gets its
own canonical type; only merge it with nosql_kv_store later if profiling the
pooled fit shows that actually helps.
"""
import logging

logger = logging.getLogger(__name__)

# raw component (as it appears on normalized rows, i.e. after
# normalizer.base_context's component derivation) -> canonical_resource_type
CANONICAL_RESOURCE_TYPE_MAP = {
    # VM / instance (~71,000 logs pooled)
    "Compute":                "vm_instance",
    "Compute_Engine":         "vm_instance",
    "EC2":                    "vm_instance",
    "Virtual_Machines":       "vm_instance",

    # Serverless function (~43,600 logs)
    "Lambda":                 "serverless_function",

    # VPC / virtual network (~271,000 logs). VPC is merged regardless of
    # which raw category (Network vs Networking) it was logged under --
    # category normalization already folds Networking -> Network upstream
    # (see config.CATEGORY_ALIASES), so no special-casing is needed here.
    "VPC":                    "vpc_network",
    "Virtual_Networks":       "vpc_network",
    "Virtual_Cloud_Networks": "vpc_network",

    # Block storage (~103,600 logs)
    "EBS":                    "block_storage",
    "Storage_Disks":          "block_storage",

    # Object storage (~47,500 logs)
    "S3":                     "object_storage",
    "Bucket":                 "object_storage",

    # File storage (~6,500 logs)
    "EFS":                    "file_storage",
    "File_Storage":           "file_storage",

    # NoSQL / KV store (~13,600 logs). ElastiCache intentionally excluded --
    # see module docstring.
    "DynamoDB":               "nosql_kv_store",

    # Relational / generic DB (~6,400 logs)
    "Databases":              "relational_generic_db",
    "Servers":                "relational_generic_db",

    # DNS (~2,600 logs)
    "Route53":                "dns",
    "Route53_Resolver":       "dns",
    "Private_DNS_Zones":      "dns",

    # In-memory cache -- its own type, not merged with nosql_kv_store
    "ElastiCache":            "in_memory_cache",

    # --- Added: previously-unmapped components now surfaced by the
    # "unmapped raw component" warning (see freshness/volume from the
    # 2026-07 production run). These are each the sole representative of
    # their function within their category in this dataset (no equivalent
    # component from another provider to pool with), so they are given
    # their own dedicated canonical type rather than force-merged into an
    # unrelated bucket. This still moves them out of the generic
    # other_<category> fallback and into named, trackable pooled types.
    #
    # Load_Balancers (L4) and Application_Gateway (L7) ARE merged together:
    # both are traffic-distribution front ends whose core signals
    # (throughput / connection count / latency) behave the same way for
    # forecasting purposes, so pooling them is a genuine strength-borrowing
    # win rather than a volume-target merge.
    "Load_Balancers":         "load_balancer",
    "Application_Gateway":    "load_balancer",

    # Azure Private Link -- private connectivity endpoint, not the same
    # traffic-shape as a VPC/VNet, kept distinct.
    "Azure_Private_Link":     "private_link_endpoint",

    # Azure storage "Accounts" -- a management-level storage account
    # resource, distinct from blob/file/disk usage patterns already
    # covered by object_storage / file_storage / block_storage.
    "Accounts":               "storage_account",

    # Azure Recovery Services (backup/DR vault) -- backup job/retention
    # behavior, not ongoing storage utilization; kept distinct.
    "Recovery_Services":      "backup_recovery_vault",

    # Container Registry -- image storage/pull activity, not a compute
    # workload; kept distinct from the cluster type below.
    "Container_Registry":     "container_registry",

    # Azure Kubernetes Service -- cluster-level compute, kept distinct
    # from the registry above.
    "Azure_Kubernetes_Service": "k8s_cluster",
}

# The 9 canonical types above that are well-populated enough to get a
# dedicated pooled model (handoff doc §4/§5, "Tier 2"). Anything mapping to
# an other_<category> bucket falls back to category-level pooling ("Tier 1")
# instead. Kept as an explicit set (rather than re-deriving it) so tier
# classification elsewhere doesn't silently drift if the map above changes.
WELL_POPULATED_CANONICAL_TYPES = frozenset(CANONICAL_RESOURCE_TYPE_MAP.values())

_warned_components = set()


def canonical_resource_type(category, component):
    """Map a (category, raw component) pair to its canonical resource type.

    Returns one of the well-populated canonical types in
    CANONICAL_RESOURCE_TYPE_MAP, or an `other_<category>` fallback bucket for
    anything else (unmapped/unseen components not yet reviewed and added to
    the map above). Never raises, never drops a value silently.
    """
    cat = category or "unknown"
    if not component:
        return f"other_{cat}"
    canonical = CANONICAL_RESOURCE_TYPE_MAP.get(str(component))
    if canonical is not None:
        return canonical
    if component not in _warned_components:
        _warned_components.add(component)
        logger.warning(
            "resource_types.canonical_resource_type: unmapped raw component "
            "%r (category=%s) -> falling back to other_%s bucket. If this "
            "type accumulates real volume, consider adding it to "
            "CANONICAL_RESOURCE_TYPE_MAP.",
            component, cat, cat,
        )
    return f"other_{cat}"


def is_well_populated(canonical_type):
    """True if `canonical_type` is one of the 9 dedicated pooled types
    (Tier 2 eligible), False if it's an other_<category> fallback bucket
    (Tier 1 only)."""
    return canonical_type in WELL_POPULATED_CANONICAL_TYPES
