from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple


@dataclass(frozen=True)
class PipelineConfig:
    mongo_uri: str = "mongodb://localhost:27017"
    db_name: str = "mydb"
    # If set, every collection the pipeline WRITES (predictions, composite
    # incidents, fleet anomalies, resource profiles) is created in this
    # database instead of db_name. Reads (source data, maintenance windows)
    # still come from db_name. Leave unset ("") to write to db_name as before.
    output_db_name: str = ""
    source_collection: str = "22predictive_analytics_dataset_sorted"
    output_collection: str = "22prediction_logs_storage"
    compute_output_collection: str = "22prediction_logs_compute"
    network_output_collection: str = "22prediction_logs_network"
    container_output_collection: str = "22prediction_logs_container"
    databases_output_collection: str = "22prediction_logs_databases"
    governance_output_collection: str = "22prediction_logs_governance"
    security_output_collection: str = "22prediction_logs_security"
    models_dir: Path = Path("models")
    min_points_for_model: int = 20
    limit: int = 0  # 0 = load all documents

    @property
    def effective_output_db(self) -> str:
        return self.output_db_name or self.db_name

    # Data quality thresholds
    freshness_max_age_minutes: int = 60       # Prediction freshness gate
    min_history_hours: float = 24.0           # Min history before predicting
    late_arrival_tolerance_minutes: int = 15  # Late arrival window
    drift_anomaly_rate_threshold: float = 0.40  # Retraining trigger

    # Reference time: the "current moment" the pipeline evaluates against.
    # None  → use datetime.now(utc) — correct for real-time streaming data.
    # Set   → treat this timestamp as "now" for staleness/freshness/forecast
    #         horizon calculations. Use --reference-time auto (detect from
    #         max(to_date) in source) or --reference-time YYYY-MM-DD for
    #         static / historical datasets so resources don't appear stale.
    reference_time: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Category definitions
# ---------------------------------------------------------------------------

TARGET_CATEGORIES = (
    "Storage", "Compute", "Network",
    "Container", "Databases",
)

# CATEGORY_ALIASES = {
#     "compute":    "Compute",
#     "network":    "Network",
#     "networking": "Network",
#     "storage":    "Storage",
#     "container":  "Container",
#     "containers": "Container",
#     "database":   "Databases",
#     "databases":  "Databases",
#     # Governance and Security are intentionally excluded from processing.
#     # Left unmapped here so any stray record in those categories is dropped
#     # by normalize_documents() rather than silently re-included.
# }

CATEGORY_ALIASES = {
    # Existing
    "compute": "Compute",
    "network": "Network",
    "networking": "Network",
    "storage": "Storage",
    "container": "Container",
    "containers": "Container",
    "database": "Databases",
    "databases": "Databases",

    # Azure / AWS / GCP resource categories
    "virtual_machines": "Compute",
    "vm_instances": "Compute",
    "virtual_machine_scale_sets": "Compute",
    "web_apps": "Compute",
    "appservice_plan": "Compute",

    "storage_accounts": "Storage",
    "file_systems": "Storage",
    "volumes": "Storage",
    "table": "Storage",

    "application_gateway": "Network",
    "load_balancer": "Network",
    "nat_gateways": "Network",
    "subnets": "Network",
    "private_endpoints": "Network",
    "private_dns_zones": "Network",
    "public_ip_address": "Network",
    "routes": "Network",
    "resolver_inbound_endpoint": "Network",
    "transit_gateways": "Network",
    "transit_gateway_attachments": "Network",

    "ms-sql_db": "Databases",
    "mariadb": "Databases",
}

# Raw spellings to match in MongoDB query — must cover every alias that maps
# to a TARGET_CATEGORIES entry. Governance/Security are deliberately absent:
# the pipeline should not even query for them.
RAW_CATEGORY_FILTER = (
    "Storage", "Compute", "Network", "Networking",
    "Container", "Databases", "Database",
)

SUPPORTED_CATEGORIES = TARGET_CATEGORIES

# Output collection per canonical category
CATEGORY_OUTPUT_COLLECTION = {
    "Storage":    "output_collection",
    "Compute":    "compute_output_collection",
    "Network":    "network_output_collection",
    "Container":  "container_output_collection",
    "Databases":  "databases_output_collection",
}

# ---------------------------------------------------------------------------
# Provider aliases
# ---------------------------------------------------------------------------

PROVIDER_ALIASES = {
    "aws":                    "AWS",
    "amazon":                 "AWS",
    "amazon web services":    "AWS",
    "azure":                  "Azure",
    "microsoft azure":        "Azure",
    "gcp":                    "GCP",
    "google":                 "GCP",
    "google cloud":           "GCP",
    "google cloud platform":  "GCP",
    "oci":                    "OCI",
    "oracle":                 "OCI",
    "oracle cloud":           "OCI",
}

# ---------------------------------------------------------------------------
# Risk thresholds
# ---------------------------------------------------------------------------

RISK_THRESHOLDS = {
    "INFO":     25,
    "WARNING":  50,
    "HIGH":     75,
    "CRITICAL": 90,
}

# ---------------------------------------------------------------------------
# Model versioning
# ---------------------------------------------------------------------------

PIPELINE_VERSION = "6.0.0"
