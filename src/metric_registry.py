"""metric_registry.py — Canonical feature definitions for all supported categories.

Categories: Compute, Network (+ Networking alias), Storage,
            Container, Databases, Governance, Security.
"""


def pct(value):
    """Pass the raw value through unchanged.

    Previously this guessed that any value between 0 and 1 was a fraction
    meaning it should be scaled to a 0-100 percentage. That guess is wrong
    whenever a resource's real percentage value is itself small (e.g. a
    genuinely near-idle CPU reporting 0.41%) — it gets misread as a 0.41
    fraction and inflated to 41. Values are used exactly as reported by
    the source system: no 0-1 -> 0-100 conversion, ever.
    """
    if value is None:
        return None
    return float(value)


def identity(value):
    if value is None:
        return None
    return float(value)


def inverse_pct(value):
    if value is None:
        return None
    return 100.0 - pct(value)


REGISTRY = {
    # ------------------------------------------------------------------
    # COMPUTE
    # ------------------------------------------------------------------
    "Compute": {
        "canonical_cpu_pct": {
            "AWS":   {"CPUUtilization": pct},
            "Azure": {"Percentage CPU": pct},
            "GCP":   {"compute.googleapis.com/instance/cpu/utilization": pct},
            "OCI":   {"CpuUtilization": pct},
        },
        "canonical_cpu_idle_pct": {
            "AWS":   {"cpu_usage_idle": pct},
            "Azure": {"Percentage CPU": inverse_pct},
            "GCP":   {"compute.googleapis.com/instance/cpu/utilization": inverse_pct},
        },
        "canonical_iowait_pct": {
            "AWS": {"cpu_usage_iowait": pct},
        },
        "canonical_mem_used_pct": {
            "AWS": {"mem_used_percent": pct},
            "GCP": {"agent.googleapis.com/memory/percent_used": pct},
            "OCI": {"MemoryUtilization": pct},
        },
        "canonical_swap_used_pct": {
            "AWS": {"swap_used_percent": pct},
        },
        "canonical_disk_read_bytes": {
            "AWS":   {"DiskReadBytes": identity},
            "Azure": {"Disk Read Bytes": identity, "OS Disk Read Bytes/sec": identity, "Data Disk Read Bytes/sec": identity},
            "GCP":   {"compute.googleapis.com/instance/disk/read_bytes_count": identity},
            "OCI":   {"DiskBytesRead": identity},
        },
        "canonical_disk_write_bytes": {
            "AWS":   {"DiskWriteBytes": identity},
            "Azure": {"Disk Write Bytes": identity, "OS Disk Write Bytes/sec": identity, "Data Disk Write Bytes/sec": identity},
            "GCP":   {"compute.googleapis.com/instance/disk/write_bytes_count": identity},
            "OCI":   {"DiskBytesWritten": identity},
        },
        "canonical_disk_read_iops": {
            "AWS":   {"DiskReadOps": identity},
            "Azure": {"Disk Read Operations/Sec": identity, "OS Disk Read Operations/Sec": identity, "Data Disk Read Operations/Sec": identity},
            "OCI":   {"DiskIopsRead": identity},
        },
        "canonical_disk_write_iops": {
            "AWS":   {"DiskWriteOps": identity},
            "Azure": {"Disk Write Operations/Sec": identity, "OS Disk Write Operations/Sec": identity, "Data Disk Write Operations/Sec": identity},
            "OCI":   {"DiskIopsWritten": identity},
        },
        "canonical_disk_used_pct": {
            "AWS":   {"disk_used_percent": pct},
            "Azure": {"\\LogicalDisk(_Total)\\% Free Space": inverse_pct, "disk/free_percent": inverse_pct},
        },
        "canonical_net_in_bytes": {
            "AWS":   {"NetworkIn": identity, "NetworkPacketsIn": identity},
            "Azure": {"Network In Total": identity, "Network In": identity},
            "GCP":   {"compute.googleapis.com/instance/network/received_bytes_count": identity},
            "OCI":   {"NetworksBytesIn": identity},
        },
        "canonical_net_out_bytes": {
            "AWS":   {"NetworkOut": identity, "NetworkPacketsOut": identity},
            "Azure": {"Network Out Total": identity, "Network Out": identity},
            "GCP":   {"compute.googleapis.com/instance/network/sent_bytes_count": identity},
            "OCI":   {"NetworksBytesOut": identity},
        },
        "status_check_failed": {
            "AWS": {"StatusCheckFailed": identity, "StatusCheckFailed_Instance": identity, "StatusCheckFailed_System": identity},
        },
        "lambda_throttles":             {"AWS": {"Throttles": identity}},
        "lambda_concurrent_executions": {"AWS": {"ConcurrentExecutions": identity}},
        "lambda_invocations":           {"AWS": {"Invocations": identity}},
        "lambda_errors":                {"AWS": {"Errors": identity}},
        "lambda_duration_ms":           {"AWS": {"Duration": identity}},
        "cpu_credit_balance": {
            "AWS":   {"CPUCreditBalance": identity},
            "Azure": {"CPU Credits Remaining": identity},
        },
        "cpu_credit_usage":             {"AWS": {"CPUCreditUsage": identity}},
        "cpu_surplus_credit_balance":   {"AWS": {"CPUSurplusCreditBalance": identity}},
        "cpu_surplus_credits_charged":  {"AWS": {"CPUSurplusCreditsCharged": identity}},
        "load_average":                 {"OCI": {"LoadAverage": identity}},
    },

    # ------------------------------------------------------------------
    # NETWORK  (Networking aliases to this in normalizer)
    # ------------------------------------------------------------------
    "Network": {
        "canonical_net_in_bytes": {
            "AWS":   {"BytesIn": identity, "PacketsIn": identity},
            "Azure": {"ByteCount": identity, "PEBytesIn": identity, "PacketCount": identity},
            "GCP":   {"networking.googleapis.com/vm_flow/ingress_bytes_count": identity},
            # OCI's SubnetIpUtilization used to be duplicated here as a raw
            # identity passthrough AND under canonical_subnet_util_pct as a
            # pct() transform below -- the same source value being read two
            # incompatible ways (unbounded bytes vs. 0-100% utilization)
            # depending on which canonical column won the fallback race in
            # choose_forecast_feature(). It is a utilization percentage, not
            # a byte counter, so it now lives only under subnet_util_pct.
        },
        "canonical_net_out_bytes": {
            "AWS":   {"BytesOut": identity, "PacketsOut": identity},
            "Azure": {"PEBytesOut": identity},
            "GCP":   {"networking.googleapis.com/vm_flow/egress_bytes_count": identity},
        },
        "canonical_active_connections": {
            "AWS":   {"ActiveConnectionCount": identity},
            "Azure": {"TotalConnectionCount": identity},
        },
        "canonical_health_pct": {
            "AWS":   {"HealthCheckPercentageHealthy": pct},
            "Azure": {"VipAvailability": pct, "DipAvailability": pct},
        },
        "canonical_health_status": {
            "AWS": {"HealthCheckStatus": identity},
        },
        "canonical_ddos_signal": {
            "AWS":   {"PacketDropCountBlackhole": identity, "PacketDropCountNoRoute": identity},
            "Azure": {
                "IfUnderDDoSAttack": identity,
                "PacketsInDDoS": identity,
                "BytesInDDoS": identity,
                "PacketsDroppedDDoS": identity,
                "TCPPacketsInDDoS": identity,
                "UDPPacketsInDDoS": identity,
                "SynCount": identity,
            },
        },
        "canonical_subnet_util_pct": {
            "Azure": {"VirtualNetworkLinkCapacityUtilization": pct, "RecordSetCapacityUtilization": pct},
            "OCI":   {"SubnetIpUtilization": pct},
            # BUG FIX: GCP's "predicted_max_vpc_flow_logs_count" used to be
            # mapped here with `identity` -- but it's a flow-log count
            # prediction (observed values 40-400+), not a 0-100% utilization
            # reading. Mapping a raw count into a "pct" column is exactly
            # the kind of unit mismatch fixed elsewhere in this file (see
            # the OCI net_in_bytes note above); it also meant this column's
            # "bounded percentage" status -- which the forecast-feature
            # priority list specifically relies on -- was a fiction for GCP.
            # GCP has no genuine subnet-utilization metric in this dataset;
            # the count now lives honestly under canonical_flow_log_count.
        },
        "canonical_flow_log_count": {
            "GCP": {"networking.googleapis.com/vpc_flow/predicted_max_vpc_flow_logs_count": identity},
        },
        "canonical_dns_queries": {
            "AWS":   {"InboundQueryVolume": identity},
            "Azure": {"QueryVolume": identity},
        },
        "canonical_throughput": {
            "AWS":   {"Throughput": identity},
        },
        "canonical_probe_failure_count": {
            # Previously unmapped entirely -- 56,339 occurrences in the
            # sample data (3rd most common GCP Network metric after the two
            # byte counters and predicted_max_vpc_flow_logs_count) were
            # silently dropped because no canonical feature registered this
            # raw metric name at all, regardless of the STAT_PRIORITY fix.
            "GCP": {"networking.googleapis.com/cloud_netslo/active_probing/probe_count": identity},
        },
    },

    # ------------------------------------------------------------------
    # STORAGE
    # ------------------------------------------------------------------
    "Storage": {
        "canonical_storage_read_bytes": {
            "AWS":   {"VolumeReadBytes": identity, "DataReadIOBytes": identity},
            "Azure": {"Composite Disk Read Bytes/sec": identity, "Ingress": identity},
            "GCP":   {"storage.googleapis.com/network/received_bytes_count": identity},
            "OCI":   {"FileSystemReadThroughput": identity},
        },
        "canonical_storage_write_bytes": {
            "AWS":   {"VolumeWriteBytes": identity, "DataWriteIOBytes": identity},
            "Azure": {"Composite Disk Write Bytes/sec": identity, "Egress": identity},
            "GCP":   {"storage.googleapis.com/network/sent_bytes_count": identity},
            "OCI":   {"FileSystemWriteThroughput": identity},
        },
        "canonical_storage_read_iops": {
            "AWS":   {"VolumeReadOps": identity},
            "Azure": {"Composite Disk Read Operations/sec": identity},
            "GCP":   {"storage.googleapis.com/api/request_count": identity},
            "OCI":   {"MetadataIOPS": identity},
        },
        "canonical_storage_write_iops": {
            "AWS":   {"VolumeWriteOps": identity},
            "Azure": {"Composite Disk Write Operations/sec": identity},
            "GCP":   {"storage.googleapis.com/api/request_count": identity},
            "OCI":   {"MetadataIOPS": identity},
        },
        "canonical_storage_used_pct": {
            "AWS": {"PercentIOLimit": pct},
            "OCI": {"FileSystemUsage": pct},
        },
        "canonical_storage_capacity_raw": {
            "AWS":   {"BucketSizeBytes": identity},
            "Azure": {"UsedCapacity": identity},
            "GCP":   {"storage.googleapis.com/storage/total_byte_seconds": identity},
        },
        "canonical_storage_latency_ms": {
            "AWS":   {"VolumeTotalReadTime": identity, "VolumeTotalWriteTime": identity},
            "Azure": {"SuccessE2ELatency": identity, "SuccessServerLatency": identity},
        },
        "canonical_queue_depth":          {"AWS":   {"VolumeQueueLength": identity}},
        "canonical_storage_transactions": {"Azure": {"Transactions": identity}},
        "canonical_volume_idle_time":     {"AWS":   {"VolumeIdleTime": identity}},
        "canonical_permitted_throughput": {"AWS":   {"PermittedThroughput": identity}},
        "canonical_burst_balance_pct":    {"AWS":   {"BurstBalance": pct}},
        "canonical_credit_balance_raw":   {"AWS":   {"BurstCreditBalance": identity}},
        "availability_pct":               {"Azure": {"Availability": pct}},
        "backup_health_event":            {"Azure": {"BackupHealthEvent": identity}},
    },

    # ------------------------------------------------------------------
    # CONTAINER  (Azure AKS only in current dataset)
    # ------------------------------------------------------------------
    "Container": {
        "canonical_node_cpu_pct": {
            "Azure": {"node_cpu_usage_percentage": pct},
        },
        "canonical_node_cpu_millicores": {
            "Azure": {"node_cpu_usage_millicores": identity},
        },
        "canonical_node_mem_pct": {
            "Azure": {
                "node_memory_working_set_percentage": pct,
                "node_memory_rss_percentage": pct,
            },
        },
        "canonical_node_mem_bytes": {
            "Azure": {
                "node_memory_working_set_bytes": identity,
                "node_memory_rss_bytes": identity,
                "kube_node_status_allocatable_memory_bytes": identity,
            },
        },
        "canonical_node_disk_pct": {
            "Azure": {"node_disk_usage_percentage": pct},
        },
        "canonical_node_disk_bytes": {
            "Azure": {"node_disk_usage_bytes": identity},
        },
        "canonical_node_net_in_bytes": {
            "Azure": {"node_network_in_bytes": identity},
        },
        "canonical_node_net_out_bytes": {
            "Azure": {"node_network_out_bytes": identity},
        },
        "canonical_pod_ready": {
            "Azure": {"kube_pod_status_ready": identity, "kube_pod_status_phase": identity},
        },
        "canonical_unschedulable_pods": {
            "Azure": {"cluster_autoscaler_unschedulable_pods_count": identity},
        },
        "canonical_unneeded_nodes": {
            "Azure": {"cluster_autoscaler_unneeded_nodes_count": identity},
        },
        "canonical_api_inflight_requests": {
            "Azure": {"apiserver_current_inflight_requests": identity},
        },
        "canonical_allocatable_cpu_cores": {
            "Azure": {"kube_node_status_allocatable_cpu_cores": identity},
        },
        "canonical_node_condition": {
            "Azure": {"kube_node_status_condition": identity},
        },
        "canonical_storage_used_bytes": {
            "Azure": {"StorageUsed": identity},
        },
    },

    # ------------------------------------------------------------------
    # DATABASES
    # ------------------------------------------------------------------
    "Databases": {
        "canonical_db_cpu_pct": {
            "AWS":   {"CPUUtilization": pct, "EngineCPUUtilization": pct},
            "Azure": {"cpu_percent": pct, "dtu_consumption_percent": pct},
        },
        "canonical_db_mem_pct": {
            "AWS":   {"DatabaseMemoryUsagePercentage": pct},
            "Azure": {"memory_percent": pct},
        },
        "canonical_db_storage_pct": {
            "Azure": {"storage_percent": pct},
        },
        "canonical_db_storage_bytes": {
            "Azure": {"storage": identity},
            "AWS":   {"allocated_data_storage": identity},
        },
        "canonical_db_connections": {
            "Azure": {"active_connections": identity, "sessions_count": identity},
        },
        "canonical_db_dtu_used": {
            "Azure": {"dtu_used": identity},
        },
        "canonical_db_read_capacity": {
            "AWS": {"ConsumedReadCapacityUnits": identity, "ProvisionedReadCapacityUnits": identity},
        },
        "canonical_db_write_capacity": {
            "AWS": {"ConsumedWriteCapacityUnits": identity, "ProvisionedWriteCapacityUnits": identity},
        },
        "canonical_db_net_in_bytes": {
            "AWS": {"NetworkBytesIn": identity},
        },
        "canonical_db_blocked_by_firewall": {
            "Azure": {"blocked_by_firewall": identity},
        },
    },

    # ------------------------------------------------------------------
    # GOVERNANCE  (selective — avoids 180-feature sparse nightmare)
    # ------------------------------------------------------------------
    "Governance": {
        "canonical_cpu_pct": {
            "AWS":   {"CPUUtilization": pct},
            "Azure": {"Percentage CPU": pct, "IntegrationRuntimeCpuPercentage": pct},
        },
        "canonical_mem_used_pct": {
            "AWS":   {"mem_used_percent": pct},
            "Azure": {"MemoryUsage": pct, "AverageMemoryWorkingSet": identity},
        },
        "canonical_runs_failed": {
            "Azure": {
                "RunsFailed": identity,
                "ActivityFailedRuns": identity,
                "PipelineFailedRuns": identity,
                "TriggerFailedRuns": identity,
            },
        },
        "canonical_runs_succeeded": {
            "Azure": {
                "RunsCompleted": identity,
                "ActivitySucceededRuns": identity,
                "PipelineSucceededRuns": identity,
            },
        },
        "canonical_run_failure_pct": {
            "Azure": {"RunFailurePercentage": pct},
        },
        "canonical_run_duration_ms": {
            "Azure": {"RunDuration": identity, "RunLatency": identity},
        },
        "canonical_group_instances": {
            "AWS": {
                "GroupInServiceInstances": identity,
                "GroupTotalInstances": identity,
                "GroupDesiredCapacity": identity,
            },
        },
        "canonical_group_capacity": {
            "AWS": {
                "GroupInServiceCapacity": identity,
                "GroupTotalCapacity": identity,
            },
        },
        "canonical_throttled_requests": {
            "AWS":   {"ThrottledRequests": identity},
            "Azure": {"ThrottledRequests": identity},
        },
        "canonical_status_check_failed": {
            "AWS": {
                "StatusCheckFailed": identity,
                "StatusCheckFailed_Instance": identity,
                "StatusCheckFailed_System": identity,
            },
        },
        "canonical_integration_queue_length": {
            "Azure": {"IntegrationRuntimeQueueLength": identity},
        },
        "canonical_resource_count": {
            "Azure": {"ResourceCount": identity},
        },
    },

    # ------------------------------------------------------------------
    # SECURITY  (Azure Key Vault + related)
    # ------------------------------------------------------------------
    "Security": {
        "canonical_api_hit": {
            "Azure": {"ServiceApiHit": identity},
        },
        "canonical_api_latency_ms": {
            "Azure": {"ServiceApiLatency": identity},
        },
        "canonical_api_result": {
            "Azure": {"ServiceApiResult": identity},
        },
        "canonical_saturation": {
            "Azure": {"SaturationShoebox": identity},
        },
        "canonical_availability_pct": {
            "Azure": {"Availability": pct},
        },
    },
}


def canonical_features(category):
    return list(REGISTRY.get(category, {}).keys())


PRIMARY_FORECAST_FEATURE = {
    "Compute":    "canonical_cpu_pct",
    "Network":    "canonical_active_connections",
    "Storage":    "canonical_storage_used_pct",
    "Container":  "canonical_node_cpu_pct",
    "Databases":  "canonical_db_cpu_pct",
    "Governance": "canonical_runs_failed",
    "Security":   "canonical_api_hit",
}


# BUG FIX: choose_forecast_feature() used to fall back to plain dict-insertion
# order from canonical_features(category) whenever the category's primary
# feature (above) wasn't present for a given resource. For Network, the
# primary feature (canonical_active_connections) only exists for AWS/Azure
# load balancers -- but ~96% of Network resources in this dataset are
# vpc_network/subnet resources that never emit it. Insertion order then
# picked raw, unbounded byte counters (canonical_net_in_bytes /
# canonical_net_out_bytes) ahead of canonical_subnet_util_pct, even though
# subnet IP capacity exhaustion -- a bounded 0-100% signal -- is exactly what
# the README advertises as the Network capability for that resource type.
# Forecasting an unbounded byte counter with no natural ceiling produces
# poor trend fits almost by construction.
#
# This priority list lets choose_forecast_feature() prefer bounded,
# semantically meaningful signals (health/utilization percentages) over
# unbounded raw counters, for any resource that doesn't have the category's
# primary feature. Categories not listed here keep the previous
# canonical_features() insertion-order behavior unchanged.
FORECAST_FEATURE_PRIORITY = {
    "Network": [
        "canonical_active_connections",
        "canonical_health_pct",
        "canonical_subnet_util_pct",
        "canonical_dns_queries",
        "canonical_throughput",
        "canonical_flow_log_count",
        "canonical_net_in_bytes",
        "canonical_net_out_bytes",
        "canonical_ddos_signal",
        "canonical_health_status",
        "canonical_probe_failure_count",
    ],
    "Storage": [
        "canonical_storage_used_pct",
        "canonical_burst_balance_pct",
        "availability_pct",
        "canonical_queue_depth",
        "canonical_storage_capacity_raw",
        "canonical_storage_read_bytes",
        "canonical_storage_write_bytes",
        "canonical_storage_read_iops",
        "canonical_storage_write_iops",
        "canonical_storage_latency_ms",
        "canonical_storage_transactions",
        "canonical_volume_idle_time",
        "canonical_permitted_throughput",
        "canonical_credit_balance_raw",
        "backup_health_event",
    ],
}


def forecast_feature_fallback_order(category):
    """Ordered list of canonical features to try for `category` when the
    primary forecast feature isn't available for a given resource. Uses the
    curated FORECAST_FEATURE_PRIORITY list where one exists (Network,
    Storage); otherwise falls back to plain registry insertion order, same
    as before."""
    if category in FORECAST_FEATURE_PRIORITY:
        return FORECAST_FEATURE_PRIORITY[category]
    return canonical_features(category)


def primary_metric_name(category, provider):
    feature = PRIMARY_FORECAST_FEATURE.get(category)
    return metric_name_for_feature(category, provider, feature)


def metric_name_for_feature(category, provider, feature):
    if not feature:
        return None
    provider_map = REGISTRY.get(category, {}).get(feature, {})
    raw_metrics = provider_map.get(provider, {})
    if raw_metrics:
        return next(iter(raw_metrics.keys()))
    return feature
