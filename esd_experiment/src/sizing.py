"""Initial conservative per-job budgets; empirical tuning belongs to validation."""
import math

GIB = 1024**3
BASE_TIMEOUTS = dict(load=7200, analyze=28800, fingerprint=28800, save=1800, default=14400)


def job_resources(weight_bytes, gpu_memory_bytes, pool_size, evidence_inflight_bytes):
    if weight_bytes <= 0 or gpu_memory_bytes <= 0 or pool_size < 1 or evidence_inflight_bytes < 0:
        raise ValueError("Sizing requires positive weight/GPU bytes and a nonempty pool")
    count = max(1, math.ceil(1.5 * weight_bytes / gpu_memory_bytes))
    if count > pool_size:
        return dict(oversized=True)
    gib = weight_bytes / GIB
    timeouts = dict(BASE_TIMEOUTS)
    timeouts["analyze"] = timeouts["fingerprint"] = max(28800, math.ceil(600 * gib))
    return dict(num_gpus=count, ram_limit_gib=max(16, .5*gib+4) + max(0, evidence_inflight_bytes/GIB-2),
                cache_limit_gib=1.1*gib+2, stage_timeout_seconds=timeouts)
