import os
import socket
from datetime import datetime, timezone
from pathlib import Path

import torch

from .config import str_env


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def get_hostnames():
    hostlist = str_env("SLURM_JOB_NODELIST") or str_env("SLURM_NODELIST")
    if not hostlist:
        return [socket.gethostname()]
    # Best-effort: leave as raw string if expanded list isn't available
    return [hostlist]


def get_rocm_version():
    env_version = str_env("ROCM_VERSION") or str_env("HSA_RUNTIME_VERSION")
    if env_version:
        return env_version
    torch_rocm = getattr(torch.version, "hip", None) or getattr(
        torch.version, "rocm", None
    )
    if torch_rocm:
        return str(torch_rocm)
    info_file = Path("/opt/rocm/.info/version")
    if info_file.exists():
        try:
            return info_file.read_text().strip()
        except Exception:
            return None
    return None


def gather_slurm_info():
    def _int_env(name):
        value = str_env(name)
        if value is None:
            return None
        try:
            return int(value)
        except ValueError:
            return None

    return {
        "nodes": _int_env("SLURM_JOB_NUM_NODES"),
        "ntasks": _int_env("SLURM_NTASKS"),
        "gpus_per_node": _int_env("SLURM_GPUS_ON_NODE"),
        "cpus_per_task": _int_env("SLURM_CPUS_PER_TASK"),
    }


def gather_system_info():
    return {
        "hostname_list": get_hostnames(),
        "partition": str_env("SLURM_PARTITION"),
        "rocm_version": get_rocm_version(),
    }
