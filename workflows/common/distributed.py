import os

import torch
import torch.distributed as dist


def _env_int(name, default=None):
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _maybe_set_env(name, value):
    if value is None:
        return
    if os.environ.get(name) is None:
        os.environ[name] = str(value)


def init_distributed():
    world_size = _env_int("WORLD_SIZE")
    if world_size is None:
        world_size = _env_int("SLURM_NTASKS", 1)
        _maybe_set_env("WORLD_SIZE", world_size)

    if world_size <= 1:
        return False

    _maybe_set_env("RANK", _env_int("SLURM_PROCID", 0))
    _maybe_set_env("LOCAL_RANK", _env_int("SLURM_LOCALID", 0))

    if os.environ.get("MASTER_ADDR") is None:
        # Single-node default; for multi-node, set MASTER_ADDR externally or use torchrun.
        _maybe_set_env("MASTER_ADDR", "127.0.0.1")
    if os.environ.get("MASTER_PORT") is None:
        _maybe_set_env("MASTER_PORT", 29500)

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return True


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def sync_device():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def barrier():
    if dist.is_initialized():
        dist.barrier()
