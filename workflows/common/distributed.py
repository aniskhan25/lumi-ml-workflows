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


def _visible_device_count():
    for name in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        value = os.environ.get(name)
        if value is None:
            continue
        if value.strip() == "":
            return 0
        return len([v for v in value.split(",") if v.strip() != ""])
    return None


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
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if device_count <= 0:
            raise RuntimeError(
                "No visible GPUs detected. Check ROCR_VISIBLE_DEVICES/HIP_VISIBLE_DEVICES/CUDA_VISIBLE_DEVICES."
            )
        visible_count = _visible_device_count()
        if visible_count is None or visible_count <= 0:
            visible_count = device_count
        local_rank = local_rank % visible_count
        if local_rank >= device_count:
            local_rank = local_rank % device_count
        os.environ["LOCAL_RANK"] = str(local_rank)
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend=backend, init_method="env://", device_id=local_rank
        )
    else:
        dist.init_process_group(backend=backend, init_method="env://")
    return True


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def sync_device():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def barrier():
    if dist.is_initialized():
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            dist.barrier(device_ids=[local_rank])
        else:
            dist.barrier()


def destroy_process_group():
    if dist.is_initialized():
        dist.destroy_process_group()
