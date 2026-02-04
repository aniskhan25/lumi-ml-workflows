import argparse
import os
import sys
import uuid
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from workflows.common.config import load_config, resolve_output_path, str_env
from workflows.common.distributed import barrier, init_distributed, is_main_process, sync_device
from workflows.common.ddp_comm import DDPCommStats, register_ddp_comm_hook
from workflows.common.metrics import mean, now_s, percentile
from workflows.common.report import write_report
from workflows.common.system import gather_slurm_info, gather_system_info, utc_now
from workflows.common.vision_model import ResNetSmall


def parse_args():
    parser = argparse.ArgumentParser(description="Vision training benchmark (synthetic)")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--output", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--container-image", default=str_env("CONTAINER_IMAGE"))
    parser.add_argument("--container-digest", default=str_env("CONTAINER_DIGEST"))
    return parser.parse_args()


def get_host_mem_gb():
    try:
        import psutil

        return psutil.virtual_memory().used / 1e9
    except Exception:
        return None


def _parse_csv_ints(value):
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return [int(v.strip()) for v in str(value).split(",") if v.strip()]


def main():
    args = parse_args()
    defaults = {
        "model": {
            "in_channels": 3,
            "num_classes": 1000,
            "layers": "2,2,2,2",
            "channels": "64,128,256,512",
            "stem_width": 64,
        },
        "train": {
            "image_size": 128,
            "batch_size": 128,
            "lr": 5e-4,
            "steps_warmup": 10,
            "steps_measure": 30,
            "grad_accum": 1,
            "dtype": "bf16",
            "measure_ddp_comm": True,
            "measure_allreduce": True,
            "allreduce_bytes": 1048576,
        },
        "output": {"directory": None},
    }
    config = load_config(args.config, defaults)

    dist_enabled = init_distributed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    model_cfg = config["model"]
    train_cfg = config["train"]

    layers = _parse_csv_ints(model_cfg.get("layers", "2,2,2,2"))
    channels = _parse_csv_ints(model_cfg.get("channels", "64,128,256,512"))

    model = ResNetSmall(
        in_channels=int(model_cfg.get("in_channels", 3)),
        num_classes=int(model_cfg.get("num_classes", 1000)),
        layers=layers,
        channels=channels,
        stem_width=int(model_cfg.get("stem_width", 64)),
    ).to(device)

    if dist_enabled:
        if torch.cuda.is_available():
            model = DDP(model, device_ids=[local_rank])
        else:
            model = DDP(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["lr"]))

    dtype = str(train_cfg.get("dtype", "bf16")).lower()
    amp_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
    use_amp = device.type == "cuda" and dtype in {"bf16", "fp16", "float16"}

    batch_size = int(train_cfg["batch_size"])
    image_size = int(train_cfg["image_size"])
    grad_accum = int(train_cfg.get("grad_accum", 1))
    warmup_steps = int(train_cfg["steps_warmup"])
    measure_steps = int(train_cfg["steps_measure"])
    total_steps = warmup_steps + measure_steps

    measure_allreduce = bool(train_cfg.get("measure_allreduce", True))
    allreduce_bytes = int(train_cfg.get("allreduce_bytes", 0))
    allreduce_tensor = None
    if dist_enabled and measure_allreduce and allreduce_bytes > 0:
        elem_size = torch.tensor(0.0).element_size()
        numel = max(1, allreduce_bytes // elem_size)
        allreduce_tensor = torch.ones(numel, device=device)

    step_times = []
    allreduce_times = []
    allreduce_bytes_total = 0
    ddp_comm_stats = None

    if dist_enabled and bool(train_cfg.get("measure_ddp_comm", True)):
        ddp_comm_stats = DDPCommStats(warmup_steps)
        register_ddp_comm_hook(model, ddp_comm_stats)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for step in range(total_steps):
        if ddp_comm_stats is not None:
            ddp_comm_stats.set_step(step)
        sync_device()
        start = now_s()
        optimizer.zero_grad(set_to_none=True)
        for micro_step in range(grad_accum):
            images = torch.randn(
                batch_size,
                int(model_cfg.get("in_channels", 3)),
                image_size,
                image_size,
                device=device,
            )
            labels = torch.randint(
                0,
                int(model_cfg.get("num_classes", 1000)),
                (batch_size,),
                device=device,
            )
            sync_ctx = nullcontext()
            if dist_enabled and grad_accum > 1 and hasattr(model, "no_sync"):
                if micro_step < grad_accum - 1:
                    sync_ctx = model.no_sync()

            with sync_ctx:
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    logits = model(images)
                    loss = F.cross_entropy(logits, labels)
                    loss = loss / grad_accum
                loss.backward()

        if dist_enabled and measure_allreduce and allreduce_tensor is not None and ddp_comm_stats is None:
            sync_device()
            ar_start = now_s()
            torch.distributed.all_reduce(allreduce_tensor)
            sync_device()
            ar_end = now_s()
            if step >= warmup_steps:
                allreduce_times.append((ar_end - ar_start) * 1000.0)
                allreduce_bytes_total += allreduce_tensor.numel() * allreduce_tensor.element_size()

        optimizer.step()
        sync_device()
        end = now_s()

        if step >= warmup_steps:
            step_times.append((end - start) * 1000.0)

    step_time_avg = mean(step_times)
    step_time_p95 = percentile(step_times, 95)
    samples_per_step = batch_size * grad_accum * world_size
    samples_per_sec = None
    if step_time_avg:
        samples_per_sec = samples_per_step / (step_time_avg / 1000.0)

    if ddp_comm_stats is not None:
        allreduce_times = ddp_comm_stats.times_ms
        allreduce_bytes_total = ddp_comm_stats.bytes_total

    gpu_max_mem_gb = None
    if torch.cuda.is_available():
        gpu_max_mem_gb = torch.cuda.max_memory_allocated() / 1e9

    report = {
        "schema_version": "1.0",
        "timestamp_utc": utc_now(),
        "run_id": args.run_id or str(uuid.uuid4()),
        "container": {
            "image_path": args.container_image,
            "image_digest": args.container_digest,
        },
        "slurm": gather_slurm_info(),
        "system": gather_system_info(),
        "workload": {
            "type": "vision_train",
            "model": "resnet_small",
            "dtype": dtype,
        },
        "metrics": {
            "samples_per_sec": samples_per_sec,
            "step_time_ms_avg": step_time_avg,
            "step_time_ms_p95": step_time_p95,
            "gpu_max_mem_gb": gpu_max_mem_gb,
            "host_mem_gb": get_host_mem_gb(),
            "allreduce_time_ms_avg": mean(allreduce_times),
            "allreduce_time_ms_total": sum(allreduce_times) if allreduce_times else 0.0,
            "allreduce_bytes_total": float(allreduce_bytes_total),
        },
    }

    barrier()
    if is_main_process():
        output_dir = config.get("output", {}).get("directory")
        output_path = resolve_output_path(
            args.output or output_dir, REPO_ROOT, "vision_train", report["run_id"]
        )
        write_report(output_path, report)
        print(f"Wrote report to {output_path}")


if __name__ == "__main__":
    main()
