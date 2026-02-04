import argparse
import os
import sys
import uuid
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from workflows.common.config import load_config, resolve_output_path, str_env
from workflows.common.distributed import barrier, init_distributed, is_main_process, sync_device
from workflows.common.metrics import mean, now_s, percentile
from workflows.common.report import write_report
from workflows.common.system import gather_slurm_info, gather_system_info, utc_now
from workflows.common.vision_model import ResNetSmall


def parse_args():
    parser = argparse.ArgumentParser(description="Vision inference benchmark (synthetic)")
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
        "infer": {
            "image_size": 128,
            "batch_size": 256,
            "steps_warmup": 10,
            "steps_measure": 30,
            "dtype": "bf16",
        },
        "output": {"directory": None},
    }
    config = load_config(args.config, defaults)

    init_distributed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    model_cfg = config["model"]
    infer_cfg = config["infer"]

    layers = _parse_csv_ints(model_cfg.get("layers", "2,2,2,2"))
    channels = _parse_csv_ints(model_cfg.get("channels", "64,128,256,512"))

    model = ResNetSmall(
        in_channels=int(model_cfg.get("in_channels", 3)),
        num_classes=int(model_cfg.get("num_classes", 1000)),
        layers=layers,
        channels=channels,
        stem_width=int(model_cfg.get("stem_width", 64)),
    ).to(device)
    model.eval()

    dtype = str(infer_cfg.get("dtype", "bf16")).lower()
    amp_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
    use_amp = device.type == "cuda" and dtype in {"bf16", "fp16", "float16"}

    batch_size = int(infer_cfg["batch_size"])
    image_size = int(infer_cfg["image_size"])
    warmup_steps = int(infer_cfg["steps_warmup"])
    measure_steps = int(infer_cfg["steps_measure"])
    total_steps = warmup_steps + measure_steps

    step_times = []

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        for step in range(total_steps):
            images = torch.randn(
                batch_size,
                int(model_cfg.get("in_channels", 3)),
                image_size,
                image_size,
                device=device,
            )
            sync_device()
            start = now_s()
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                _ = model(images)
            sync_device()
            end = now_s()
            if step >= warmup_steps:
                step_times.append((end - start) * 1000.0)

    step_time_avg = mean(step_times)
    step_time_p95 = percentile(step_times, 95)
    samples_per_step = batch_size * world_size
    samples_per_sec = None
    latency_ms_per_sample = None
    if step_time_avg:
        samples_per_sec = samples_per_step / (step_time_avg / 1000.0)
        latency_ms_per_sample = step_time_avg / batch_size

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
            "type": "vision_infer",
            "model": "resnet_small",
            "dtype": dtype,
        },
        "metrics": {
            "samples_per_sec": samples_per_sec,
            "step_time_ms_avg": step_time_avg,
            "step_time_ms_p95": step_time_p95,
            "gpu_max_mem_gb": gpu_max_mem_gb,
            "host_mem_gb": get_host_mem_gb(),
            "latency_ms_per_sample": latency_ms_per_sample,
        },
    }

    barrier()
    if is_main_process():
        output_dir = config.get("output", {}).get("directory")
        output_path = resolve_output_path(
            args.output or output_dir, REPO_ROOT, "vision_infer", report["run_id"]
        )
        write_report(output_path, report)
        print(f"Wrote report to {output_path}")


if __name__ == "__main__":
    main()
