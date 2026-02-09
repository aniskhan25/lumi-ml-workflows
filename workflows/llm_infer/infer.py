import argparse
import os
import sys
import uuid
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from workflows.common.config import load_config, resolve_output_path, str_env
from workflows.common.distributed import (
    barrier,
    destroy_process_group,
    init_distributed,
    is_main_process,
    sync_device,
)
from workflows.common.metrics import mean, now_s, percentile
from workflows.common.report import write_report
from workflows.common.system import gather_slurm_info, gather_system_info, utc_now


class GPTSmall(nn.Module):
    def __init__(self, vocab_size, n_layers, n_heads, d_model, d_ff, max_seq_len, dropout):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        self.register_buffer("attn_mask", mask, persistent=False)

    def forward(self, tokens):
        bsz, seq_len = tokens.shape
        positions = torch.arange(0, seq_len, device=tokens.device).unsqueeze(0)
        x = self.token_emb(tokens) + self.pos_emb(positions)
        x = self.drop(x)
        attn_mask = self.attn_mask[:seq_len, :seq_len]
        x = self.encoder(x, mask=attn_mask)
        return self.lm_head(x)


def parse_args():
    parser = argparse.ArgumentParser(description="LLM inference benchmark (synthetic)")
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


def main():
    args = parse_args()
    defaults = {
        "model": {
            "vocab_size": 50304,
            "n_layers": 6,
            "n_heads": 8,
            "d_model": 512,
            "d_ff": 2048,
            "dropout": 0.0,
        },
        "infer": {
            "batch_size": 8,
            "prompt_len": 128,
            "decode_len": 256,
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

    total_len = int(infer_cfg["prompt_len"]) + int(infer_cfg["decode_len"])
    model = GPTSmall(
        vocab_size=int(model_cfg["vocab_size"]),
        n_layers=int(model_cfg["n_layers"]),
        n_heads=int(model_cfg["n_heads"]),
        d_model=int(model_cfg["d_model"]),
        d_ff=int(model_cfg["d_ff"]),
        max_seq_len=total_len,
        dropout=float(model_cfg.get("dropout", 0.0)),
    ).to(device)
    model.eval()

    dtype = str(infer_cfg.get("dtype", "bf16")).lower()
    amp_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
    use_amp = device.type == "cuda" and dtype in {"bf16", "fp16", "float16"}

    batch_size = int(infer_cfg["batch_size"])
    warmup_steps = int(infer_cfg["steps_warmup"])
    measure_steps = int(infer_cfg["steps_measure"])
    total_steps = warmup_steps + measure_steps

    step_times = []

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        for step in range(total_steps):
            tokens = torch.randint(
                0, int(model_cfg["vocab_size"]), (batch_size, total_len), device=device
            )
            sync_device()
            start = now_s()
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                _ = model(tokens)
            sync_device()
            end = now_s()
            if step >= warmup_steps:
                step_times.append((end - start) * 1000.0)

    step_time_avg = mean(step_times)
    step_time_p95 = percentile(step_times, 95)
    tokens_per_step = batch_size * total_len * world_size
    tokens_per_sec = None
    latency_ms_per_token = None
    if step_time_avg:
        tokens_per_sec = tokens_per_step / (step_time_avg / 1000.0)
        latency_ms_per_token = step_time_avg / total_len

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
            "type": "llm_infer",
            "model": "gpt_small",
            "dtype": dtype,
        },
        "metrics": {
            "tokens_per_sec": tokens_per_sec,
            "step_time_ms_avg": step_time_avg,
            "step_time_ms_p95": step_time_p95,
            "gpu_max_mem_gb": gpu_max_mem_gb,
            "host_mem_gb": get_host_mem_gb(),
            "latency_ms_per_token": latency_ms_per_token,
        },
    }

    barrier()
    if is_main_process():
        output_dir = config.get("output", {}).get("directory")
        output_path = resolve_output_path(
            args.output or output_dir, REPO_ROOT, "llm_infer", report["run_id"]
        )
        write_report(output_path, report)
        print(f"Wrote report to {output_path}")

    destroy_process_group()


if __name__ == "__main__":
    main()
