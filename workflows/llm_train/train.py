import argparse
import os
import sys
import uuid
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

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
from workflows.common.ddp_comm import DDPCommStats, register_ddp_comm_hook
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
    parser = argparse.ArgumentParser(description="LLM training benchmark (synthetic)")
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
        "train": {
            "batch_size": 8,
            "seq_len": 1024,
            "lr": 3e-4,
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

    model = GPTSmall(
        vocab_size=int(model_cfg["vocab_size"]),
        n_layers=int(model_cfg["n_layers"]),
        n_heads=int(model_cfg["n_heads"]),
        d_model=int(model_cfg["d_model"]),
        d_ff=int(model_cfg["d_ff"]),
        max_seq_len=int(train_cfg["seq_len"]),
        dropout=float(model_cfg.get("dropout", 0.0)),
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
    seq_len = int(train_cfg["seq_len"])
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
            tokens = torch.randint(
                0, int(model_cfg["vocab_size"]), (batch_size, seq_len + 1), device=device
            )
            inputs = tokens[:, :-1]
            targets = tokens[:, 1:]
            sync_ctx = nullcontext()
            if dist_enabled and grad_accum > 1 and hasattr(model, "no_sync"):
                if micro_step < grad_accum - 1:
                    sync_ctx = model.no_sync()

            with sync_ctx:
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    logits = model(inputs)
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
                    )
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
    tokens_per_step = batch_size * seq_len * world_size
    tokens_per_sec = None
    if step_time_avg:
        tokens_per_sec = tokens_per_step / (step_time_avg / 1000.0)

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
            "type": "llm_train",
            "model": "gpt_small",
            "dtype": dtype,
        },
        "metrics": {
            "tokens_per_sec": tokens_per_sec,
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
            args.output or output_dir, REPO_ROOT, "llm_train", report["run_id"]
        )
        write_report(output_path, report)
        print(f"Wrote report to {output_path}")

    destroy_process_group()


if __name__ == "__main__":
    main()
