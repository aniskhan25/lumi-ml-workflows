# Metrics

Core metrics reported by each workflow:
- `tokens_per_sec` (LLM train/infer)
- `samples_per_sec` (vision train/infer, future)
- `step_time_ms_avg`
- `step_time_ms_p95`
- `latency_ms_per_token` (LLM inference)
- `latency_ms_per_sample` (vision inference)
- `gpu_max_mem_gb`
- `host_mem_gb`

Communication metrics (DDP):
- `allreduce_time_ms_avg`
- `allreduce_time_ms_total`
- `allreduce_bytes_total`

When DDP is enabled, these are measured via a DDP communication hook on gradient buckets. If DDP is not enabled, the scripts can fall back to the synthetic allreduce timing.

System metadata:
- Container image path + digest
- Slurm job settings (nodes, tasks, GPUs)
- Hostnames, partition, ROCm version (best-effort)
