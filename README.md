# LUMI ML Workflows Benchmark Repo

Short, repeatable **pure PyTorch** workflows for validating container updates on **LUMI**. The MVP focuses on synthetic-data LLM training and inference with structured JSON results for fast comparisons.

## What’s in the MVP
- LLM training (single-node, DDP-ready)
- LLM inference throughput
- Vision training (single-node, DDP-ready)
- Vision inference throughput
- JSON results with system + Slurm metadata
- Slurm sbatch templates for LUMI

## Quick start (single node, local)
```bash
cd /Users/anisrahm/Documents/lumi-ml-workflows
python workflows/llm_train/train.py --config workflows/llm_train/config.yaml
python workflows/llm_infer/infer.py --config workflows/llm_infer/config.yaml
python workflows/vision_train/train.py --config workflows/vision_train/config.yaml
python workflows/vision_infer/infer.py --config workflows/vision_infer/config.yaml
```

Results are written to `results/latest/` by default.

## Slurm (LUMI)
1. Confirm `REPO_ROOT` (defaults to `/project/project_462000131/anisrahm/lumi-ml-workflows`) in `slurm/env.sh`.
2. Results default to `RESULTS_DIR=/project/project_462000131/anisrahm/lumi-ml-workflows/results`, `RESULTS_LATEST_ONLY=1`, and `RESULTS_INCLUDE_NODES=1` (writes `*_1n.json`, `*_2n.json`).
2. Submit a job:
```bash
sbatch workflows/llm_train/sbatch_single.sh
sbatch workflows/llm_infer/sbatch_single.sh
sbatch workflows/vision_train/sbatch_single.sh
sbatch workflows/vision_infer/sbatch_single.sh
```

## Slurm (2 nodes)
```bash
sbatch workflows/llm_train/sbatch_2n.sh
sbatch workflows/llm_infer/sbatch_2n.sh
sbatch workflows/vision_train/sbatch_2n.sh
sbatch workflows/vision_infer/sbatch_2n.sh
```

## Single vs multi-node experiments
Single-node runs are the **baseline** for correctness and per-node throughput. They use 1 node with 8 GPUs and 8 ranks (1 GPU per rank), which matches the per‑GPU CPU allocation of the multi-node runs.

Multi-node runs test **scale-out** across the network. They use 2 nodes with 8 GPUs each (16 total) and 16 ranks. Use these to compare scaling and communication overhead vs the single-node baseline.

Results use `RESULTS_INCLUDE_NODES=1`, so filenames include node count (e.g., `llm_train_1n.json`, `llm_train_2n.json`).

## Results summary
Generate a quick Markdown table (includes nodes/GPUs/CPUs/partition):
```bash
python scripts/summarize_results.py --results results/latest
```

Latest run summary:

| workload | model | dtype | run_id | nodes | ntasks | gpus_per_node | cpus_per_task | partition | tokens_per_sec | samples_per_sec | step_time_ms_avg | step_time_ms_p95 | allreduce_time_ms_avg | gpu_max_mem_gb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llm_infer | gpt_small | bf16 | 27f0c305-1022-4798-abc6-b9d23680fd2e | 1 | 1 | 8 | 7 | - | 2655864.45 | - | 9.253 | 9.296 | - | 1.084 |
| llm_infer | gpt_small | bf16 | 928d8d5d-ef14-4cc6-a913-20fb125be2a8 | 2 | 16 | 8 | 7 | - | 5350785.26 | - | 9.186 | 9.210 | - | 1.084 |
| llm_train | gpt_small | bf16 | e2e003bd-7585-462e-8996-929860710699 | 1 | 1 | 8 | 7 | - | 897740.81 | - | 73.001 | 73.984 | 0.1615 | 6.894 |
| llm_train | gpt_small | bf16 | 370d0b54-6a8f-4961-b883-285e5d35560c | 2 | 16 | 8 | 7 | - | 240296.26 | - | 545.460 | 581.622 | 0.1637 | 6.894 |
| vision_infer | resnet_small | bf16 | 00596983-5cdf-432c-ba87-3284da256afb | 1 | 1 | 8 | 7 | - | - | 204902.38 | 9.995 | 10.022 | - | 0.6849 |
| vision_infer | resnet_small | bf16 | 5618968b-468b-4629-9843-8593e086d80b | 2 | 16 | 8 | 7 | - | - | 405191.64 | 10.109 | 10.219 | - | 0.6849 |
| vision_train | resnet_small | bf16 | fd4d3af2-f4aa-4c14-bbb6-5f9613bbb336 | 1 | 1 | 8 | 7 | - | - | 34002.63 | 30.115 | 37.487 | 0.1972 | 0.8213 |
| vision_train | resnet_small | bf16 | ab708d7b-ba8d-4749-a428-1892916e6859 | 2 | 16 | 8 | 7 | - | - | 16171.67 | 126.641 | 133.157 | 0.1536 | 0.8174 |

## Repo layout
- `workflows/` – workload implementations and configs
- `slurm/` – common Slurm helpers and templates
- `scripts/` – helper scripts (compare, run all)
- `results/` – schema and output JSON files
- `docs/` – usage + metrics details
