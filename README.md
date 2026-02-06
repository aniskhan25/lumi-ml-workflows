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
1. Confirm `REPO_ROOT` (defaults to `/tmp/lumi-ml-workflows`) and container image in `slurm/env.sh`.
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

## Results summary
Generate a quick Markdown table from `results/latest/`:
```bash
python scripts/summarize_results.py --results results/latest
```

## Repo layout
- `workflows/` – workload implementations and configs
- `slurm/` – common Slurm helpers and templates
- `scripts/` – helper scripts (compare, run all)
- `results/` – schema and output JSON files
- `docs/` – usage + metrics details
