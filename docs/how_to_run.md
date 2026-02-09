# How to Run

## Local / single-node
```bash
python workflows/llm_train/train.py --config workflows/llm_train/config.yaml
python workflows/llm_infer/infer.py --config workflows/llm_infer/config.yaml
```

## LUMI (Slurm)
1. Confirm `REPO_ROOT` (defaults to `/project/project_462000131/anisrahm/lumi-ml-workflows`) and container path in `slurm/env.sh`.
1. Results default to `RESULTS_DIR=/project/project_462000131/anisrahm/lumi-ml-workflows/results`, `RESULTS_LATEST_ONLY=1`, and `RESULTS_INCLUDE_NODES=1` (writes `*_1n.json`, `*_2n.json`).
1. Submit jobs:
```bash
sbatch workflows/llm_train/sbatch_single.sh
sbatch workflows/llm_infer/sbatch_single.sh
sbatch workflows/vision_train/sbatch_single.sh
sbatch workflows/vision_infer/sbatch_single.sh
```

## LUMI (Slurm, 2 nodes)
```bash
sbatch workflows/llm_train/sbatch_2n.sh
sbatch workflows/llm_infer/sbatch_2n.sh
sbatch workflows/vision_train/sbatch_2n.sh
sbatch workflows/vision_infer/sbatch_2n.sh
```

Multi-node runs use `torchrun` with `MASTER_ADDR` set from the Slurm nodelist.

## Output
Each run writes a JSON report to `results/latest/` by default. The schema is in `results/schema.json`.
