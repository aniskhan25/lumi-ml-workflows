# How to Run

## Local / single-node
```bash
python workflows/llm_train/train.py --config workflows/llm_train/config.yaml
python workflows/llm_infer/infer.py --config workflows/llm_infer/config.yaml
```

## LUMI (Slurm)
1. Set container path and any module loads in `slurm/env.sh`.
2. Submit jobs:
```bash
sbatch workflows/llm_train/sbatch_single.sh
sbatch workflows/llm_infer/sbatch_single.sh
sbatch workflows/vision_train/sbatch_single.sh
sbatch workflows/vision_infer/sbatch_single.sh
```

For multi-node runs, set `MASTER_ADDR`/`MASTER_PORT` in the job or use `torchrun` explicitly.

## Output
Each run writes a JSON report to `results/latest/` by default. The schema is in `results/schema.json`.
