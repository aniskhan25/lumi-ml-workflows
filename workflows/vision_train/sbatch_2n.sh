#!/bin/bash
#SBATCH --job-name=vision-train-2n
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --partition=standard-g

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

source "$REPO_ROOT/slurm/env.sh"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((10000 + SLURM_JOB_ID % 50000))

SRUN_ARGS=()
if [[ -n "${CONTAINER_IMAGE:-}" ]]; then
  SRUN_ARGS+=(--container-image "$CONTAINER_IMAGE")
fi

srun --nodes=2 --ntasks-per-node=1 "${SRUN_ARGS[@]}" \
  torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank="$SLURM_NODEID" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    "$SCRIPT_DIR/train.py" \
    --config "$SCRIPT_DIR/config.yaml"
