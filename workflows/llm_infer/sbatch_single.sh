#!/bin/bash
#SBATCH --job-name=llm-infer
#SBATCH --account=project_462000131
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

source "$REPO_ROOT/slurm/env.sh"

MASTER_ADDR=127.0.0.1
MASTER_PORT=$((10000 + SLURM_JOB_ID % 50000))

SRUN_ARGS=()
if [[ -n "${CONTAINER_IMAGE:-}" ]]; then
  SRUN_ARGS+=(--container-image "$CONTAINER_IMAGE")
fi

srun --ntasks-per-node=1 "${SRUN_ARGS[@]}" \
  torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    "$SCRIPT_DIR/infer.py" \
    --config "$SCRIPT_DIR/config.yaml"
