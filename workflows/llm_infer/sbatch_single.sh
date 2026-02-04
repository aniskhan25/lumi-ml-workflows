#!/bin/bash
#SBATCH --job-name=llm-infer
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --partition=standard-g

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

source "$REPO_ROOT/slurm/env.sh"

SRUN_ARGS=()
if [[ -n "${CONTAINER_IMAGE:-}" ]]; then
  SRUN_ARGS+=(--container-image "$CONTAINER_IMAGE")
fi

srun "${SRUN_ARGS[@]}" \
  python "$SCRIPT_DIR/infer.py" \
  --config "$SCRIPT_DIR/config.yaml"
