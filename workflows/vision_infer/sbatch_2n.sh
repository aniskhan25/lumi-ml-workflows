#!/bin/bash
#SBATCH --job-name=vision-infer-2n
#SBATCH --account=project_462000131
#SBATCH --partition=small-g
#SBATCH --output=/scratch/project_462000131/anisrahm/slurm/%x-%j.out
#SBATCH --error=/scratch/project_462000131/anisrahm/slurm/%x-%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=01:00:00

set -euo pipefail

export SLURM_CPU_BIND=none

if [[ "${DEBUG_ENV:-0}" == "1" ]]; then
  echo "DEBUG_ENV=1 (printing env from rank 0)"
  srun --cpu-bind=none --nodes=2 --ntasks-per-node=1 /bin/bash -lc 'env | egrep "LOCAL_RANK|SLURM_LOCALID|ROCR_VISIBLE_DEVICES|HIP_VISIBLE_DEVICES|CUDA_VISIBLE_DEVICES"'
fi

REPO_GIT_URL="${REPO_GIT_URL:-https://github.com/aniskhan25/lumi-ml-workflows.git}"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
USE_NODE_LOCAL=0

if [[ -z "$REPO_ROOT" ]]; then
  USE_NODE_LOCAL=1
elif [[ "$REPO_ROOT" == /tmp/* || "$REPO_ROOT" == /var/tmp/* ]]; then
  USE_NODE_LOCAL=1
fi

if [[ "$USE_NODE_LOCAL" -eq 1 ]]; then
  if [[ -z "$REPO_GIT_URL" ]]; then
    echo "REPO_GIT_URL is empty; cannot clone repo on nodes." >&2
    exit 1
  fi
  export REPO_GIT_URL
  if [[ -z "${RESULTS_DIR:-}" ]]; then
    export RESULTS_DIR="/project/project_462000131/anisrahm/lumi-ml-workflows/results"
  fi
  if command -v module >/dev/null 2>&1; then
    module use /appl/local/csc/modulefiles/
    module load pytorch/2.7
  fi

  export SLURM_CPU_BIND=none
  srun --cpu-bind=none --nodes=2 --ntasks-per-node=1 /bin/bash -lc '\
    set -euo pipefail; \
    REPO_ROOT_LOCAL="${SLURM_TMPDIR:-/tmp}/lumi-ml-workflows"; \
    if [[ ! -d "$REPO_ROOT_LOCAL/.git" ]]; then \
      git clone "$REPO_GIT_URL" "$REPO_ROOT_LOCAL"; \
    fi'

  srun --cpu-bind=none --nodes=2 --ntasks-per-node=8 /bin/bash -lc '\
    set -euo pipefail; \
    REPO_ROOT_LOCAL="${SLURM_TMPDIR:-/tmp}/lumi-ml-workflows"; \
    python "$REPO_ROOT_LOCAL/workflows/vision_infer/infer.py" \
      --config "$REPO_ROOT_LOCAL/workflows/vision_infer/config.yaml"'
  exit 0
fi

if [[ -d "$REPO_ROOT/slurm" ]]; then
  cd "$REPO_ROOT"
  source "$REPO_ROOT/slurm/env.sh"
else
  if [[ -z "${RESULTS_DIR:-}" ]]; then
    export RESULTS_DIR="/project/project_462000131/anisrahm/lumi-ml-workflows/results"
  fi
  if command -v module >/dev/null 2>&1; then
    module use /appl/local/csc/modulefiles/
    module load pytorch/2.7
  fi
fi

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((10000 + SLURM_JOB_ID % 50000))

export SLURM_CPU_BIND=none
srun --cpu-bind=none --nodes=2 --ntasks-per-node=8 \
  python "$REPO_ROOT/workflows/vision_infer/infer.py" \
  --config "$REPO_ROOT/workflows/vision_infer/config.yaml"
