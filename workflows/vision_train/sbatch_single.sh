#!/bin/bash
#SBATCH --job-name=vision-train
#SBATCH --account=project_462000131
#SBATCH --partition=small-g
#SBATCH --chdir=/tmp
#SBATCH --output=/scratch/project_462000131/anisrahm/slurm/%x-%j.out
#SBATCH --error=/scratch/project_462000131/anisrahm/slurm/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=00:10:00

set -euo pipefail

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
    echo "REPO_ROOT not found and REPO_GIT_URL is empty. Set REPO_ROOT or REPO_GIT_URL." >&2
    exit 1
  fi
  export REPO_GIT_URL
  REPO_REF="${REPO_REF:-origin/main}"
  export REPO_REF
  if [[ -z "${RESULTS_DIR:-}" ]]; then
    export RESULTS_DIR="/project/project_462000131/anisrahm/lumi-ml-workflows/results"
  fi
  if [[ -z "${RESULTS_LATEST_ONLY:-}" ]]; then
    export RESULTS_LATEST_ONLY=1
  fi
  if [[ -z "${RESULTS_INCLUDE_NODES:-}" ]]; then
    export RESULTS_INCLUDE_NODES=1
  fi
  if command -v module >/dev/null 2>&1; then
    module use /appl/local/csc/modulefiles/
    module load singularity >/dev/null 2>&1 || true
    module load apptainer >/dev/null 2>&1 || true
    module load lumi-container-wrapper >/dev/null 2>&1 || true
    module load pytorch/2.7
  fi

  TMP_BASE="${SLURM_TMPDIR:-/tmp}"
  REPO_ROOT="$TMP_BASE/lumi-ml-workflows"
  if [[ ! -d "$REPO_ROOT/.git" ]]; then
    git clone "$REPO_GIT_URL" "$REPO_ROOT"
  fi
  git -C "$REPO_ROOT" fetch --all --prune
  git -C "$REPO_ROOT" reset --hard "${REPO_REF:-origin/main}"
else
  if [[ -d "$REPO_ROOT/slurm" ]]; then
    cd "$REPO_ROOT"
    source "$REPO_ROOT/slurm/env.sh"
  else
    if [[ -z "${RESULTS_DIR:-}" ]]; then
      export RESULTS_DIR="/project/project_462000131/anisrahm/lumi-ml-workflows/results"
    fi
    if command -v module >/dev/null 2>&1; then
      module use /appl/local/csc/modulefiles/
      module load singularity >/dev/null 2>&1 || true
      module load apptainer >/dev/null 2>&1 || true
      module load lumi-container-wrapper >/dev/null 2>&1 || true
      module load pytorch/2.7
    fi
  fi
fi

cd "$REPO_ROOT"

if [[ -f "$REPO_ROOT/slurm/env.sh" ]]; then
  source "$REPO_ROOT/slurm/env.sh"
fi

MASTER_ADDR=127.0.0.1
MASTER_PORT=$((10000 + SLURM_JOB_ID % 50000))

export SLURM_CPU_BIND=none
PYTHON_BIN="${PYTHON_BIN:-python}"
srun --cpu-bind=none --ntasks-per-node=1 \
  "$PYTHON_BIN" -m torch.distributed.run \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    "$REPO_ROOT/workflows/vision_train/train.py" \
    --config "$REPO_ROOT/workflows/vision_train/config.yaml"
