#!/bin/bash
#SBATCH --job-name=llm-train
#SBATCH --account=project_462000131
#SBATCH --partition=small-g
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

if [[ -z "$REPO_ROOT" || ! -d "$REPO_ROOT/slurm" ]]; then
  if [[ -z "$REPO_GIT_URL" ]]; then
    echo "REPO_ROOT not found and REPO_GIT_URL is empty. Set REPO_ROOT or REPO_GIT_URL." >&2
    exit 1
  fi
  TMP_BASE="${SLURM_TMPDIR:-/tmp}"
  REPO_ROOT="$TMP_BASE/lumi-ml-workflows"
  if [[ ! -d "$REPO_ROOT/.git" ]]; then
    git clone "$REPO_GIT_URL" "$REPO_ROOT"
  fi
fi

cd "$REPO_ROOT"

source "$REPO_ROOT/slurm/env.sh"

MASTER_ADDR=127.0.0.1
MASTER_PORT=$((10000 + SLURM_JOB_ID % 50000))

SRUN_SUPPORTS_CONTAINER=0
if srun --help 2>&1 | grep -q -- '--container-image'; then
  SRUN_SUPPORTS_CONTAINER=1
fi

SRUN_ARGS=()
if [[ -n "${CONTAINER_IMAGE:-}" && "$SRUN_SUPPORTS_CONTAINER" -eq 1 ]]; then
  SRUN_ARGS+=(--container-image "$CONTAINER_IMAGE")
elif [[ -n "${CONTAINER_IMAGE:-}" ]]; then
  echo "srun does not support --container-image; ignoring CONTAINER_IMAGE" >&2
fi

srun --ntasks-per-node=1 "${SRUN_ARGS[@]}" \
  torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    "$REPO_ROOT/workflows/llm_train/train.py" \
    --config "$REPO_ROOT/workflows/llm_train/config.yaml"
