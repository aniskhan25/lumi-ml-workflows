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

if [[ -z "$REPO_ROOT" || "$REPO_ROOT" == /tmp/* || "$REPO_ROOT" == /var/tmp/* || ! -d "$REPO_ROOT/slurm" ]]; then
  if [[ -z "$REPO_GIT_URL" ]]; then
    echo "REPO_ROOT not found and REPO_GIT_URL is empty. Set REPO_ROOT or REPO_GIT_URL." >&2
    exit 1
  fi
  TMP_BASE="${SLURM_TMPDIR:-/tmp}"
  REPO_ROOT="$TMP_BASE/lumi-ml-workflows"
  if [[ ! -d "$REPO_ROOT/.git" ]]; then
    git clone "$REPO_GIT_URL" "$REPO_ROOT"
  fi
  git -C "$REPO_ROOT" fetch --all --prune
  git -C "$REPO_ROOT" reset --hard "${REPO_REF:-origin/main}"
fi

cd "$REPO_ROOT"
source "$REPO_ROOT/slurm/env.sh"

MASTER_ADDR=127.0.0.1
MASTER_PORT=$((10000 + SLURM_JOB_ID % 50000))

PYTHON_BIN="${PYTHON_BIN:-python}"
srun --export=ALL --cpu-bind=none \
  "$PYTHON_BIN" -m torch.distributed.run \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    "$REPO_ROOT/workflows/vision_train/train.py" \
    --config "$REPO_ROOT/workflows/vision_train/config.yaml"
