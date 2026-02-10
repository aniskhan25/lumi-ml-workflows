#!/bin/bash
#SBATCH --job-name=llm-train-2n
#SBATCH --account=project_462000131
#SBATCH --partition=small-g
#SBATCH --chdir=/tmp
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

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((10000 + SLURM_JOB_ID % 50000))
export MASTER_ADDR MASTER_PORT

SRUN_BASE=(srun --export=ALL --cpu-bind=none)

REPO_GIT_URL="${REPO_GIT_URL:-https://github.com/aniskhan25/lumi-ml-workflows.git}"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
REPO_ROOT_LOCAL="${SLURM_TMPDIR:-/tmp}/lumi-ml-workflows"
export REPO_ROOT_LOCAL

if [[ -z "$REPO_ROOT" || "$REPO_ROOT" == /tmp/* || "$REPO_ROOT" == /var/tmp/* ]]; then
  if [[ -z "$REPO_GIT_URL" ]]; then
    echo "REPO_GIT_URL is empty; cannot clone repo on nodes." >&2
    exit 1
  fi
  export REPO_GIT_URL
  REPO_REF="${REPO_REF:-origin/main}"
  export REPO_REF

  "${SRUN_BASE[@]}" --nodes=2 --ntasks=2 --ntasks-per-node=1 /bin/bash -c '\
    set -euo pipefail; \
    if [[ ! -d "$REPO_ROOT_LOCAL/.git" ]]; then \
      git clone "$REPO_GIT_URL" "$REPO_ROOT_LOCAL"; \
    fi; \
    git -C "$REPO_ROOT_LOCAL" fetch --all --prune; \
    git -C "$REPO_ROOT_LOCAL" reset --hard "$REPO_REF"'

  "${SRUN_BASE[@]}" --nodes=2 --ntasks=16 --ntasks-per-node=8 /bin/bash -c '\
    set -euo pipefail; \
    export REPO_ROOT="$REPO_ROOT_LOCAL"; \
    source "$REPO_ROOT_LOCAL/slurm/env.sh"; \
    "$PYTHON_BIN" "$REPO_ROOT_LOCAL/workflows/llm_train/train.py" \
      --config "$REPO_ROOT_LOCAL/workflows/llm_train/config.yaml"'
  exit 0
fi

if [[ -d "$REPO_ROOT/slurm" ]]; then
  cd "$REPO_ROOT"
  source "$REPO_ROOT/slurm/env.sh"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
"${SRUN_BASE[@]}" --nodes=2 --ntasks=16 --ntasks-per-node=8 \
  "$PYTHON_BIN" "$REPO_ROOT/workflows/llm_train/train.py" \
  --config "$REPO_ROOT/workflows/llm_train/config.yaml"
