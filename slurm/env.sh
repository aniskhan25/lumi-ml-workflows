#!/bin/bash

# Common Slurm environment setup for LUMI
DEFAULT_REPO_ROOT="/project/project_462000131/anisrahm/lumi-ml-workflows"

if [[ -z "${REPO_ROOT:-}" ]]; then
  export REPO_ROOT="$DEFAULT_REPO_ROOT"
fi

if [[ -z "${RESULTS_DIR:-}" ]]; then
  export RESULTS_DIR="/project/project_462000131/anisrahm/lumi-ml-workflows/results"
fi

if [[ -z "${RESULTS_LATEST_ONLY:-}" ]]; then
  export RESULTS_LATEST_ONLY=1
fi

if [[ -z "${RESULTS_INCLUDE_NODES:-}" ]]; then
  export RESULTS_INCLUDE_NODES=1
fi

if ! command -v module >/dev/null 2>&1; then
  if [[ -f /etc/profile.d/modules.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh
  elif [[ -f /usr/share/Modules/init/bash ]]; then
    # shellcheck disable=SC1091
    source /usr/share/Modules/init/bash
  elif [[ -f /etc/profile ]]; then
    # shellcheck disable=SC1091
    source /etc/profile
  fi
fi

if command -v module >/dev/null 2>&1; then
  module use /appl/local/csc/modulefiles/
  module load singularity >/dev/null 2>&1 || true
  module load apptainer >/dev/null 2>&1 || true
  module load lumi-container-wrapper >/dev/null 2>&1 || true
  module load pytorch/2.7
fi

if command -v python >/dev/null 2>&1; then
  export PYTHON_BIN="$(command -v python)"
elif command -v python3 >/dev/null 2>&1; then
  export PYTHON_BIN="$(command -v python3)"
fi

export PYTHONUNBUFFERED=1
