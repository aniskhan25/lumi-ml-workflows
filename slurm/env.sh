#!/bin/bash

# Common Slurm environment setup for LUMI
DEFAULT_REPO_ROOT="/project/project_462000131/anisrahm/lumi-ml-workflows"

if [[ -z "${REPO_ROOT:-}" ]]; then
  export REPO_ROOT="$DEFAULT_REPO_ROOT"
fi

if [[ -z "${RESULTS_DIR:-}" ]]; then
  export RESULTS_DIR="/project/project_462000131/anisrahm/lumi-ml-workflows/results"
fi

if command -v module >/dev/null 2>&1; then
  module use /appl/local/csc/modulefiles/
  module load pytorch/2.7
fi

export PYTHONUNBUFFERED=1
