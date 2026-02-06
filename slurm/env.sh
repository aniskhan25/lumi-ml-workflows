#!/bin/bash

# Common Slurm environment setup for LUMI
# Default container path can be overridden by setting CONTAINER_IMAGE.
DEFAULT_CONTAINER="/appl/local/csc/soft/ai/images/pytorch_2.7.1_lumi.sif"
DEFAULT_REPO_ROOT="/project/project_462000131/anisrahm/lumi-ml-workflows"

if [[ -z "${CONTAINER_IMAGE:-}" ]]; then
  export CONTAINER_IMAGE="$DEFAULT_CONTAINER"
fi

if [[ -z "${REPO_ROOT:-}" ]]; then
  export REPO_ROOT="$DEFAULT_REPO_ROOT"
fi

if command -v module >/dev/null 2>&1; then
  module use /appl/local/csc/modulefiles/
  module load pytorch/2.7
fi

export PYTHONUNBUFFERED=1
