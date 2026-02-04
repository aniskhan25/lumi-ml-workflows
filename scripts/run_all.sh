#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

python "$REPO_ROOT/workflows/llm_train/train.py" --config "$REPO_ROOT/workflows/llm_train/config.yaml"
python "$REPO_ROOT/workflows/llm_infer/infer.py" --config "$REPO_ROOT/workflows/llm_infer/config.yaml"
python "$REPO_ROOT/workflows/vision_train/train.py" --config "$REPO_ROOT/workflows/vision_train/config.yaml"
python "$REPO_ROOT/workflows/vision_infer/infer.py" --config "$REPO_ROOT/workflows/vision_infer/config.yaml"
