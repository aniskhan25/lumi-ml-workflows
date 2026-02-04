#!/bin/bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <result_a.json> <result_b.json>"
  exit 1
fi

python - <<'PY'
import json
import sys
from pathlib import Path

path_a = Path(sys.argv[1])
path_b = Path(sys.argv[2])

a = json.loads(path_a.read_text())
b = json.loads(path_b.read_text())

metrics_a = a.get("metrics", {})
metrics_b = b.get("metrics", {})

keys = sorted(set(metrics_a.keys()) | set(metrics_b.keys()))

print(f"Compare: {path_a.name} vs {path_b.name}")
for key in keys:
    va = metrics_a.get(key)
    vb = metrics_b.get(key)
    if va is None and vb is None:
        continue
    if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
        delta = vb - va
        pct = None
        if va != 0:
            pct = (delta / va) * 100.0
        if pct is None:
            print(f"{key:24s}: {va} -> {vb} (delta {delta:+.4g})")
        else:
            print(f"{key:24s}: {va} -> {vb} (delta {delta:+.4g}, {pct:+.2f}%)")
    else:
        print(f"{key:24s}: {va} -> {vb}")
PY
