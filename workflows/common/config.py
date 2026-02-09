import json
import os
import re
from pathlib import Path


def _parse_scalar(value):
    v = value.strip()
    if not v:
        return ""
    lower = v.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    if lower in {"null", "none"}:
        return None
    if re.fullmatch(r"[-+]?\d+", v):
        try:
            return int(v)
        except ValueError:
            return v
    if re.fullmatch(r"[-+]?\d*\.\d+(e[-+]?\d+)?", v) or re.fullmatch(
        r"[-+]?\d+e[-+]?\d+", v
    ):
        try:
            return float(v)
        except ValueError:
            return v
    if (v.startswith("\"") and v.endswith("\"")) or (
        v.startswith("'") and v.endswith("'")
    ):
        return v[1:-1]
    return v


def _simple_yaml_load(text):
    root = {}
    stack = [(0, root)]
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip("\n")
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent % 2 != 0:
            raise ValueError("Only 2-space indentation supported in simple YAML")
        key_val = line.strip()
        if ":" not in key_val:
            raise ValueError(f"Invalid YAML line: {raw}")
        key, val = key_val.split(":", 1)
        key = key.strip()
        val = val.strip()

        while stack and indent < stack[-1][0]:
            stack.pop()
        if not stack:
            raise ValueError("Malformed indentation")
        current = stack[-1][1]

        if val == "":
            new_map = {}
            current[key] = new_map
            stack.append((indent + 2, new_map))
        else:
            current[key] = _parse_scalar(val)
    return root


def _merge_dicts(base, override):
    for key, value in override.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            _merge_dicts(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path, defaults):
    if not path:
        return defaults
    cfg_path = Path(path)
    if not cfg_path.exists():
        return defaults

    text = cfg_path.read_text()
    data = None
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
    except Exception:
        try:
            data = _simple_yaml_load(text)
        except Exception:
            try:
                data = json.loads(text)
            except Exception:
                data = None

    if not isinstance(data, dict):
        return defaults

    merged = _merge_dicts(dict(defaults), data)
    return merged


def resolve_output_path(output_path, repo_root, workload_type, run_id):
    if output_path:
        path = Path(output_path)
        if path.is_dir():
            return path / f"{workload_type}_{run_id}.json"
        return path
    env_root = str_env("RESULTS_DIR")
    latest_only = str_env("RESULTS_LATEST_ONLY")
    use_latest_only = str(latest_only).lower() in {"1", "true", "yes", "on"}
    if env_root:
        base_dir = Path(env_root)
    else:
        base_dir = Path(repo_root) / "results" / "latest"
    base_dir.mkdir(parents=True, exist_ok=True)
    if use_latest_only:
        return base_dir / f"{workload_type}.json"
    return base_dir / f"{workload_type}_{run_id}.json"


def str_env(name, default=None):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value
