import argparse
import json
from pathlib import Path

DEFAULT_METRICS = [
    "tokens_per_sec",
    "samples_per_sec",
    "step_time_ms_avg",
    "step_time_ms_p95",
    "allreduce_time_ms_avg",
    "gpu_max_mem_gb",
]

DEFAULT_CONTEXT = [
    "nodes",
    "ntasks",
    "gpus_per_node",
    "cpus_per_task",
    "partition",
]

def load_reports(result_dir):
    reports = []
    for path in sorted(Path(result_dir).glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        data["_path"] = path
        reports.append(data)
    return reports


def format_value(value):
    if value is None:
        return "-"
    if isinstance(value, float):
        if abs(value) >= 1000:
            return f"{value:.2f}"
        if abs(value) >= 1:
            return f"{value:.3f}"
        return f"{value:.4f}"
    return str(value)


def report_row(report, metrics):
    workload = report.get("workload", {})
    metrics_data = report.get("metrics", {})
    slurm = report.get("slurm", {})
    system = report.get("system", {})
    row = {
        "workload": workload.get("type", "?"),
        "model": workload.get("model", "?"),
        "dtype": workload.get("dtype", "?"),
        "run_id": report.get("run_id", "?"),
        "nodes": format_value(slurm.get("nodes")),
        "ntasks": format_value(slurm.get("ntasks")),
        "gpus_per_node": format_value(slurm.get("gpus_per_node")),
        "cpus_per_task": format_value(slurm.get("cpus_per_task")),
        "partition": format_value(system.get("partition")),
    }
    for key in metrics:
        row[key] = format_value(metrics_data.get(key))
    return row


def to_markdown(rows, metrics):
    headers = ["workload", "model", "dtype", "run_id"] + DEFAULT_CONTEXT + metrics
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        line = "| " + " | ".join(str(row.get(h, "-")) for h in headers) + " |"
        lines.append(line)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Summarize benchmark results")
    parser.add_argument("--results", default="results/latest")
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    parser.add_argument("--write", default=None, help="Optional file to write markdown table")
    args = parser.parse_args()

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    reports = load_reports(args.results)
    rows = [report_row(r, metrics) for r in reports]

    table = to_markdown(rows, metrics)
    if args.write:
        Path(args.write).write_text(table)
    else:
        print(table)


if __name__ == "__main__":
    main()
