# LUMI ML Workflows Benchmark Repo — Implementation Plan

## 1. Purpose and Goals

We need a dedicated repository for **end‑to‑end ML workflows** on **LUMI** using **pure PyTorch**. The repo will validate container updates against **short, repeatable training/inference workloads** for:

- **LLM training**
- **LLM inference**
- **Vision training**
- **Vision inference**

**Primary goals**
1. Provide **fast validation** (1–5 minute jobs) for container updates.
2. Produce **structured JSON results** for comparisons across containers and time.
3. Support **single‑node and 2‑node** workloads.
4. Avoid framework dependencies beyond **PyTorch**.

---

## 2. Scope (MVP)

**In scope**
- Synthetic data workflows (no external datasets required).
- Short, fixed‑step runs with warmup.
- Single‑node and 2‑node DDP training jobs.
- Single‑node inference throughput tests.

**Out of scope (for MVP)**
- Real datasets (add later if needed).
- Model‑parallel / pipeline‑parallel.
- Long training runs.

---

## 3. Metrics Required (HPC‑Relevant)

**Core metrics**
- `tokens_per_sec` (LLM train/infer)
- `samples_per_sec` (Vision train/infer)
- `step_time_ms_avg` and `step_time_ms_p95`
- `gpu_max_mem_gb`
- `host_mem_gb`

**Communication metrics**
- `allreduce_time_ms_avg`
- `allreduce_time_ms_total`
- `allreduce_bytes_total`

**System metadata**
- Container path/digest
- Slurm info (nodes, tasks, GPUs)
- Hostnames, partition, ROCm version

---

## 4. Repository Structure (Proposed)

```
lumi-ml-workflows/
  README.md
  docs/
    how_to_run.md
    metrics.md
    configs.md
  workflows/
    llm_train/
      train.py
      config.yaml
      sbatch_single.sh
      sbatch_2n.sh
    llm_infer/
      infer.py
      config.yaml
      sbatch_single.sh
    vision_train/
      train.py
      config.yaml
      sbatch_single.sh
      sbatch_2n.sh
    vision_infer/
      infer.py
      config.yaml
      sbatch_single.sh
  slurm/
    env.sh
    single_8g_8r.sh
    single_8g_16r.sh
    multi_2n_8rpn.sh
  scripts/
    run_all.sh
    compare_results.sh
  results/
    schema.json
    latest/
```

---

## 5. Workflow Design (Pure PyTorch)

### 5.1 LLM Training (DDP)
- **Model**: small Transformer/GPT‑like (pure PyTorch)
- **Data**: synthetic tokens
- **Seq length**: 1024 (tunable)
- **Steps**: warmup 10, measure 30
- **Outputs**: tokens/sec, step time, comm time, memory

### 5.2 LLM Inference
- **Model**: same as training
- **Batch size**: 8–16
- **Prompt len**: 128, decode len: 256
- **Steps**: 20–50
- **Outputs**: tokens/sec, latency per token

### 5.3 Vision Training (DDP)
- **Model**: ResNet‑50 or ViT‑B/16 (pure PyTorch)
- **Data**: synthetic images
- **Batch size**: 64–128
- **Steps**: warmup 10, measure 30
- **Outputs**: samples/sec, step time, memory

### 5.4 Vision Inference
- **Model**: same as training
- **Batch size**: 256
- **Steps**: 50
- **Outputs**: samples/sec, latency

---

## 6. Results Format (JSON Schema)

Each workflow emits a JSON report per run:

```json
{
  "schema_version": "1.0",
  "timestamp_utc": "...",
  "run_id": "...",
  "container": { "image_path": "...", "image_digest": "..." },
  "slurm": { "nodes": 2, "ntasks": 16, "gpus_per_node": 8, "cpus_per_task": 1 },
  "system": { "hostname_list": ["nid..."], "partition": "standard-g", "rocm_version": "..." },
  "workload": { "type": "llm_train", "model": "gpt_small", "dtype": "bf16" },
  "metrics": {
    "tokens_per_sec": 12345.6,
    "step_time_ms_avg": 12.3,
    "step_time_ms_p95": 14.2,
    "allreduce_time_ms_avg": 3.1,
    "gpu_max_mem_gb": 42.1
  }
}
```

---

## 7. Slurm Execution

- Provide **single‑node** and **2‑node** sbatch scripts.
- Use `torchrun --nproc_per_node=8`.
- For 2‑node: set `MASTER_ADDR` from Slurm nodelist and pick a unique `MASTER_PORT`.

---

## 8. Comparison and Reporting

- `scripts/compare_results.sh` compares metrics across two containers.
- Auto‑generate a **README summary table** after each run.
- Store latest outputs in `results/latest/`.

---

## 9. Risks and Mitigations

**Risk**: variability in short runs
- **Mitigation**: fixed step counts + warmup + short averaging window.

**Risk**: container/environment drift
- **Mitigation**: embed full system metadata in JSON.

**Risk**: missing comm timing
- **Mitigation**: instrument `torch.distributed` and log comm stats.

---

## 10. Milestones

**Milestone 1 (Week 1): Repo + LLM workflows**
- Scaffold repo structure.
- Implement LLM train + infer (single‑node).
- Emit JSON results matching schema.

**Milestone 2 (Week 2): Vision + multi‑node**
- Add vision train + infer.
- Add 2‑node DDP for LLM + vision.
- Add comm timing metrics.

**Milestone 3 (Week 3): Reporting**
- Add comparison script.
- Auto‑render README summary tables.
- Add example `results/latest` data.

---

## 11. Execution Readiness

To run on LUMI, the team needs:
- LUMI account + partition access
- Container paths (old/new)
- `torchrun` available inside the container
- 1–2 nodes allocation

---

## 12. Team Decisions Required

1. Confirm **model choice** (Transformer/GPT for LLM, ResNet/ViT for vision).
2. Confirm **step counts** to keep jobs within 1–5 minutes.
3. Confirm **priority metrics** for go/no‑go decisions.
