# Configs

Workloads read YAML configs from their workflow directories.

## LLM train (`workflows/llm_train/config.yaml`)
Key sections:
- `model`: transformer sizes
- `train`: batch/seq length, step counts, dtype
- `output`: optional default output directory
Notes:
`train.measure_ddp_comm` enables DDP gradient allreduce timing. `train.measure_allreduce` enables synthetic allreduce timing (used when DDP is not active).

## LLM infer (`workflows/llm_infer/config.yaml`)
Key sections:
- `model`: transformer sizes
- `infer`: batch/prompt/decode lengths, step counts
- `output`: optional default output directory

## Vision train (`workflows/vision_train/config.yaml`)
Key sections:
- `model`: ResNet-like settings (layers, channels)
- `train`: batch/image size, step counts, dtype
- `output`: optional default output directory
Notes:
`train.measure_ddp_comm` enables DDP gradient allreduce timing. `train.measure_allreduce` enables synthetic allreduce timing (used when DDP is not active).

## Vision infer (`workflows/vision_infer/config.yaml`)
Key sections:
- `model`: ResNet-like settings (layers, channels)
- `infer`: batch/image size, step counts, dtype
- `output`: optional default output directory
