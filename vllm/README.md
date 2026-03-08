# DTS — vLLM Backend

A vLLM-based implementation of Decoding Tree Sketching, optimized for multi-GPU cluster environments.

## Overview

This backend replaces the HuggingFace Transformers inference with [vLLM](https://github.com/vllm-project/vllm), using a **two-phase architecture**:

- **Phase 1 (Branching):** Generate one token at a time. At each step, compute entropy and varentropy from the model's top-k logprobs. Branch when the branching condition is met. Once the number of active hypotheses reaches `max_active_hyps`, freeze — no further branching.

- **Phase 2 (Completion):** After freeze, complete all remaining hypotheses in a single batched vLLM `generate()` call, leveraging vLLM's automatic prefix caching (APC) for shared prefixes.

### Decoding Modes

- **Greedy** (`--num_traces 1`): Return the first trace to finish.
- **Stable** (`--num_traces 8`): Collect multiple traces and use majority vote on extracted answers.

## Installation

```bash
# From the repository root
cd vllm
pip install -e .
```

**Requirements:** Python ≥ 3.10, vLLM ≥ 0.8, PyTorch ≥ 2.4

## Usage

All commands should be run from the `vllm/` directory.

### DTS Greedy (1 trace)
```bash
python run_benchmark.py dts \
    --model_name 1.5B --dataset_name aime24 \
    -e 2.5 -k 3 -a 48 -m 32768 -t 0.6 \
    -s 0 -n 5 --num_traces 1 --enforce_eager
```

### DTS Stable (8 traces, majority vote)
```bash
python run_benchmark.py dts \
    --model_name 1.5B --dataset_name aime24 \
    -e 2.5 -k 3 -a 48 -m 32768 -t 0.6 \
    -s 0 -n 5 --num_traces 8 --enforce_eager
```

### Standard Baseline
```bash
python run_benchmark.py standard \
    --model_name 1.5B --dataset_name aime24 \
    -t 0.6 -s 0 -n 5
```

### SLURM (GPU Cluster)
```bash
sbatch scripts/run_greedy.slurm
sbatch scripts/run_stable.slurm
```

## Parameters

| Parameter | Flag | Default | Description |
|-----------|------|---------|-------------|
| Entropy threshold | `-e` | 2.5 | Branch when H ≤ this |
| Varentropy threshold | `-v` | 1.5 | Branch when V > this |
| Branch top-k | `-k` | 3 | Children per branch point |
| Max active hypotheses | `-a` | 48 | Freeze threshold |
| Max new tokens | `-m` | 32768 | Per-hypothesis token limit |
| Temperature | `-t` | 0.6 | Sampling temperature |
| Num traces | `--num_traces` | 8 | 1 for Greedy, 8 for Stable |
| Trials | `-n` | 1 | Number of seeds to run |
| Seed | `-s` | 0 | Starting random seed |

## Results (5 seeds, A100-SXM4-80GB)

### DeepSeek-R1-Distill-Qwen-1.5B

#### AIME 2024 (temp=0.6)

| Method | Accuracy | Paper |
|--------|----------|-------|
| Standard | 28.67% | 26.67% |
| DTS-Greedy (1 trace) | 53.33% | 54.67% |
| DTS-Stable (8 traces) | **67.33%** | 64.67% |

**DTS-Greedy per-seed:**

| Seed | Accuracy | Avg Tokens |
|------|----------|------------|
| 0 | 50.00% | 5755 |
| 1 | 60.00% | 5551 |
| 2 | 56.67% | 5296 |
| 3 | 53.33% | 5708 |
| 4 | 46.67% | 5193 |

Avg wall time per seed: 11647s (194 min). Average: 388s/question.

#### AIME 2025 (temp=0.5)

| Method | Accuracy | Paper |
|--------|----------|-------|
| DTS-Greedy (1 trace) | **30.00%** | 34.67% |

**DTS-Greedy per-seed:**

| Seed | Accuracy | Avg Tokens |
|------|----------|------------|
| 0 | 26.67% | 6237 |
| 1 | 30.00% | 6083 |
| 2 | 26.67% | 6129 |
| 3 | 40.00% | 6364 |
| 4 | 26.67% | 6549 |

Avg wall time per seed: 12157s (203 min). Average: 405s/question.

### DeepSeek-R1-Distill-Qwen-7B

#### AIME 2025 (temp=0.6)

| Method | Accuracy |
|--------|----------|
| DTS-Greedy (1 trace) | **52.67%** |

Avg branch events: 24.0, Avg tokens: 6273. Total time: 17931s (299 min). Average: 119.5s/question.

## Known Limitations

- **Top-20 logprobs:** vLLM's API returns at most 20 logprobs per token, whereas the original implementation uses the full vocabulary (~150K tokens) for entropy computation. This approximation may affect branching decisions, particularly when the sampling temperature differs from `entropy_temp=0.6`.

## License

MIT License. See [LICENSE](../LICENSE).
