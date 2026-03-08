#!/bin/bash
#SBATCH --job-name=dts-greedy
#SBATCH --account=a100acct
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

export PYTHONUNBUFFERED=1

echo "=== DTS-Greedy: AIME24, 5 seeds ==="
echo "Start time: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dts
cd ~/Decoding-Tree-Sketching/vllm
mkdir -p logs

# Greedy mode: 1 trace per question, 5 seeds
python run_benchmark.py dts \
    --model_name 1.5B \
    --dataset_name aime24 \
    -e 2.5 -k 3 -a 48 -m 32768 \
    -t 0.6 -s 0 -n 5 \
    --num_traces 1 \
    --enforce_eager \
    --gpu_memory_utilization 0.9 \
    --max_model_len 40960 \
    --tag greedy

echo ""
echo "=== Done at $(date) ==="
