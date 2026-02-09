#!/bin/bash
set -e

MODELS=("1.5B" "7B" "qwen30p6" "phi-4-mini-reasoning")
DATASETS=("aime24" "aime25" "gpqa_diamond" "livebench_reasoning")

SEEDS=(0 1 2 3 4)
NUM_TRACES_LIST=(1 8)

ENTROPY=2.5
TOP_K=3
MAX_ACTIVE=48
MAX_TOKENS=32768
TRIALS=1

echo "========================================="
echo "  Running ALL DTS Experiments"
echo "========================================="

for model in "${MODELS[@]}"; do
  for dataset in "${DATASETS[@]}"; do

    local_temp=""
    if [ "$dataset" == "aime24" ]; then
      local_temp=0.6
    elif [ "$dataset" == "aime25" ]; then
      local_temp=0.5
    elif [ "$dataset" == "gpqa_diamond" ]; then
      local_temp=0.6
    elif [ "$dataset" == "livebench_reasoning" ]; then
      local_temp=0.6
    else
      echo "ERROR: Unknown dataset $dataset. Exiting."
      exit 1
    fi

    for seed in "${SEEDS[@]}"; do
      for num_traces in "${NUM_TRACES_LIST[@]}"; do

        echo -e "\n--- [DTS] Model: $model, Dataset: $dataset, Temp: $local_temp, Seed: $seed, NumTraces: $num_traces ---"

        python decoding_tree_sketching/run_experiments.py dts \
          --model_name "$model" \
          --dataset_name "$dataset" \
          --config_file "configs/config.yaml" \
          -e "$ENTROPY" \
          -k "$TOP_K" \
          -a "$MAX_ACTIVE" \
          -m "$MAX_TOKENS" \
          -t "$local_temp" \
          -s "$seed" \
          -n "$TRIALS" \
          --num_traces "$num_traces" \

        echo "--- [DTS] Finished: $model, $dataset, seed=$seed, num_traces=$num_traces ---"

      done
    done
  done
done
