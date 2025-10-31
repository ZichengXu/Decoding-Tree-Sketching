set -e

MODELS=("1.5B" "7B")
DATASETS=("aime24" "aime25")

ENTROPY=2.5
TOP_K=3
MAX_ACTIVE=12
MAX_TOKENS=32768
TRIALS=5
SEED=0

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
        else
            echo "ERROR: Unknown dataset $dataset. Exiting."
            exit 1
        fi
        
        echo -e "\n--- [DTS] Model: $model, Dataset: $dataset, Temp: $local_temp---"
        
        python decoding_tree_sketching/run_experiments.py dts \
            --model_name "$model" \
            --dataset_name "$dataset" \
            --config_file "configs/config.yaml" \
            -e $ENTROPY \
            -k $TOP_K \
            -a $MAX_ACTIVE \
            -m $MAX_TOKENS \
            -t $local_temp \
            -s $SEED \
            -n $TRIALS
            
        echo "--- [DTS] Finished: $model, $dataset ---"
    done
done

echo -e "\n========================================="
echo "  ALL DTS Experiments Complete"
echo "========================================="
