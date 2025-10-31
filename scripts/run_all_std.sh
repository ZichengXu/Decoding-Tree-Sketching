set -e

MODELS=("1.5B" "7B")
DATASETS=("aime24" "aime25")

TRIALS=5
SEED=0

echo "========================================="
echo "  Running ALL Standard Experiments"
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
        
        echo -e "\n--- [Standard] Model: $model, Dataset: $dataset, Temp: $local_temp ---"
        
        python decoding_tree_sketching/run_experiments.py standard \
            --model_name "$model" \
            --dataset_name "$dataset" \
            --config_file "configs/config.yaml" \
            -t $local_temp \
            -s $SEED \
            -n $TRIALS
            
        echo "--- [Standard] Finished: $model, $dataset ---"
    done
done

echo -e "\n========================================="
echo "  ALL Standard Experiments Complete"
echo "========================================="
