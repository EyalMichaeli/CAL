#!/bin/bash

GPU_ID="0"  # IMPORTANT: change this to the GPU you want to use

dataset="compcars-parts"
# Define the hyperparameter values
# batch_sizes=("4" "8" "16")
# batch_sizes=("16")
# batch_sizes=("8")
batch_sizes=("16")

# learning_rates=("0.0001" "0.001" "0.01" "0.1")
# learning_rates=("0.0001" "0.001" "0.01")
learning_rates=("0.0001" "0.001")
weight_decays=("0.00001" "0.0001" "0.001")

start_from_last_used_hyperparameters=false
last_used_batch_size="4"
last_used_lr="0.0001"
last_used_wd="0.00001" 


# if start_from_last_used_hyperparameters is true, it will set is_training to false, and only when the combination of last used hyperparameters is found, 
# will set is_training to true
is_training=true
if [ "$start_from_last_used_hyperparameters" = true ] ; then
    is_training=false
    echo "Starting from the last used hyperparameters: batch_size=$last_used_batch_size, lr=$last_used_lr, wd=$last_used_wd"
fi

# Loop through the hyperparameter combinations
for batch_size in "${batch_sizes[@]}"; do
    for lr in "${learning_rates[@]}"; do
        for wd in "${weight_decays[@]}"; do
            #### Skip the hyperparameter combinations that have already been trained
            if [ "$is_training" = false ] ; then
                echo "skipping"
                if [ "$batch_size" = "$last_used_batch_size" ] && [ "$lr" = "$last_used_lr" ] && [ "$wd" = "$last_used_wd" ] ; then
                    is_training=true
                    # will train the next loop iteration
                fi
                continue
            fi
            
            echo "starting training..."
            ####
            nohup python train.py \
                --gpu_id "$GPU_ID" \
                --seed 1 \
                --train_sample_ratio 1.0 \
                --epochs 120 \
                --batch_size "$batch_size" \
                --learning_rate "$lr" \
                --weight_decay "$wd" \
                --logdir "logs/$dataset/hparam_search" \
                --dataset "$dataset" \
                > "nohup_outputs/$dataset/base.log" &
            
            # Wait for the current training run to finish before starting the next one
            wait
        done
    done
done

"""
nohup trainings_scripts/hparam_search.sh > nohup_hparam.log 2>&1 &
"""