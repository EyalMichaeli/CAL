#!/bin/bash

cd ../..

# Define the hyperparameter values
batch_sizes=("16" "8" "4")
# learning_rates=("0.0001" "0.001" "0.01" "0.1")
learning_rates=("0.0001" "0.001")
weight_decays=("1e-5" "1e-4" "1e-3")

last_used_batch_size="4"
last_used_lr="0.001"
last_used_wd="1e-3"
start_from_last_used_hyperparameters=true
is_training=true
# if start_from_last_used_hyperparameters is true, it will set is_training to false, and only when the combination of last used hyperparameters is found, 
# will set is_training to true
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
                if [ "$batch_size" = "$last_used_batch_size" ] && [ "$lr" = "$last_used_lr" ] && [ "$wd" = "$last_used_wd" ] ; then
                    is_training=true
                    # will train the next loop iteration
                fi
                continue
            fi
            ####
            nohup python train.py \
                --gpu_id 2 \
                --seed 1 \
                --train_sample_ratio 1.0 \
                --epochs 160 \
                --batch_size "$batch_size" \
                --learning_rate "$lr" \
                --weight_decay "$wd" \
                --logdir logs/planes/hparam_search \
                --dataset planes \
                > "nohup_outputs/planes/base.log" &
            
            # Wait for the current training run to finish before starting the next one
            wait
        done
    done
done

# run with nohup trainings_scripts/hparam_search/hparam_search_planes.sh