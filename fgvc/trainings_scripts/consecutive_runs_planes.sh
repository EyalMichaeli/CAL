#!/bin/bash

############################################################################################################
# run several with the same hparams, but diff seeds and train_sample_ratios
# Define the hyperparameter values
dataset="planes"
batch_size="4"
learning_rate="0.001"
weight_decay="0.0001"
epochs="160"
gpu_id="0"

# iterate over
seeds=("3" "4")
train_sample_ratios=("0.25" "0.5" "0.75" "1.0")

# Run the training 
for seed in "${seeds[@]}"
do
    for train_sample_ratio in "${train_sample_ratios[@]}"
    do
        echo "Running with seed: $seed and train_sample_ratio: $train_sample_ratio"
        python train.py \
            --gpu_id $gpu_id \
            --seed $seed \
            --train_sample_ratio $train_sample_ratio \
            --epochs $epochs \
            --logdir logs/$dataset/base \
            --learning_rate $learning_rate \
            --weight_decay $weight_decay \
            --batch_size $batch_size \
            --dataset $dataset 
        wait # Wait for the previous training process to finish before starting the next one
    done
done

############################################################################################################



# run with nohup trainings_scripts/consecutive_runs_planes.sh > script_output_planes.log 2>&1 &
