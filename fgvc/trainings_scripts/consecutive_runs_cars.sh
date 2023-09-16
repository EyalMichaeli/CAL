#!/bin/bash

############################################################################################################
# run several with the same hparams, but diff seeds and train_sample_ratios
# Define the hyperparameter values found in hparam_search
dataset="cars"
batch_size="8"
learning_rate="0.001"
weight_decay="0.001"
epochs="160"
gpu_id="2"

# iterate over
seeds=("1" "2" "3" "4")
# train_sample_ratios=("0.25" "0.5" "0.75" "1.0")
train_sample_ratios=("1.0")
# special_augs=("cutmix" "randaug" "classic" "no")
special_augs=("classic")

# Run the training 
for seed in "${seeds[@]}"
do
    for train_sample_ratio in "${train_sample_ratios[@]}"
    do
        for special_aug in "${special_augs[@]}"
        do
            echo "Running with seed: $seed and train_sample_ratio: $train_sample_ratio and special_aug: $special_aug"
            python train.py \
                --gpu_id $gpu_id \
                --seed $seed \
                --train_sample_ratio $train_sample_ratio \
                --epochs $epochs \
                --logdir logs/$dataset/base_cutmix_and_classic \
                --learning_rate $learning_rate \
                --weight_decay $weight_decay \
                --batch_size $batch_size \
                --special_aug $special_aug \
                --dataset $dataset \
                --use_cutmix
            wait # Wait for the previous training process to finish before starting the next one
        done
    done
done

############################################################################################################



# run with 
"""
nohup trainings_scripts/consecutive_runs_cars.sh > script_output_cars.log 2>&1 &
"""