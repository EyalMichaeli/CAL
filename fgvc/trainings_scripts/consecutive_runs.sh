#!/bin/bash

############################################################################################################
# run several with the same hparams, but diff seeds and train_sample_ratios
# Define the hyperparameter values
dataset="compcars-parts"
gpu_id="1"

# iterate over
# seeds=("1" "2" "3" "4")
# seeds=("3" "4")
seeds=("1" "2")
# seeds=("2" "3" "4")
# seeds=("1" "2" "3")
# seeds=("4")
# train_sample_ratios=("0.25" "0.5" "0.75" "1.0")
train_sample_ratios=("1.0")
# special_augs=("cutmix" "randaug" "classic" "no")
special_augs=("classic")

# Run the training 
for train_sample_ratio in "${train_sample_ratios[@]}"
do
    for special_aug in "${special_augs[@]}"
    do
        for seed in "${seeds[@]}"
        do
            echo "Running with seed: $seed and train_sample_ratio: $train_sample_ratio and special_aug: $special_aug"
            python train.py \
                --gpu_id $gpu_id \
                --seed $seed \
                --train_sample_ratio $train_sample_ratio \
                --logdir logs/$dataset/base-fixed_classes_401 \
                --special_aug $special_aug \
                --dataset $dataset 
            wait # Wait for the previous training process to finish before starting the next one
        done
    done
done


############################################################################################################



# run with 
"""
nohup trainings_scripts/consecutive_runs.sh > script_output.log 2>&1 &
"""