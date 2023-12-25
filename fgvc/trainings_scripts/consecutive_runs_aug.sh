#!/bin/bash
############################################################################################################
# Define the hyperparameter values
# dataset="arch_dataset"
# dataset="planes"
dataset="compcars-parts"
gpu_id="1"

# aug_json="/mnt/raid/home/eyal_michaeli/datasets/aug_json_files/planes/sd_xl-turbo/canny/gpt-meta_class/images_lpips_filter_None_None_clip_filtering_per_class_discount_2.json"
# run_name="aug-controlNet-sd_v1.5-edges-120-200-gpt_meta_class-clip_semantic_filtering"
aug_json="/mnt/raid/home/eyal_michaeli/datasets/aug_json_files/compcars-parts/sd_xl-turbo_SDEdit_strength_0.75/canny/captions/images_lpips_filter_None_None_clip_filtering_per_class_discount_2.json"
run_name="aug-controlNet-SD_XL_TURBO-SDEdit_0.75-edges-120-200-gpt_meta_class_prompts-clip_semantic_filtering-higher_res"
# aug_json="/mnt/raid/home/eyal_michaeli/datasets/aug_json_files/compcars-parts/sd_xl-turbo_SDEdit_strength_0.5/None/gpt-meta_class/images_lpips_filter_None_None_clip_filtering_per_class_discount_2.json"
# run_name="aug-SD-XL-TURBO-SDEdit_0.5-gpt_meta_class_prompts-clip_semantic_filtering"
# aug_json="/mnt/raid/home/eyal_michaeli/datasets/aug_json_files/compcars/sd_v1.5_SDEdit_strength_0.9/None/txt2sentence-per_class/images_lpips_filter_None_None_clip_filtering_per_class_discount_1.json"
# run_name="aug-SD-1.5-SDEdit_0.9-txt2sentance_per_class-clip_semantic_filtering"

# iterate over
# seeds=("1" "2" "3" "4")
seeds=("1" "2" "3")
# seeds=("4")
# train_sample_ratios=("0.25" "0.5" "0.75" "1.0")
train_sample_ratios=("1.0")
# special_augs=("cutmix" "randaug" "classic" "no")
special_augs=("classic")

# aug_sample_ratios=("0.1" "0.3" "0.5")
aug_sample_ratios=("0.1")
# aug_sample_ratios=("0.3")
# aug_sample_ratios=("0.5")

# Run the training 
for train_sample_ratio in "${train_sample_ratios[@]}"
do
    for special_aug in "${special_augs[@]}"
    do
        for aug_sample_ratio in "${aug_sample_ratios[@]}"
        do
            for seed in "${seeds[@]}"
            do
                echo "Running with seed: $seed and train_sample_ratio: $train_sample_ratio and special_aug: $special_aug and aug_sample_ratio: $aug_sample_ratio"
                python train.py \
                    --gpu_id $gpu_id \
                    --seed $seed \
                    --train_sample_ratio $train_sample_ratio \
                    --logdir logs/$dataset/$run_name \
                    --special_aug $special_aug \
                    --aug_json $aug_json \
                    --aug_sample_ratio $aug_sample_ratio \
                    --dataset $dataset 
                wait # Wait for the previous training process to finish before starting the next one
            done
        done
    done
done
############################################################################################################
# run with 
"""
nohup trainings_scripts/consecutive_runs_aug.sh > aug_script_output.log 2>&1 &
"""