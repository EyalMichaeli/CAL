#!/bin/bash
############################################################################################################
# Define the hyperparameter values
# dataset="arch_dataset"
# dataset="cars"
dataset="cub"
gpu_id="0"

aug_json="/mnt/raid/home/eyal_michaeli/datasets/compcars/aug_data-parts/controlnet/sd_xl-turbo_SDEdit_strength_0.75/canny/gpt-meta_class/v1_prompt_with_sub_class-res_768-num_2-gs_0-num_inf_steps_2_controlnet_scale_1.0_low_120_high_200_seed_0/semantic_filtering-model_confidence_based_filtering_aug.json"
run_name="aug-controlNet-SD_XL_TURBO-0.75-num_inf_steps_2-edges-120-200-gpt_meta_class-clip_semantic_filtering"
# aug_json="/mnt/raid/home/eyal_michaeli/datasets/compcars/aug_data-parts/controlnet/sd_xl-turbo_SDEdit_strength_0.75/canny/gpt-meta_class/v1_prompt_with_sub_class-res_768-num_2-gs_0-num_inf_steps_2_controlnet_scale_1.0_low_120_high_200_seed_0/clip_filtering_per_class_discount_2_semantic_filtering_aug.json"
# run_name="aug-controlNet-SD_XL_TURBO-num_inf_steps_2-SDEdit_0.75-edges-120-200-gpt_meta_class_prompts-clip_semantic_filtering"
# aug_json="/mnt/raid/home/eyal_michaeli/datasets/stanford_cars/aug_data/controlnet/sd_xl-turbo/None/gpt-meta_class/v1_prompt_with_sub_class-res_768-num_2-gs_0-num_inf_steps_4_seed_0/clip_filtering_per_class_discount_1_semantic_filtering_aug.json"
# run_name="aug-SD-XL-turbo-num_inf_steps_4-gpt_meta_class_prompts-clip_semantic_filtering"
# aug_json="/mnt/raid/home/eyal_michaeli/datasets/FGVC-Aircraft/fgvc-aircraft-2013b/aug_data/controlnet/sd_xl-turbo_SDEdit_strength_0.5/None/gpt-meta_class/v1_prompt_with_sub_class-res_768-num_2-gs_0-num_inf_steps_2_seed_0/clip_filtering_per_class_discount_1_semantic_filtering_aug.json"
# run_name="aug-SD_XL_TURBO-num_inf_steps_2-SDEdit_0.5-gpt_meta_class_prompts-clip_semantic_filtering"

# iterate over
# seeds=("1" "2" "3" "4")
seeds=("1" "2" "3")
# seeds=("3" "4")
# seeds=("4")
# train_sample_ratios=("0.25" "0.5" "0.75" "1.0")
train_sample_ratios=("1.0")
# special_augs=("cutmix" "randaug" "classic" "no")
special_augs=("classic")

# aug_sample_ratios=("0.1" "0.3" "0.5")
# aug_sample_ratios=("0.1" "0.3")

# for compcars-parts:
# aug_sample_ratios=("0.1")

# for planes: 
# aug_sample_ratios=("0.2")

# for cars:
# aug_sample_ratios=("0.5")
# aug_sample_ratios=("0.3")
# aug_sample_ratios=("0.1" "0.3" "0.5")

# for CUB:
aug_sample_ratios=("0.1" "0.3" "0.5")


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