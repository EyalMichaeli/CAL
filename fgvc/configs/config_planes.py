##################################################
# Training Config
##################################################
workers = 4                 # number of Dataloader workers
epochs = 160              # number of epochs
batch_size = 4           # batch size
learning_rate = 0.001        # initial learning rate

##################################################
# Model Config
##################################################
image_size = (224, 224)     # size of training images
net = 'resnet101'  # feature extractor
num_attentions = 32     # number of attention maps
beta = 5e-2                 # param for update feature centers
weight_decay = 0.0001        # weight decay

##################################################
# Dataset/Path Config
##################################################

# saving directory of .ckpt models
model_name = 'model.ckpt'
log_name = 'train.log'

# checkpoint model for resume training
ckpt = False
# ckpt = save_dir + model_name


"""
# train base with all the data
nohup python train.py \
    --gpu_id 2 \
    --seed 1 \
    --train_sample_ratio 1.0 \
    --epochs 160 \
    --logdir logs/planes/hparam_search \
    --learning_rate 0.0001 \
    --weight_decay 0.0001 \
    --batch_size 4 \
    --dataset planes \
    > nohup_outputs/planes/base.log &

# train base with 75% of the data
nohup python train.py \
    --gpu_id 0 \
    --seed 1 \
    --train_sample_ratio 0.75 \
    --epochs 160 \
    --logdir logs/planes/base_seed_1_sample_ratio_0.75_resnet_50 \
    --dataset planes \
    > nohup_outputs/planes/base.log &

# train base with 50% of the data
nohup python train.py \
    --gpu_id 3 \
    --seed 1 \
    --train_sample_ratio 0.5 \
    --logdir logs/planes/---normal_base_resnet_101_mathing_lr_bs \
    --dataset planes \
    > nohup_outputs/planes/base.log &

    
# train base with 50% of the data, with special augmentation
nohup python train.py \
    --gpu_id 0 \
    --seed 2 \
    --train_sample_ratio 0.5 \
    --logdir logs/planes/base_seed__sample_ratio_0.5_resnet_50_cutmix \
    --dataset planes \
    --special_aug cutmix \
    > nohup_outputs/planes/base.log &

    
# run augmented 50%
nohup python train.py \
    --gpu_id 3 \
    --seed 1 \
    --train_sample_ratio 0.5 \
    --logdir logs/planes/aug-merged-blip-v15-ip2p-v10 \
    --dataset planes \
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/planes/merged_ip2p-v10-blip-v15.json \
    --aug_sample_ratio 0.5 \
    --stop_aug_after_epoch 160 \
    > nohup_outputs/planes/aug.log &

    
# run augmented 50%, with special augmentation
nohup python train.py \
    --gpu_id 0 \
    --seed 1 \
    --train_sample_ratio 0.5 \
    --logdir logs/planes/augmented_seed_2_sample_ratio_0.5_resnet_50_aug_ratio_0.4_ \
    --dataset planes \
    --aug_json  \
    --aug_sample_ratio 0.4 \
    --special_aug cutmix \
    > nohup_outputs/planes/aug.log &

    
#### 75% of the data
# run augmented
nohup python train.py \
    --gpu_id 3 \
    --seed 1 \
    --train_sample_ratio 0.75 \
    --epochs 160 \
    --logdir logs/planes/augmented_seed_1_sample_ratio_0.75_resnet_50_aug_ratio_0.5_merged_v0-v1-v3-v4-v8_should_be_5x \
    --dataset planes \
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/planes/ip2p/merged_v0-v1-v3-v4-v8_should_be_5x.json \
    --aug_sample_ratio 0.5 \
    > nohup_outputs/planes/aug.log &


#### all data
# run augmented
nohup python train.py \
    --gpu_id 3 \
    --seed 1 \
    --train_sample_ratio 1.0 \
    --epochs 160 \
    --logdir logs/planes/augmented_seed_1_sample_ratio_1.0_resnet_50_aug_ratio_0.5_merged_v0-v1-v3-v4-v8_should_be_5x \
    --dataset planes \
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/planes/ip2p/merged_v0-v1-v3-v4-v8_should_be_5x.json \
    --aug_sample_ratio 0.5 \
    > nohup_outputs/planes/aug.log &


new jsons:
# BLIP diffusion
# v0

# ip2p
# v0



older jsons:
# v0: both gpt and constant. 70% gpt. LPIPS filter 0.1-0.8
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/planes/ip2p/planes_2023_0721_2241_01_planes_ip2p_regular_blip_gpt_type_object_wise_with_background_and_time_of_day_v1_less_focus_on_colors_1x_image_w_1.5_blip_gpt_v1_ratio_0.3_mainly_background_images_lpips_filter_0.1_0.8.json \
# v1: only constant. LPIPS filter 0.1-0.8
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/planes/ip2p/planes_2023_0722_1446_02_planes_ip2p_regular_blip_gpt_type_object_wise_with_background_and_time_of_day_v1_less_focus_on_colors_1x_image_w_1.5_blip_gpt_v1_ratio_1.0_mainly_background_images_lpips_filter_0.1_0.8.json \
# v2, only constant. painting. LPIPS filter 0.1-0.8
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/planes/ip2p/planes_2023_0723_1715_26_planes_ip2p_regular_blip_gpt_type_object_wise_with_background_and_time_of_day_v1_less_focus_on_colors_1x_image_w_1.5_blip_gpt_v1_ratio_1.0_paintings_images_lpips_filter_0.1_0.8.json \
# v3, only constant. changing colors of the planes. LPIPS filter 0.1-0.8
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/planes/ip2p/planes_2023_0724_1155_12_planes_ip2p_regular_blip_gpt_type_object_wise_with_background_and_time_of_day_v1_less_focus_on_colors_1x_image_w_1.5_blip_gpt_v1_ratio_1.0_color-wise-constant_images_lpips_filter_0.1_0.8.json\
# v4, similar to v1, plus weather changes. LPIPS filter 0.1-0.8. 1x images
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/planes/ip2p/planes_2023_0724_2113_02_planes_ip2p_regular_blip_gpt_type_object_wise_with_background_and_time_of_day_v1_less_focus_on_colors_1x_image_w_1.5_blip_gpt_v1_ratio_1.0_background-plus-weather_images_lpips_filter_0.1_0.8.json \
# v5, MERGED v1 and v4. LPIPS filter 0.1-0.8. 2x images
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/planes/ip2p/merged_v1_v4_total_should_be_2x.json \
# v6, MERGED v0-v5. LPIPS filter 0.1-0.8. 5x images
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/planes/ip2p/merged_v0-v4_should_be_5x.json \
# v7, MERGED v0-v1, v3-v4. LPIPS filter 0.1-0.8. 4x images
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/planes/ip2p/merged_v0-v1-plus-v3-v4_should_be_4x.json \
# v8, all constant together. LPIPS filter 0.1-0.8. 1x images
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/planes/ip2p/planes_2023_0731_1957_03_planes_ip2p_regular_blip_gpt_type_object_wise_with_background_and_time_of_day_v1_less_focus_on_colors_1x_image_w_1.5_blip_gpt_v1_ratio_1.0_all-constant-instructions_images_lpips_filter_0.1_0.8.json \
# v9, weather + plane color + background. LPIPS filter 0.1-0.8. 5x images
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/planes/ip2p/merged_v0-v1-v3-v4-v8_should_be_5x.json \
# v10, same as v8. LPIPS filter 0.1-0.8. 2x images
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/planes/ip2p/planes_2023_0731_1957_03_planes_ip2p_regular_blip_gpt_type_object_wise_with_background_and_time_of_day_v1_less_focus_on_colors_1x_image_w_1.5_blip_gpt_v1_ratio_1.0_all-constant-instructions_images_lpips_filter_0.1_0.4.json \
# v11: v9+v10. LPIPS filter 0.1-0.8. 7x images
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/planes/ip2p/merged_v0-v1-v3-v4-v8-v10_should_be_7x.json \
# v12, using BLIP diffusion for the first time. LPIPS filter 0.1-0.4. 1x images
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/planes/ip2p/blip_diffusion_v0_guidance_scale_5_1x_images_lpips_filter_0.1_0.4.json \
# v13: v10 + blip diffusion. LPIPS filter 0.1-0.4. 3x images
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/planes/ip2p/merged_v1-_and_blip_diffusionv0.json \
# v14, same as v12. 
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/planes/ip2p/blip_diffusion_v0_guidance_scale_5_1x_images2_lpips_filter_0.1_0.4.json \
# v15: v12+v14. LPIPS filter 0.1-0.4. 2x images
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/planes/ip2p/merged_blip_diffusion_v0-2x.json \
# v16, same as v15. LPIPS filter 0.1-0.4. 2x images
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/planes/ip2p/blip_diffusion_v0_guidance_scale_5_1x_images3_2x_lpips_filter_0.1_0.4.json \
# v17: v15 + v16. LPIPS filter 0.1-0.4. 4x images
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/planes/ip2p/merged_blip_diffusion_v0-4x.json \
# v18, using BLIP diffusion, but the style image is the same as source image. 1x images
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/planes/blip_diffusion/blip_diffusion_v0_same_source_and_style_num_per_image_1_num_per_pair_1_guidance_scale_7.5_num_inference_steps_50_images_lpips_filter_0.1_0.7.json \
# v19, using BLIP diffusion, g scale is 7.5. 1x images
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/planes/ip2p/blip_diffusion_v0_num_per_image_1_num_per_pair_1_guidance_scale_7.5_num_inference_steps_50_images_lpips_filter_0.1_0.4.json \
# v20, using BLIP diffusion, g scale is 10. 1x images
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/planes/ip2p/blip_diffusion_v0_num_per_image_1_num_per_pair_1_guidance_scale_10_num_inference_steps_50_images_lpips_filter_0.1_0.4.json \
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/planes/blip_diffusion/blip_diffusion_v0_same_source_and_style_num_per_image_1_num_per_pair_1_guidance_scale_7.5_num_inference_steps_50_images_lpips_filter_0.1_0.7.json
# v21, using BLIP diffusion, but the style image is the same as source image. 1x images
    --aug_json 
# v22, merged v18+v21. 2x images
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/planes/ip2p/merged_same_source_and_style_gs_7.5-2x.json \
# v23, BLIP diffusion, plane class in prompt. 1x images
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/planes/blip_diffusion/blip_diffusion_v2-_random_prompt_prompt_with_sub_class_num_per_image_1_num_per_pair_1_guidance_scale_7.5_num_inference_steps_50_0_images_lpips_filter_0.1_0.7.json \

# v24, merged 
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/planes/merged_ip2p-v10-blip-v15.json \
"""