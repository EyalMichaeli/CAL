##################################################
# Training Config
##################################################
workers = 10                # number of Dataloader workers
epochs = 160              # number of epochs
batch_size = 8            # batch size
learning_rate = 0.001      # initial learning rate

##################################################
# Model Config
##################################################
image_size = (224, 224)     # size of training images
net = 'resnet101'  # feature extractor
num_attentions = 32     # number of attention maps
beta = 5e-2                 # param for update feature centers
weight_decay = 0.001        # weight decay

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

# train base 
nohup python train.py \
    --gpu_id 3 \
    --seed 1 \
    --train_sample_ratio 1.0 \
    --logdir logs/cars/base \
    --dataset cars \
    --special_aug classic \
    > nohup_outputs/cars/base.log 2>&1 &

    
# train base with with special augmentation
nohup python train.py \
    --gpu_id 0 \
    --seed 1 \
    --train_sample_ratio 0.5 \
    --logdir logs/cars/base_special_aug_randaug \
    --dataset cars \
    --special_aug randaug \
    > nohup_outputs/cars/base.log 2>&1 &

    
# run augmented - SD (txt2img)
nohup python train.py \
    --gpu_id 3 \
    --seed 2 \
    --train_sample_ratio 1.0 \
    --logdir logs/cars/aug-SD-XL-clip_filtering-txt2sentance-sce \
    --dataset cars \
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/cars/sd_xl/sd_xl_None_v3-SD-XL-text2sentance_prompts_random_prompt_prompt_with_sub_class_num_per_image_2_gs_7.5_num_inf_steps_30_seed_0_images_lpips_filter_None_None_clip_filtering_true.json \
    --aug_sample_ratio 0.5 \
    --stop_aug_after_epoch 160 \
    --use_target_soft_cross_entropy \
    > nohup_outputs/cars/aug.log 2>&1 &


# run augmented - controlNet
nohup python train.py \
    --gpu_id 3 \
    --seed 2 \
    --train_sample_ratio 1.0 \
    --logdir logs/cars/aug-controlNet-edges-120-200-txt2sentance-clip_filtering-sce \
    --dataset cars \
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/cars/sd_v1.5/sd_v1.5_canny_v3-SD-text2sentance_prompts_random_prompt_prompt_with_sub_class_num_per_image_2_gs_7.5_num_inf_steps_30_low_120_high_200_seed_0_images_lpips_filter_None_None_clip_filtering_true.json \
    --aug_sample_ratio 0.5 \
    --stop_aug_after_epoch 160 \
    --use_target_soft_cross_entropy \
    > nohup_outputs/cars/aug.log 2>&1 &

    
# run augmented with special augmentation
nohup python train.py \
    --gpu_id 0 \
    --seed 1 \
    --train_sample_ratio 0.5 \
    --logdir logs/cars/augmented_seed_2_sample_ratio_0.5_resnet_50_aug_ratio_0.4_ \
    --dataset cars \
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/cars/ip2p/cars_2023_0823_2212_29_cars_ip2p_hive_sd_1.5_rw_only_const_instruct_1x_image_w_1.5_all-constant-instructions_images_lpips_filter_0.1_0.7.json \
    --aug_sample_ratio 0.4 \
    --special_aug cutmix \
    > nohup_outputs/cars/aug.log 2>&1 &




# BLIP diffusion
# v0: first version of BLIP diffusion. stylizaiton, same class image. gs=7.5. 1x
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/cars/ip2p/blip_diffusion_v0_num_per_image_1_num_per_pair_1_guidance_scale_7.5_num_inference_steps_50_images_lpips_filter_0.1_0.7.json \
# v1, same as v0 + matching car direction (back, front). 1x
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/cars/ip2p/blip_diffusion_v1-matching_car_direction_num_per_image_1_num_per_pair_1_guidance_scale_7.5_num_inference_steps_50_0_images_lpips_filter_0.1_0.7.json \
# v1.1, same as v1 + varying prompts 1x
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/cars/ip2p/blip_diffusion_v1.1-matching_car_direction_num_per_image_1_num_per_pair_1_guidance_scale_7.5_num_inference_steps_50_0_images_lpips_filter_0.1_0.7.json \
# v1.2, same as v1.1 2x
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/cars/ip2p/blip_diffusion_v1.2-matching_car_direction_num_per_image_1_num_per_pair_1_guidance_scale_7.5_num_inference_steps_50_0_images_lpips_filter_0.1_0.7.json \
# v2, same as v1 + prompt contains car sub class. 1x
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/cars/ip2p/blip_diffusion_v2-_random_prompt_same_car_direction_prompt_with_sub_class_num_per_image_1_num_per_pair_1_guidance_scale_7.5_num_inference_steps_50_0_images_lpips_filter_0.1_0.7.json \
# v3, edit not stylization. 1x. gs=7.5. prompt with sub class. same car direction.
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/cars/blip_diffusion/blip_diffusion_edit_v2-only_val_random_prompt_same_car_direction_prompt_with_sub_class_num_per_image_1_gs_7.5_num_inf_steps_50_seed_0_images_lpips_filter_0.1_0.7.json \

    
# ip2p
# v0: first version of ip2p. mainly car color changes and weather. 1x
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/cars/ip2p/cars_2023_0823_2212_29_cars_ip2p_hive_sd_1.5_rw_only_const_instruct_1x_image_w_1.5_all-constant-instructions_images_lpips_filter_0.1_0.7.json \
# v1: Magic brush. Mainly changing the background and weather. 1x
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/cars/ip2p/cars_2023_0825_1054_20_cars_ip2p_magic_brush_only_const_instruct_1x_image_w_1.5_all-constant-instructions_images_lpips_filter_0.1_0.7.json \
# v2: Magic brush. Mainly changing the background and weather. 1x
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/cars/ip2p/cars_2023_0826_1725_54_cars_ip2p_magic_brush_only_const_instruct_1x_image_w_1.5_all-constant-instructions_images_lpips_filter_0.1_0.7.json \


# controlNet
# v0: canny 30-70. 2x. prompt with sub class.
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/cars/sd_v1.5/sd_v1.5_canny_v0_random_prompt_prompt_with_sub_class_num_per_image_2_gs_7.5_num_inf_steps_30_seed_0_images_lpips_filter_0.1_0.7.json \
# v1: canny 70-120. 2x. prompt with sub class.
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/cars/sd_v1.5/sd_v1.5_canny_v0-higer_canny_thresholds_random_prompt_prompt_with_sub_class_num_per_image_2_gs_7.5_num_inf_steps_30_canny_low_70_canny_high_120_seed_0_images_lpips_filter_0.1_0.7.json \
# v2: canny 120-200. 2x. prompt with sub class.
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/cars/sd_v1.5/sd_v1.5_canny_v0-even_higher_canny_random_prompt_prompt_with_sub_class_num_per_image_2_gs_7.5_num_inf_steps_30_canny_low_120_canny_high_200_seed_0_images_lpips_filter_0.1_0.7.json \
# v3: same, text2sentance model. 2x.
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/cars/sd_v1.5/sd_v1.5_canny_v3-SD-text2sentance_prompts_random_prompt_prompt_with_sub_class_num_per_image_2_gs_7.5_num_inf_steps_30_low_120_high_200_seed_0_images_lpips_filter_None_None.json \
# v4: same as v3, with CLIP filtering. 2x.
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/cars/sd_v1.5/sd_v1.5_canny_v3-SD-text2sentance_prompts_random_prompt_prompt_with_sub_class_num_per_image_2_gs_7.5_num_inf_steps_30_low_120_high_200_seed_0_images_lpips_filter_None_None_clip_filtering_true.json \

    
# Stable diffusion:
# v0: same pormpts used for controlnet. no lpips. 2x
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/cars/sd_v1.5/sd_v1.5_None_v0-txt2img-sd_v1.5_random_prompt_prompt_with_sub_class_num_per_image_2_gs_7.5_num_inf_steps_30_canny_low_120_canny_high_200_seed_0_images_lpips_filter_None_None.json \
# v1: same as v0, SD XL. 2x
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/planes/sd_xl/sd_xl_None_v0-even_higher_canny_random_prompt_prompt_with_sub_class_num_per_image_2_gs_7.5_num_inf_steps_30_seed_0_images_lpips_filter_None_None.json \
# v3: SD XL. text2sentance model. 2x.
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/cars/sd_xl/sd_xl_None_v3-SD-XL-text2sentance_prompts_random_prompt_prompt_with_sub_class_num_per_image_2_gs_7.5_num_inf_steps_30_seed_0_images_lpips_filter_None_None.json \
# v4: same as v3, with CLIP filtering. 2x.
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/cars/sd_xl/sd_xl_None_v3-SD-XL-text2sentance_prompts_random_prompt_prompt_with_sub_class_num_per_image_2_gs_7.5_num_inf_steps_30_seed_0_images_lpips_filter_None_None_clip_filtering_true.json \
# v5: same as v0, with CLIP filtering. 2x.
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/planes/sd_v1.5/sd_v1.5_None_v0-txt2img-sd_v1.5_random_prompt_prompt_with_sub_class_num_per_image_2_gs_7.5_num_inf_steps_30_canny_low_120_canny_high_200_seed_0_images_lpips_filter_None_None_clip_filtering_True.json \
# v6: SD v1.5 text2sentance model, CLIP filtering. 2x.
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/planes/sd_v1.5/sd_v1.5_None_v3-text2sentance_prompts_random_prompt_prompt_with_sub_class_num_per_image_2_gs_7.5_num_inf_steps_30_seed_0_images_lpips_filter_None_None_clip_filtering_True.json \
# v7: SD v1.5 SDEdit, text2sentance model, CLIP filtering. 2x.
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/planes/sd_v1.5_SDEdit/sd_v1.5_SDEdit_None_v4-text2sentance_prompts-SDEdit_random_prompt_prompt_with_sub_class_num_per_image_2_gs_7.5_num_inf_steps_30_SDEdit_strength_0.5_seed_0_images_lpips_filter_None_None_clip_filtering_True.json \
# v7: SD XL SDEdit, text2sentance model, CLIP filtering. 2x.
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/cars/sd_xl_SDEdit/sd_xl_SDEdit_None_v4-text2sentance_prompts-SDEdit_random_prompt_prompt_with_sub_class_num_per_image_2_gs_7.5_num_inf_steps_30_SDEdit_strength_0.5_seed_0_images_lpips_filter_None_None_clip_filtering_True.json \
    
#########
# merged jsons

# BLIP diffusion
# Merged_v0: v0+v1. 2x
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/cars/blip_diffusion/merged_v0_v1-2x.json \
# Merged_v1: v0+v1+v1.1. 1-3x
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/cars/blip_diffusion/merged_v0_v1.1-3x.json \
# Merged_v2: v0+v1+v1.1+v1.2. 3-5x
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/cars/ip2p/merged_v0_v1.2-5x.json \

# ip2p
# Merged_v0: v1+v2
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/cars/ip2p/merged_v1_v2-2x.json \

# mixes
    # merge of merges: blip v2 + ip2p v0. 2-5x + 1-2x = 3-7x (actually was up to more than that somehow, probably mistake)
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/cars/merged-merged_blip_v2_3-5x-plus-merged-ip2p-v0-2x.json \
"""