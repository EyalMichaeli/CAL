##################################################
# Training Config
##################################################
workers = 4                # number of Dataloader workers
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

# train base with all the data
nohup python train.py \
    --gpu_id 3 \
    --seed 1 \
    --train_sample_ratio 1.0 \
    --logdir logs/cars/base \
    --dataset cars \
    --special_aug classic \
    > nohup_outputs/cars/base.log 2>&1 &

# train base with 75% of the data
nohup python train.py \
    --gpu_id 2 \
    --seed 1 \
    --train_sample_ratio 0.75 \
    --epochs 160 \
    --logdir logs/cars/base \
    --dataset cars \
    > nohup_outputs/cars/base.log 2>&1 &

# train base with 50% of the data
nohup python train.py \
    --gpu_id 3 \
    --seed 1 \
    --train_sample_ratio 0.5 \
    --logdir logs/cars/base \
    --dataset cars \
    > nohup_outputs/cars/base.log 2>&1 &

    
# train base with 50% of the data, with special augmentation
nohup python train.py \
    --gpu_id 0 \
    --seed 1 \
    --train_sample_ratio 0.5 \
    --logdir logs/cars/base_special_aug_randaug \
    --dataset cars \
    --special_aug randaug \
    > nohup_outputs/cars/base.log 2>&1 &

    
# run augmented 50%
nohup python train.py \
    --gpu_id 3 \
    --seed 1 \
    --train_sample_ratio 0.5 \
    --logdir logs/cars/aug_blip-v3-edit-1x \
    --dataset cars \
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/cars/blip_diffusion/blip_diffusion_edit_v2-only_val_random_prompt_same_car_direction_prompt_with_sub_class_num_per_image_1_gs_7.5_num_inf_steps_50_seed_0_images_lpips_filter_0.1_0.7.json \
    --aug_sample_ratio 0.5 \
    --stop_aug_after_epoch 160 \
    > nohup_outputs/cars/aug.log 2>&1 &

    
# run augmented 50%, with special augmentation
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

    
#### 75% of the data
# run augmented
nohup python train.py \
    --gpu_id 3 \
    --seed 1 \
    --train_sample_ratio 0.75 \
    --epochs 160 \
    --logdir logs/cars/augmented_seed_1_sample_ratio_0.75_resnet_50_aug_ratio_0.5_merged_v0-v1-v3-v4-v8_should_be_5x \
    --dataset cars \
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/cars/ip2p/merged_v0-v1-v3-v4-v8_should_be_5x.json \
    --aug_sample_ratio 0.5 \
    > nohup_outputs/cars/aug.log 2>&1 &


#### all data
# run augmented
nohup python train.py \
    --gpu_id 3 \
    --seed 1 \
    --train_sample_ratio 1.0 \
    --epochs 160 \
    --logdir logs/cars/augmented_seed_1_sample_ratio_1.0_resnet_50_aug_ratio_0.5_merged_v0-v1-v3-v4-v8_should_be_5x \
    --dataset cars \
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/cars/ip2p/merged_v0-v1-v3-v4-v8_should_be_5x.json \
    --aug_sample_ratio 0.5 \
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

"""