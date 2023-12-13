##################################################
# Training Config
##################################################
workers = 4                # number of Dataloader workers
epochs = 160              # number of epochs
batch_size = 16          # batch size
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
    --gpu_id 2 \
    --seed 5 \
    --train_sample_ratio 1.0 \
    --logdir logs/arch_dataset/base \
    --dataset arch_dataset \
    --special_aug classic \
    > nohup_outputs/arch_dataset/base.log 2>&1 &

    
# train base with with special augmentation
nohup python train.py \
    --gpu_id 0 \
    --seed 1 \
    --train_sample_ratio 0.5 \
    --logdir logs/arch_dataset/base_special_aug_randaug \
    --dataset arch_dataset \
    --special_aug randaug \
    > nohup_outputs/arch_dataset/base.log 2>&1 &

    
# run augmented - SD (txt2img)
nohup python train.py \
    --gpu_id 0 \
    --seed 2 \
    --train_sample_ratio 1.0 \
    --logdir logs/arch_dataset/aug-REALLY-SD-XL-captions_as_prompts-clip_filtering \
    --dataset arch_dataset \
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/arch_dataset/None/None_captions_v1_random_prompt_prompt_with_sub_class_num_per_image_2_gs_7.5_num_inf_steps_30_SDEdit_strength_0.5_seed_0_images_lpips_filter_None_None_clip_filtering_per_class_discount_1.json \
    --aug_sample_ratio 0.3 \
    --stop_aug_after_epoch 160 \
    > nohup_outputs/arch_dataset/aug.log 2>&1 &


# run augmented - controlNet
nohup python train.py \
    --gpu_id 3 \
    --seed 1 \
    --train_sample_ratio 1.0 \
    --logdir logs/arch_dataset/aug-controlNet-edges-120-200-captions_as_prompts-clip_filtering \
    --dataset arch_dataset \
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/arch_dataset/canny/canny_captions_v1_random_prompt_prompt_with_sub_class_num_per_image_2_gs_7.5_num_inf_steps_30_controlnet_scale_1.0_low_120_high_200_seed_0_images_lpips_filter_None_None_clip_filtering_per_class_discount_1.json \
    --aug_sample_ratio 0.5 \
    --stop_aug_after_epoch 160 \
    > nohup_outputs/arch_dataset/aug.log 2>&1 &

    
# run augmented with special augmentation
nohup python train.py \
    --gpu_id 0 \
    --seed 1 \
    --train_sample_ratio 0.5 \
    --logdir logs/arch_dataset/augmented_seed_2_sample_ratio_0.5_resnet_50_aug_ratio_0.4_ \
    --dataset arch_dataset \
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/arch_dataset/ip2p/arch_dataset_2023_0823_2212_29_arch_dataset_ip2p_hive_sd_1.5_rw_only_const_instruct_1x_image_w_1.5_all-constant-instructions_images_lpips_filter_0.1_0.7.json \
    --aug_sample_ratio 0.4 \
    --special_aug cutmix \
    > nohup_outputs/arch_dataset/aug.log 2>&1 &




# BLIP diffusion
# v0: 
    
# ip2p
# v0: 

# controlNet
# v0: canny 120-200. captions as prompts. CLIP filtering. 2x.
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/arch_dataset/canny/canny_captions_v1_random_prompt_prompt_with_sub_class_num_per_image_2_gs_7.5_num_inf_steps_30_controlnet_scale_1.0_low_120_high_200_seed_0_images_lpips_filter_None_None_clip_filtering_per_class_discount_1.json \
# v1: canny 120-200. txt2sentance (per class) prompts. CLIP filtering. 2x.
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/arch_dataset/canny_canny/canny_txt2sentence-per_class_v1_prompt_with_sub_class_num_per_image_2_gs_7.5_num_inf_steps_50_controlnet_scale_1.0_low_120_high_200_seed_0_images_lpips_filter_None_None_clip_filtering_per_class_discount_1.json \

# Stable diffusion:
# v0: SD XL SDEdit (strength = 0.5), captions as prompts, CLIP filtering. 2x.
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/arch_dataset/None/None_captions_v1_random_prompt_prompt_with_sub_class_num_per_image_2_gs_7.5_num_inf_steps_30_SDEdit_strength_0.5_seed_0_images_lpips_filter_None_None_clip_filtering_per_class_discount_1.json \
# v1: SD XL SDEdit (strength = 0.5), txt2sentance (per class) prompts, CLIP filtering. 2x.
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/arch_dataset/None_None/None_txt2sentence-per_class_v1_prompt_with_sub_class_num_per_image_2_gs_7.5_num_inf_steps_50_SDEdit_strength_0.5_seed_0_images_lpips_filter_None_None_clip_filtering_per_class_discount_1.json \
    
#########

# merged jsons


"""