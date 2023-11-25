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

# train base 
nohup python train.py \
    --gpu_id 2 \
    --seed 1 \
    --train_sample_ratio 1.0 \
    --logdir logs/cars2/base \
    --dataset cars2 \
    --special_aug classic \
    > nohup_outputs/cars2/base.log 2>&1 &

    
# train base with with special augmentation
nohup python train.py \
    --gpu_id 0 \
    --seed 1 \
    --train_sample_ratio 0.5 \
    --logdir logs/cars2/base_special_aug_randaug \
    --dataset cars2 \
    --special_aug randaug \
    > nohup_outputs/cars2/base.log 2>&1 &

    
# run augmented - SD (txt2img)
nohup python train.py \
    --gpu_id 3 \
    --seed 2 \
    --train_sample_ratio 1.0 \
    --logdir logs/cars2/aug-SD-XL-clip_filtering-txt2sentance-sce \
    --dataset cars2 \
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/cars2/sd_xl/sd_xl_None_v3-SD-XL-text2sentance_prompts_random_prompt_prompt_with_sub_class_num_per_image_2_gs_7.5_num_inf_steps_30_seed_0_images_lpips_filter_None_None_clip_filtering_true.json \
    --aug_sample_ratio 0.5 \
    --stop_aug_after_epoch 160 \
    --use_target_soft_cross_entropy \
    > nohup_outputs/cars2/aug.log 2>&1 &


# run augmented - controlNet
nohup python train.py \
    --gpu_id 3 \
    --seed 2 \
    --train_sample_ratio 1.0 \
    --logdir logs/cars2/aug-controlNet-edges-120-200-txt2sentance-clip_filtering-sce \
    --dataset cars2 \
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/cars2/sd_v1.5/sd_v1.5_canny_v3-SD-text2sentance_prompts_random_prompt_prompt_with_sub_class_num_per_image_2_gs_7.5_num_inf_steps_30_low_120_high_200_seed_0_images_lpips_filter_None_None_clip_filtering_true.json \
    --aug_sample_ratio 0.5 \
    --stop_aug_after_epoch 160 \
    --use_target_soft_cross_entropy \
    > nohup_outputs/cars2/aug.log 2>&1 &

    
# run augmented with special augmentation
nohup python train.py \
    --gpu_id 0 \
    --seed 1 \
    --train_sample_ratio 0.5 \
    --logdir logs/cars2/augmented_seed_2_sample_ratio_0.5_resnet_50_aug_ratio_0.4_ \
    --dataset cars2 \
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/cars2/ip2p/cars2_2023_0823_2212_29_cars2_ip2p_hive_sd_1.5_rw_only_const_instruct_1x_image_w_1.5_all-constant-instructions_images_lpips_filter_0.1_0.7.json \
    --aug_sample_ratio 0.4 \
    --special_aug cutmix \
    > nohup_outputs/cars2/aug.log 2>&1 &




# BLIP diffusion
# v0: 
    
# ip2p
# v0: 

# controlNet
# v0: 

# Stable diffusion:
# v0: 

#########

# merged jsons


"""