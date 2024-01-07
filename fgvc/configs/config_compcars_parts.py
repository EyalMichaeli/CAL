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
weight_decay = 0.00001      # weight decay

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
    --gpu_id 1 \
    --seed 1 \
    --train_sample_ratio 1.0 \
    --logdir logs/compcars-parts/base \
    --dataset compcars-parts \
    --special_aug classic \
    > nohup_outputs/compcars-parts/base.log 2>&1 &

    
# train base with with special augmentation
nohup python train.py \
    --gpu_id 0 \
    --seed 1 \
    --train_sample_ratio 0.5 \
    --logdir logs/compcars-parts/base_special_aug_randaug \
    --dataset compcars-parts \
    --special_aug randaug \
    > nohup_outputs/compcars-parts/base.log 2>&1 &

    
# run augmented - SD (txt2img)
nohup python train.py \
    --gpu_id 3 \
    --seed 2 \
    --train_sample_ratio 1.0 \
    --logdir logs/compcars-parts/aug-SD-XL-clip_filtering-txt2sentance \
    --dataset compcars-parts \
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/compcars-parts/None/None_txt2sentence_v1_random_prompt_prompt_with_sub_class_num_per_image_2_gs_7.5_num_inf_steps_30_SDEdit_strength_0.5_seed_0_images_lpips_filter_None_None_clip_filtering_per_class_discount_1.json \
    --aug_sample_ratio 0.5 \
    --stop_aug_after_epoch 160 \
    --use_target_soft_cross_entropy \
    > nohup_outputs/compcars-parts/aug.log 2>&1 &


# run augmented - controlNet
nohup python train.py \
    --gpu_id 1 \
    --seed 1 \
    --train_sample_ratio 1.0 \
    --logdir logs/compcars-parts/aug-controlNet-sd_v1.5-edges-120-200-gpt_meta_class-clip_semantic_filtering \
    --dataset compcars-parts \
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/compcars/aug_data-parts/controlnet/sd_v1.5_SDEdit_strength_0.75/canny/gpt-meta_class/v1_prompt_with_sub_class-res_512-num_2-gs_7.5-num_inf_steps_50_controlnet_scale_1.0_low_120_high_200_seed_0/clip_filtering_per_class_discount_2_semantic_filtering_aug.json \
    --aug_sample_ratio 0.1 \
    > nohup_outputs/compcars-parts/aug.log 2>&1 &

    
# run augmented with special augmentation
nohup python train.py \
    --gpu_id 0 \
    --seed 1 \
    --train_sample_ratio 0.5 \
    --logdir logs/compcars-parts/augmented_seed_2_sample_ratio_0.5_resnet_50_aug_ratio_0.4_ \
    --dataset compcars-parts \
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/compcars-parts/ip2p/cars2_2023_0823_2212_29_cars2_ip2p_hive_sd_1.5_rw_only_const_instruct_1x_image_w_1.5_all-constant-instructions_images_lpips_filter_0.1_0.7.json \
    --aug_sample_ratio 0.4 \
    --special_aug cutmix \
    > nohup_outputs/compcars-parts/aug.log 2>&1 &




# BLIP diffusion
# v0: 
    
# ip2p
# v0: 

# controlNet
# v0: canny 120-200. SDEdit (strength = 0.75), captions as prompts, CLIP + semantic filtering. 2x.
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/compcars-parts/sd_v1.5_SDEdit_strength_0.75/canny/captions/images_lpips_filter_None_None_clip_filtering_per_class_discount_2.json \
# v1: same, with SD XL turbo (num inf steps = 4):
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/compcars-parts/sd_xl-turbo_SDEdit_strength_0.75/canny/captions/images_lpips_filter_None_None_clip_filtering_per_class_discount_2.json \
# v2: same as above, with chatgpt prompts (meta class):
    --aug_json 
# v3:same as v0. GPT prompts.
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/compcars/aug_data-parts/controlnet/sd_v1.5_SDEdit_strength_0.75/canny/gpt-meta_class/v1_prompt_with_sub_class-res_512-num_2-gs_7.5-num_inf_steps_50_controlnet_scale_1.0_low_120_high_200_seed_0/clip_filtering_per_class_discount_2_semantic_filtering_aug.json \

    
# Stable diffusion:
# v0: SD XL SDEdit (strength = 0.5), txt2sentance as prompts, CLIP + semantic filtering. 2x.
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/compcars-parts/sd_xl_SDEdit_strength_0.5/None/captions/images_lpips_filter_None_None_clip_filtering_per_class_discount_2.json \
# v1: same, with SD XL turbo:
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/compcars-parts/sd_xl-turbo_SDEdit_strength_0.5/None/captions/images_lpips_filter_None_None_clip_filtering_per_class_discount_2.json \
# v2: same as above, with chatgpt prompts (meta class):
    --aug_json 
    
#########

# merged jsons


"""