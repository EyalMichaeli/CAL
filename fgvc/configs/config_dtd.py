##################################################
# Training Config
##################################################
workers = 4                 # number of Dataloader workers
epochs = 160              # number of epochs
batch_size = 16          # batch size
learning_rate = 0.001        # initial learning rate

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
    --gpu_id 0 \
    --seed 2 \
    --train_sample_ratio 1.0 \
    --logdir logs/dtd/base \
    --dataset dtd \
    > nohup_outputs/dtd/base.log 2>&1 &

    
# run augmented - controlNet
nohup python train.py \
    --gpu_id 2 \
    --seed 1 \
    --train_sample_ratio 1.0 \
    --logdir logs/dtd/aug-controlNet-edges-120-200-captions_prompts-clip_filtering\
    --dataset dtd \
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/dtd/sd_v1.5/sd_v1.5_canny_v1-prompt_is_caption_random_prompt_prompt_with_sub_class_num_per_image_2_gs_7.5_num_inf_steps_30_controlnet_scale_1.0_low_120_high_200_seed_0_images_lpips_filter_None_None_clip_filtering_True.json \
    --aug_sample_ratio 0.1 \
    > nohup_outputs/dtd/aug.log 2>&1 &

    
# run augmented: SD (txt2img)
nohup python train.py \
    --gpu_id 2 \
    --seed 1 \
    --train_sample_ratio 1.0 \
    --logdir logs/dtd/aug-SD-xl-captions_prompts-clip_filtering\
    --dataset dtd \
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/dtd/sd_xl/sd_xl_None_v1-prompt_is_caption_random_prompt_prompt_with_sub_class_num_per_image_2_gs_7.5_num_inf_steps_30_seed_0_images_lpips_filter_None_None_clip_filtering_True.json \
    --aug_sample_ratio 0.1 \
    > nohup_outputs/dtd/aug.log 2>&1 &


# run augmented 
nohup python train.py \
    --gpu_id 0 \
    --seed 2 \
    --train_sample_ratio 1.0 \
    --logdir logs/dtd/aug-merged_ip2p-v10-controlnet-v3\
    --dataset dtd \
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/dtd/merged_ip2p-v10-controlnet-v3.json \
    --aug_sample_ratio 0.4 \
    > nohup_outputs/dtd/aug.log 2>&1 &
    
    
# run augmented with special augmentation
nohup python train.py \
    --gpu_id 0 \
    --seed 1 \
    --train_sample_ratio 0.5 \
    --logdir logs/dtd/augmented_seed_2_sample_ratio_0.5_resnet_50_aug_ratio_0.4_ \
    --dataset dtd \
    --aug_json  \
    --aug_sample_ratio 0.4 \
    --special_aug cutmix \
    > nohup_outputs/dtd/aug.log 2>&1 &



new jsons:
# BLIP diffusion
# v1.0: 
    
# ip2p
# v1.0


# ControlNet: 
# v1.0: image catpions as prompts. 2x. CLIP filtering.
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/dtd/sd_v1.5/sd_v1.5_canny_v1-prompt_is_caption_random_prompt_prompt_with_sub_class_num_per_image_2_gs_7.5_num_inf_steps_30_controlnet_scale_1.0_low_120_high_200_seed_0_images_lpips_filter_None_None_clip_filtering_True.json \
    
Stable diffusion:
# v0: image catpions as prompts. 2x. CLIP filtering.
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/dtd/sd_v1.5/sd_v1.5_None_v1-prompt_is_caption_random_prompt_prompt_with_sub_class_num_per_image_2_gs_7.5_num_inf_steps_30_seed_0_images_lpips_filter_None_None_clip_filtering_True.json \
# v1: SD XL. image catpions as prompts. 2x. CLIP filtering. 2x.
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/dtd/sd_xl/sd_xl_None_v1-prompt_is_caption_random_prompt_prompt_with_sub_class_num_per_image_2_gs_7.5_num_inf_steps_30_seed_0_images_lpips_filter_None_None_clip_filtering_True.json \
    

# MERGED
# v0: 


"""