##################################################
# Training Config
##################################################
workers = 4                 # number of Dataloader workers
epochs = 100              # number of epochs
batch_size = 16           # batch size
learning_rate = 1e-3        # initial learning rate

##################################################
# Model Config
##################################################
image_size = (224, 224)     # size of training images
net = 'resnet50'  # feature extractor
num_attentions = 32     # number of attention maps
beta = 5e-2                 # param for update feature centers

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
    --epochs 160 \
    --batch_size 16 \
    --learning_rate 1e-3 \
    --logdir logs/cars/hparam_search \
    --dataset cars \
    > nohup_outputs/cars/base.log &

# train base with 75% of the data
nohup python train.py \
    --gpu_id 0 \
    --seed 1 \
    --train_sample_ratio 0.75 \
    --epochs 160 \
    --logdir logs/cars/base_seed_1_sample_ratio_0.75_resnet_50 \
    --dataset cars \
    > nohup_outputs/cars/base.log &

# train base with 50% of the data
nohup python train.py \
    --gpu_id 0 \
    --seed 1 \
    --train_sample_ratio 0.5 \
    --logdir logs/cars/base \
    --dataset cars \
    > nohup_outputs/cars/base.log &

    
# train base with 50% of the data, with special augmentation
nohup python train.py \
    --gpu_id 0 \
    --seed 2 \
    --train_sample_ratio 0.5 \
    --logdir logs/cars/base_seed__sample_ratio_0.5_resnet_50_cutmix \
    --dataset cars \
    --special_aug cutmix \
    > nohup_outputs/cars/base.log &

    
# run augmented 50%
nohup python train.py \
    --gpu_id 0 \
    --seed 2 \
    --train_sample_ratio 0.5 \
    --logdir logs/cars/augmented_seed_2_sample_ratio_0.5_resnet_50_aug_ratio_0.5_blip_diffusion_merged_same_source_and_style_gs_7.5-2x \
    --dataset cars \
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/cars/ip2p/merged_same_source_and_style_gs_7.5-2x.json \
    --aug_sample_ratio 0.5 \
    --stop_aug_after_epoch 100 \
    > nohup_outputs/cars/aug.log &

    
# run augmented 50%, with special augmentation
nohup python train.py \
    --gpu_id 0 \
    --seed 1 \
    --train_sample_ratio 0.5 \
    --logdir logs/cars/augmented_seed_2_sample_ratio_0.5_resnet_50_aug_ratio_0.4_ \
    --dataset cars \
    --aug_json  \
    --aug_sample_ratio 0.4 \
    --special_aug cutmix \
    > nohup_outputs/cars/aug.log &

    
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
    > nohup_outputs/cars/aug.log &


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
    > nohup_outputs/cars/aug.log &


# v0: 
"""