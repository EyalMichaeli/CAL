import datetime
import importlib
import os
from pathlib import Path

import numpy as np
import config as config

import time
import logging
import warnings
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import random
from models import WSDAN_CAL
from utils import CenterLoss, AverageMeter, TopKAccuracyMetric, ModelCheckpoint, batch_augment
from datasets import get_trainval_datasets
import argparse
import wandb


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--logdir', type=str, default='logs')
parser.add_argument('--dataset', type=str, default='planes')
# augmentation options
parser.add_argument("--aug_json", type=str, default=None,
                    help="path to augmentation json file")
parser.add_argument("--aug_sample_ratio", type=float, default=None,
                    help="ratio to augment the original image")
parser.add_argument("--limit_aug_per_image", type=int, default=None,
                    help="limit augmentations per image, default None, which is take all")
parser.add_argument("--special_aug", type=str, default=None,
                    help="special classic augmentation to use, out of randaug, autoaug")
# add arg to take only some amount for the train set
parser.add_argument("--train_sample_ratio", type=float, default=1.0,
                    help="ratio of train set to take")

args = parser.parse_args()


"""
# train base with all the data
# increase epochs in config to 160 if runninng this
nohup sh -c 'python train.py \
    --gpu_id 0 \
    --seed 1 \
    --train_sample_ratio 1.0 \
    --logdir logs/planes/base_seed_1_sample_ratio_1.0_resnet_50 \
    --dataset planes' \
    2>&1 | tee -a nohup_outputs/planes/base.log &


# train base with 50% of the data
nohup python train.py \
    --gpu_id 3 \
    --seed 1 \
    --train_sample_ratio 0.5 \
    --logdir logs/planes/base_seed_1_sample_ratio_0.5_resnet_50 \
    --dataset planes \
    > nohup_outputs/planes/base.log &

    
# train base with 50% of the data, with special augmentation
nohup python train.py \
    --gpu_id 0 \
    --seed 1 \
    --train_sample_ratio 0.5 \
    --logdir logs/planes/base_seed_1_sample_ratio_0.5_resnet_50_autoaug \
    --dataset planes \
    --special_aug autoaug \
    > nohup_outputs/planes/base.log &

    
# run augmented
nohup python train.py \
    --gpu_id 3 \
    --seed 2 \
    --train_sample_ratio 0.5 \
    --logdir logs/planes/augmented_seed_2_sample_ratio_0.5_resnet_50_aug_ratio_0.2_merged_v0-v4_should_be_5x \
    --dataset planes \
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/planes/ip2p/merged_v0-v4_should_be_5x.json \
    --aug_sample_ratio 0.2 \
    > nohup_outputs/planes/aug.log &

# run augmented
nohup python train.py \
    --gpu_id 0 \
    --seed 1 \
    --train_sample_ratio 0.5 \
    --logdir logs/planes/augmented_seed_2_sample_ratio_0.5_resnet_50_aug_ratio_0.4_painting_should_be_1x \
    --dataset planes \
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/planes/ip2p/planes_2023_0723_1715_26_planes_ip2p_regular_blip_gpt_type_object_wise_with_background_and_time_of_day_v1_less_focus_on_colors_1x_image_w_1.5_blip_gpt_v1_ratio_1.0_paintings_images_lpips_filter_0.1_0.8.json \
    --aug_sample_ratio 0.4 \
    > nohup_outputs/planes/aug.log &


# v0: both gpt and constant. 70% gpt. LPIPS filter 0.1-0.8
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/planes/ip2p/planes_2023_0721_2241_01_planes_ip2p_regular_blip_gpt_type_object_wise_with_background_and_time_of_day_v1_less_focus_on_colors_1x_image_w_1.5_blip_gpt_v1_ratio_0.3_mainly_background_images_lpips_filter_0.1_0.8.json \
# v1: only constant. LPIPS filter 0.1-0.8
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/planes/ip2p/planes_2023_0722_1446_02_planes_ip2p_regular_blip_gpt_type_object_wise_with_background_and_time_of_day_v1_less_focus_on_colors_1x_image_w_1.5_blip_gpt_v1_ratio_1.0_mainly_background_images_lpips_filter_0.1_0.8.json \
# v2, only constant. painting. LPIPS filter 0.1-0.8
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/planes/ip2p/planes_2023_0723_1715_26_planes_ip2p_regular_blip_gpt_type_object_wise_with_background_and_time_of_day_v1_less_focus_on_colors_1x_image_w_1.5_blip_gpt_v1_ratio_1.0_paintings_images_lpips_filter_0.1_0.8.json \
# v3, only constant. changing colors of the planes. LPIPS filter 0.1-0.8
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/planes/ip2p/planes_2023_0724_1155_12_planes_ip2p_regular_blip_gpt_type_object_wise_with_background_and_time_of_day_v1_less_focus_on_colors_1x_image_w_1.5_blip_gpt_v1_ratio_1.0_color-wise-constant_images_lpips_filter_0.1_0.8.json\
# v4, similar to v1, plus weather changes. LPIPS filter 0.1-0.8
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/planes/ip2p/planes_2023_0724_2113_02_planes_ip2p_regular_blip_gpt_type_object_wise_with_background_and_time_of_day_v1_less_focus_on_colors_1x_image_w_1.5_blip_gpt_v1_ratio_1.0_background-plus-weather_images_lpips_filter_0.1_0.8.json \
# v5, MERGED v1 and v4. LPIPS filter 0.1-0.8
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/planes/ip2p/merged_v1_v4_total_should_be_2x.json \
# v6, MERGED v0-v5. LPIPS filter 0.1-0.8
    --aug_json /mnt/raid/home/eyal_michaeli/datasets/aug_json_files/planes/ip2p/merged_v0-v4_should_be_5x.json \
"""

# General loss functions
cross_entropy_loss = nn.CrossEntropyLoss()
center_loss = CenterLoss()

# loss and metric
loss_container = AverageMeter(name='loss')
top1_container = AverageMeter(name='top1')
top5_container = AverageMeter(name='top5')

raw_metric = TopKAccuracyMetric(topk=(1, 5))
crop_metric = TopKAccuracyMetric(topk=(1, 5))
drop_metric = TopKAccuracyMetric(topk=(1, 5))

best_acc = 0.0



def init_logging(logdir):
    r"""
    Create log directory for storing checkpoints and output images.
    Given a log dir like logs/test_run, creates a new directory logs/2020_0101_1234_test_run

    Args:
        logdir (str): Log directory name
    """
    # log dir
    date_uid = str(datetime.datetime.now().strftime("%Y_%m%d_%H%M_%S"))
    logdir_path = Path(logdir)
    logdir = str(logdir_path.parent / f"{date_uid}_{logdir_path.name}")
    os.makedirs(logdir, exist_ok=True)
    # log file
    log_file = os.path.join(logdir, 'log.log')
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logging.getLogger().addHandler(fh)
    return logdir


def main():

    ##################################
    # Logging setting
    ##################################
    config.save_dir = init_logging(args.logdir)
    wandb.init(project="CAL-aug-experiments", group=args.dataset, name=Path(config.save_dir).name)

    logging.info(f"args: {args}")

    # set gpu id
    logging.info(f"gpu_id: {args.gpu_id}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    # print pid
    logging.info("PID: {}".format(os.getpid()))

    # Setup random seed
    logging.info("Using seed: {}".format(args.seed))
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.seed_all()
    np.random.seed(args.seed)
    random.seed(args.seed)
    # set deterministic cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    train_dataset, validate_dataset = get_trainval_datasets(args.dataset, config.image_size, train_sample_ratio=args.train_sample_ratio, 
                                                            aug_json=args.aug_json, aug_sample_ratio=args.aug_sample_ratio, limit_aug_per_image=args.limit_aug_per_image,
                                                            special_aug=args.special_aug)

    num_classes = train_dataset.num_classes

    ##################################
    # Initialize model
    ##################################
    logs = {}
    start_epoch = 0
    net = WSDAN_CAL(num_classes=num_classes, M=config.num_attentions, net=config.net, pretrained=True)

    # feature_center: size of (#classes, #attention_maps * #channel_features)
    feature_center = torch.zeros(num_classes, config.num_attentions * net.num_features).cuda()

    if config.ckpt and os.path.isfile(config.ckpt):
        # Load ckpt and get state_dict
        checkpoint = torch.load(config.ckpt)

        # Get epoch and some logs
        logs = checkpoint['logs']
        start_epoch = int(logs['epoch']) # start from the beginning

        # Load weights
        state_dict = checkpoint['state_dict']
        net.load_state_dict(state_dict)
        logging.info('Network loaded from {}'.format(config.ckpt))
        logging.info('Network loaded from {} @ {} epoch'.format(config.ckpt, start_epoch))

        # load feature center
        if 'feature_center' in checkpoint:
            feature_center = checkpoint['feature_center'].cuda()
            logging.info('feature_center loaded from {}'.format(config.ckpt))

    logging.info('Network weights save to {}'.format(config.save_dir))

    ##################################
    # Use cuda
    ##################################
    net.cuda()

    learning_rate = config.learning_rate
    logging.info(f"Learning rate: {learning_rate}")
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)


    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                                               num_workers=config.workers, pin_memory=True, drop_last=True, shuffle=True)
    validate_loader = DataLoader(validate_dataset, batch_size=config.batch_size * 4,
                                               num_workers=config.workers, pin_memory=True, drop_last=True, shuffle=False)


    callback_monitor = 'val_{}'.format(raw_metric.name)
    callback = ModelCheckpoint(savepath=os.path.join(config.save_dir, config.model_name),
                                monitor=callback_monitor,
                                mode='max')
    if callback_monitor in logs:
        callback.set_best_score(logs[callback_monitor])
    else:
        callback.reset()
        logging.info('Start training: Total epochs: {}, Batch size: {}, Training size: {}, Validation size: {}'.
                     format(config.epochs, config.batch_size, len(train_dataset), len(validate_dataset)))
        logging.info('')

    for epoch in tqdm(range(start_epoch, config.epochs)):
        logging.info("\n")
        callback.on_epoch_begin()
        logs['epoch'] = epoch + 1
        logs['lr'] = optimizer.param_groups[0]['lr']
        logging.info(f"current lr: {optimizer.param_groups[0]['lr']}")

        logging.info('Epoch {:03d}, Learning Rate {:g}'.format(epoch + 1, optimizer.param_groups[0]['lr']))

        pbar = tqdm(total=len(train_loader), unit=' batches')
        pbar.set_description('Epoch {}/{}'.format(epoch + 1, config.epochs))

        train(epoch=epoch,
              logs=logs,
              data_loader=train_loader,
              net=net,
              feature_center=feature_center,
              optimizer=optimizer,
              pbar=pbar)

        if (epoch) % 5 == 0:
            validate(logs=logs,
                    data_loader=validate_loader,
                    net=net,
                    pbar=pbar,
                    epoch=epoch)

        callback.on_epoch_end(logs, net, feature_center=feature_center)
        pbar.close()

def adjust_learning(optimizer, epoch, iter):
    """Decay the learning rate based on schedule"""
    base_lr = config.learning_rate
    base_rate = 0.9
    base_duration = 2.0
    lr = base_lr * pow(base_rate, (epoch + iter) / base_duration)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(**kwargs):
    # Retrieve training configuration
    epoch = kwargs['epoch']
    logs = kwargs['logs']
    data_loader = kwargs['data_loader']
    net = kwargs['net']
    feature_center = kwargs['feature_center']
    optimizer = kwargs['optimizer']
    pbar = kwargs['pbar']

    # metrics initialization
    loss_container.reset()
    raw_metric.reset()
    crop_metric.reset()
    drop_metric.reset()

    # begin training
    start_time = time.time()
    net.train()
    batch_len = len(data_loader)
    for i, (X, y) in tqdm(enumerate(data_loader), total=len(data_loader), unit=' batches', desc='Epoch {}/{}'.format(epoch + 1, config.epochs)):
        float_iter = float(i) / batch_len
        adjust_learning(optimizer, epoch, float_iter)
        now_lr = optimizer.param_groups[0]['lr']

        optimizer.zero_grad()

        # obtain data for training
        X = X.cuda()
        y = y.cuda()

        y_pred_raw, y_pred_aux, feature_matrix, attention_map = net(X)

        # Update Feature Center
        feature_center_batch = F.normalize(feature_center[y], dim=-1)
        feature_center[y] += config.beta * (feature_matrix.detach() - feature_center_batch)

        ##################################
        # Attention Cropping
        ##################################
        with torch.no_grad():
            crop_images = batch_augment(X, attention_map[:, :1, :, :], mode='crop', theta=(0.4, 0.6), padding_ratio=0.1)
            drop_images = batch_augment(X, attention_map[:, 1:, :, :], mode='drop', theta=(0.2, 0.5))
        aug_images = torch.cat([crop_images, drop_images], dim=0)
        y_aug = torch.cat([y, y], dim=0)

        # crop images forward
        y_pred_aug, y_pred_aux_aug, _, _ = net(aug_images)

        y_pred_aux = torch.cat([y_pred_aux, y_pred_aux_aug], dim=0)
        y_aux = torch.cat([y, y_aug], dim=0)

        # loss
        batch_loss = cross_entropy_loss(y_pred_raw, y) / 3. + \
                     cross_entropy_loss(y_pred_aux, y_aux) * 3. / 3. + \
                     cross_entropy_loss(y_pred_aug, y_aug) * 2. / 3. + \
                     center_loss(feature_matrix, feature_center_batch)

        # backward
        batch_loss.backward()
        optimizer.step()

        # metrics: loss and top-1,5 error
        with torch.no_grad():
            epoch_loss = loss_container(batch_loss.item())
            epoch_raw_acc = raw_metric(y_pred_raw, y)
            epoch_crop_acc = crop_metric(y_pred_aug, y_aug)
            epoch_drop_acc = drop_metric(y_pred_aux, y_aux)

    # end of this epoch
    last_batch_info = 'Loss {:.4f}, Raw Acc ({:.2f}, {:.2f}), Aug Acc ({:.2f}, {:.2f}), Aux Acc ({:.2f}, {:.2f}), lr {:.5f}'.format(
        epoch_loss, epoch_raw_acc[0], epoch_raw_acc[1],
        epoch_crop_acc[0], epoch_crop_acc[1], epoch_drop_acc[0], epoch_drop_acc[1], now_lr)

    pbar.update()
    pbar.set_postfix_str(last_batch_info)

    # end of this epoch
    logs['train_{}'.format(loss_container.name)] = epoch_loss
    logs['train_raw_{}'.format(raw_metric.name)] = epoch_raw_acc
    logs['train_crop_{}'.format(crop_metric.name)] = epoch_crop_acc
    logs['train_drop_{}'.format(drop_metric.name)] = epoch_drop_acc
    logs['train_info'] = last_batch_info
    end_time = time.time()
    total_time = end_time - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    # wandb
    wandb.log({
        'train_loss': epoch_loss,
        'train_raw_acc': epoch_raw_acc[0],
        'train_crop_acc': epoch_crop_acc[0],
        'train_drop_acc': epoch_drop_acc[0],
        'train_lr': now_lr,
        'epoch': epoch,
        'total_epoch_time': total_time
    })

    # write log for this epoch
    logging.info('Train: {}'.format(last_batch_info))
    # time
    logging.info('Total epoch Time: {}'.format(total_time_str))
    


def validate(**kwargs):
    # Retrieve training configuration
    global best_acc
    epoch = kwargs['epoch']
    logs = kwargs['logs']
    data_loader = kwargs['data_loader']
    net = kwargs['net']
    pbar = kwargs['pbar']

    # metrics initialization
    loss_container.reset()
    raw_metric.reset()
    drop_metric.reset()

    # begin validation
    start_time = time.time()
    net.eval()
    logging.info('Start validating')
    with torch.no_grad():
        for i, (X, y) in tqdm(enumerate(data_loader)):
            # obtain data
            X = X.cuda()
            y = y.cuda()

            ##################################
            # Raw Image
            ##################################
            y_pred_raw, y_pred_aux, _, attention_map = net(X)

            crop_images3 = batch_augment(X, attention_map, mode='crop', theta=0.1, padding_ratio=0.05)
            y_pred_crop3, y_pred_aux_crop3, _, _ = net(crop_images3)

            ##################################
            # Final prediction
            ##################################
            y_pred = (y_pred_raw + y_pred_crop3) / 2.
            y_pred_aux = (y_pred_aux + y_pred_aux_crop3) / 2.

            # loss
            batch_loss = cross_entropy_loss(y_pred, y)
            batch_loss = batch_loss.data
            epoch_loss = loss_container(batch_loss.item())

            y_pred = y_pred
            y_pred_aux = y_pred_aux
            y = y

            # metrics: top-1,5 error
            epoch_acc = raw_metric(y_pred, y)
            aux_acc = drop_metric(y_pred_aux, y)

    # end of validation
    logs['val_{}'.format(loss_container.name)] = epoch_loss
    logs['val_{}'.format(raw_metric.name)] = epoch_acc
    end_time = time.time()
    total_time = end_time - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    # wandb
    wandb.log({
        'val_loss': epoch_loss,
        'val_raw_acc': epoch_acc[0],
        'val_crop_acc': aux_acc[0],
        'val_drop_acc': aux_acc[0],
        'epoch': epoch,
        'total_val_time': total_time
    })
    
    batch_info = 'Val Loss {:.4f}, Val Acc ({:.2f}, {:.2f})'.format(epoch_loss, epoch_acc[0], epoch_acc[1])

    pbar.set_postfix_str('{}, {}'.format(logs['train_info'], batch_info))

    if epoch_acc[0] > best_acc:
        best_acc = epoch_acc[0]
        save_model(net, logs, 'model_bestacc.pth')

    if aux_acc[0] > best_acc:
        best_acc = aux_acc[0]
        save_model(net, logs, 'model_bestacc.pth')

    if epoch % 10 == 0:
        save_model(net, logs, 'model_epoch%d.pth' % epoch)


    batch_info = 'Val Loss {:.4f}, Val Acc ({:.2f}, {:.2f}), Val Aux Acc ({:.2f}, {:.2f}), Best {:.2f}'.format(
        epoch_loss, epoch_acc[0], epoch_acc[1], aux_acc[0], aux_acc[1], best_acc)
    logging.info(batch_info)

    # write log for this epoch
    logging.info('Valid: {}'.format(batch_info))
    logging.info('Total Val Time: {}'.format(total_time_str))

    logging.info('')

def save_model(net, logs, ckpt_name):
    torch.save({'logs': logs, 'state_dict': net.state_dict()}, config.save_dir + '/model_bestacc.pth')

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    class Args:
        def __init__(self):
            self.seed = 0
            self.gpu_id = 0
            self.logdir = 'logs'
            self.dataset = 'planes'
            self.aug_json = None
            self.aug_sample_ratio = None
            self.limit_aug_per_image = None
            self.train_sample_ratio = 1.0


    """
    use this:
    --gpu_id 0 \
    --seed 1 \
    --train_sample_ratio 1.0 \
    --logdir logs/planes/base_seed_1_sample_ratio_1.0_resnet_50 \
    --dataset planes
    """

    DEBUG = False
    if DEBUG:
        args = Args()
        args.seed = 1
        args.gpu_id = 0
        args.logdir = 'logs/planes/base_seed_1_sample_ratio_1.0_resnet_50'
        args.dataset = 'planes'
        args.train_sample_ratio = 1.0

    main()

