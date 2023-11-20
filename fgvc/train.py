import datetime
import os
from pathlib import Path
import traceback
import numpy as np
import time
import logging
import warnings
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import random
import argparse
import wandb

from models import WSDAN_CAL
from util import CenterLoss, AverageMeter, TopKAccuracyMetric, ModelCheckpoint, batch_augment
from datasets import get_datasets


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--logdir', type=str, default='logs')
parser.add_argument('--dataset', type=str, default='planes')
parser.add_argument('--epochs', type=int, default=None)
parser.add_argument('--learning_rate', type=float, default=None)
parser.add_argument('--batch_size', type=int, default=None)
parser.add_argument('--weight_decay', type=float, default=None)
# augmentation options
parser.add_argument("--aug_json", type=str, default=None,
                    help="path to augmentation json file")
parser.add_argument("--aug_sample_ratio", type=float, default=None,
                    help="ratio to augment the original image")
parser.add_argument("--limit_aug_per_image", type=int, default=None,
                    help="limit augmentations per image, default None, which is take all")
parser.add_argument("--stop_aug_after_epoch", type=int, default=None,
                    help="stop augmenting after this epoch")
parser.add_argument("--special_aug", type=str, default="classic",
                    help="special classic augmentation to use, out of randaug, autoaug")
# add arg to take only some amount for the train set
parser.add_argument("--train_sample_ratio", type=float, default=1.0,
                    help="ratio of train set to take")
parser.add_argument("--dont_use_wsdan", action="store_true", default=False,
                    help="Don't use wsdan augmentation")
parser.add_argument("--use_cutmix", action="store_true", default=False,
                    help="Use cutmix augmentation")
parser.add_argument("--use_target_soft_cross_entropy", action="store_true", default=False,
                    help="Use soft target cross entropy loss")

args = parser.parse_args()

if args.dataset == 'planes':
    import configs.config_planes as config
elif args.dataset == 'cars':
    import configs.config_cars as config
elif args.dataset == 'dtd':
    import configs.config_dtd as config
else:
    raise ValueError('Unsupported dataset {}'.format(args.dataset))


    
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

best_val_acc = 0.0


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
    try:
        config.save_dir = init_logging(args.logdir)

        # only if stated in args:
        config.epochs = args.epochs if args.epochs else config.epochs
        config.learning_rate = args.learning_rate if args.learning_rate else config.learning_rate
        config.batch_size = args.batch_size if args.batch_size else config.batch_size

        if not DONT_WANDB:
            wandb.init(project=f"CAL-aug-exp-{args.dataset}", name=Path(config.save_dir).name)

        args.net = config.net
        args.image_size = config.image_size
        args.num_attentions = config.num_attentions
        args.beta = config.beta
        if not args.learning_rate:
            args.learning_rate = config.learning_rate
        if not args.batch_size:
            args.batch_size = config.batch_size
        if not args.weight_decay:
            args.weight_decay = config.weight_decay

        logging.info(f"args: {args}")
        # log args to wandb
        if not DONT_WANDB:
            wandb.config.update(args)

        # set gpu id
        logging.info(f"gpu_id: {args.gpu_id}")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

        # print pid
        logging.info("PID: {}".format(os.getpid()))

        if args.seed:
            # Setup random seed
            logging.info("Using seed: {}".format(args.seed))
            torch.manual_seed(args.seed)
            random.seed(args.seed)   
            np.random.seed(args.seed)
            torch.cuda.seed_all()
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        if args.dont_use_wsdan:
            logging.info("Not using wsdan augmentation")

        train_dataset, validate_dataset, test_dataset = get_datasets(args.dataset, config.image_size, train_sample_ratio=args.train_sample_ratio, 
                                                                aug_json=args.aug_json, aug_sample_ratio=args.aug_sample_ratio, limit_aug_per_image=args.limit_aug_per_image,
                                                                special_aug=args.special_aug, use_cutmix=args.use_cutmix)

        num_classes = train_dataset.num_classes

        ##################################
        # Initialize model
        ##################################
        logs = {}
        start_epoch = 0
        net = WSDAN_CAL(num_classes=num_classes, M=config.num_attentions, net=config.net, pretrained=True)

        # init model for soft target cross entropy
        if args.use_target_soft_cross_entropy:
            from utils.utils import PlanesUtils, CarsUtils
            import clip
            import losses
            # the import here because in utils.py there is an initialization of a device, which interrupt the cuda device setting in the start of this script

            logging.info("IMPORTANT: Using soft target cross entropy loss")
            global soft_target_cross_entropy, clip_selector, image_stem_to_class_dict

            soft_target_cross_entropy = losses.SoftTargetCrossEntropy_T()  # input is logits, CLIP logits
            model, preprocess = clip.load('RN50', 'cuda', jit=False)
            if args.dataset == "planes":
                planes = PlanesUtils()
                classnames = planes.get_classes()
                prompts = ["a photo of a " + name + ", a type of aircraft." for name in classnames]
                image_stem_to_class_dict = planes.get_image_stem_to_class_dict()  # id --> class
            elif args.dataset == "cars":
                cars = CarsUtils()
                classnames = cars.get_classes()
                prompts = ["a photo of a " + name + ", a type of car." for name in classnames]
                image_stem_to_class_dict = cars.get_image_stem_to_class_dict()  # id --> class

            
            tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()
            clip_selector = losses.CLIP_selector(model, preprocess, preprocess, tokenized_prompts)

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
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size * 4, 
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
            if args.aug_json and args.stop_aug_after_epoch and epoch >= args.stop_aug_after_epoch:
                train_dataset.stop_aug = True
                logging.info(f"Reached args.stop_aug_after_epoch={args.stop_aug_after_epoch}, stopped augmentation")
                
            logging.info("\n")
            callback.on_epoch_begin()
            logs['epoch'] = epoch + 1
            logs['lr'] = optimizer.param_groups[0]['lr']

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

            if (epoch) % 5 == 0 or epoch == config.epochs - 1:
                validate(logs=logs,
                        data_loader=validate_loader,
                        net=net,
                        pbar=pbar,
                        epoch=epoch,
                        is_test=False)
                validate(logs=logs,
                        data_loader=test_loader,
                        net=net,
                        pbar=pbar,
                        epoch=epoch,
                        is_test=True)

            callback.on_epoch_end(logs, net, feature_center=feature_center)
            pbar.close()
    
    except KeyboardInterrupt:
        logging.info('KeyboardInterrupt at epoch {}'.format(epoch + 1))

    except Exception as e:
        logging.info(traceback.format_exc())
        raise


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
        if not args.dont_use_wsdan:  # use wsdan augmentation loss
            REGULAR_CE_RATIO = 0.5
            if args.use_target_soft_cross_entropy:
                batch_loss = center_loss(feature_matrix, feature_center_batch)  # not realated to CE loss 
                batch_loss += REGULAR_CE_RATIO * ( cross_entropy_loss(y_pred_raw, y) / 3. + \
                            cross_entropy_loss(y_pred_aux, y_aux) * 3. / 3. + \
                            cross_entropy_loss(y_pred_aug, y_aug) * 2. / 3. )  # regular CE loss
                
                # add soft target cross entropy loss
                global soft_target_cross_entropy, clip_selector
                logits = clip_selector(X)

                logits_aug = torch.cat([logits, logits], dim=0)  # same as y_aug
                logits_aux = torch.cat([logits, logits_aug], dim=0)  # same as y_aux
                batch_loss += (1 - REGULAR_CE_RATIO) * ( soft_target_cross_entropy(y_pred_raw, logits) / 3. + \
                            soft_target_cross_entropy(y_pred_aux, logits_aux) * 3. / 3. + \
                            soft_target_cross_entropy(y_pred_aug, logits_aug) * 2. / 3. ) # soft target CE loss

            else: # regular loss with normal CE
                batch_loss = cross_entropy_loss(y_pred_raw, y) / 3. + \
                            cross_entropy_loss(y_pred_aux, y_aux) * 3. / 3. + \
                            cross_entropy_loss(y_pred_aug, y_aug) * 2. / 3. + \
                            center_loss(feature_matrix, feature_center_batch)
        else:
            # not divinding by 3 because not using 3 diff losses. This is not efficient because still computing it, just no using it.
            batch_loss = cross_entropy_loss(y_pred_raw, y) + center_loss(feature_matrix, feature_center_batch)

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

    if not DONT_WANDB:
        # wandb
        wandb.log({
            'train_loss': epoch_loss,
            'train_raw_acc': epoch_raw_acc[0],
            'train_crop_acc': epoch_crop_acc[0],
            'train_drop_acc': epoch_drop_acc[0],
            'train_lr': now_lr,
            'epoch': epoch,
            'epoch_time': total_time
        })

    # write log for this epoch
    logging.info('Train: {}'.format(last_batch_info))
    # time
    logging.info('Total epoch Time: {}'.format(total_time_str))
    


def validate(**kwargs):
    # Retrieve training configuration
    global best_val_acc
    epoch = kwargs['epoch']
    logs = kwargs['logs']
    data_loader = kwargs['data_loader']
    net = kwargs['net']
    pbar = kwargs['pbar']
    is_test = kwargs['is_test']
    val_str = 'test' if is_test else 'val'

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
    logs[f'{val_str}_{loss_container.name}'] = epoch_loss
    logs[f'{val_str}_{raw_metric.name}'] = epoch_acc
    end_time = time.time()
    total_time = end_time - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    
    batch_info = f'{val_str} Loss {epoch_loss:.4f}, {val_str} Acc ({epoch_acc[0]:.2f}, {epoch_acc[1]:.2f})'

    pbar.set_postfix_str('{}, {}'.format(logs['train_info'], batch_info))

    if epoch_acc[0] > best_val_acc:
        best_val_acc = epoch_acc[0]
        save_model(net, logs, 'model_bestacc.pth')

    if not DONT_WANDB:
        # wandb
        wandb.log({
            f'{val_str}_loss': epoch_loss,
            f'{val_str}_raw_acc': epoch_acc[0],
            f'{val_str}_best_raw_acc': best_val_acc,
            f'{val_str}_crop_acc': aux_acc[0],
            f'{val_str}_drop_acc': aux_acc[0],
            'epoch': epoch,
            f'{val_str}_time': total_time
        })

    # if aux_acc[0] > best_acc:
    #     best_acc = aux_acc[0]
    #     save_model(net, logs, 'model_bestacc.pth')

    # if epoch % 10 == 0:
    #     save_model(net, logs, 'model_epoch%d.pth' % epoch)


    batch_info = 'Val Loss {:.4f}, Val Acc ({:.2f}, {:.2f}), Val Aux Acc ({:.2f}, {:.2f}), Best {:.2f}'.format(
        epoch_loss, epoch_acc[0], epoch_acc[1], aux_acc[0], aux_acc[1], best_val_acc)
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
            self.seed = 2
            self.gpu_id = 1
            self.epochs = 100
            self.logdir = 'logs/planes/test_delete_me'
            self.dataset = 'planes'
            self.learning_rate = 0.001
            self.batch_size = 10
            self.weight_decay = 1e-5
            self.train_sample_ratio = 1.0
            self.aug_json = "/mnt/raid/home/eyal_michaeli/datasets/aug_json_files/planes/ip2p/merged_blip_diffusion_v0-4x.json"
            self.aug_sample_ratio = 0.5
            self.limit_aug_per_image = None
            self.special_aug = None
            self.stop_aug_after_epoch = 100
            self.use_target_soft_cross_entropy = 0
            self.dont_use_wsdan = False
            self.use_cutmix = False


    """
    use this:

    """
    DONT_WANDB = False
    DEBUG = 0
    if DEBUG:
        DONT_WANDB = True
        args = Args()


    main()

