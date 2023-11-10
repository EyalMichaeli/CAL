import logging
from .aircraft_dataset import Planes
from .bird_dataset import BirdDataset
from .car_dataset import Cars
from util import get_transform
from cutmix.cutmix import CutMix


def get_trainval_datasets(dataset, resize, train_sample_ratio=1.0, aug_json=None, aug_sample_ratio=None, limit_aug_per_image=None, special_aug=None, use_cutmix=False):
    train_transform = get_transform(resize=resize, phase='train', special_aug=special_aug)
    val_transform = get_transform(resize=resize, phase='val')
    if dataset == 'planes':
        train, val = Planes(split='train', transform=train_transform, train_sample_ratio=train_sample_ratio, aug_json=aug_json, aug_sample_ratio=aug_sample_ratio, limit_aug_per_image=limit_aug_per_image), Planes(split='val', transform=val_transform)
    elif dataset == 'cub':
        train, val =  BirdDataset(phase='train', resize=resize), BirdDataset(phase='val', resize=resize)
    elif dataset == 'cars':
        train, val =  Cars(split='train', transform=train_transform, train_sample_ratio=train_sample_ratio, aug_json=aug_json, aug_sample_ratio=aug_sample_ratio, limit_aug_per_image=limit_aug_per_image), Cars(split='val', transform=val_transform)
    else:
        raise ValueError('Unsupported dataset {}'.format(dataset))

    if use_cutmix or special_aug == "cutmix":
        logging.info("Using CutMix augmentation")
        # we used the same params for cutmix as ALIA, DA-Fusion
        # DA-Fusion: https://github.com/brandontrabucco/da-fusion/blob/main/train_classifier.py#L134
        return CutMix(train, num_class=train.num_classes, beta=1.0, prob=0.5, num_mix=2).dataset, val
    else:
        return train, val