import logging

from .aircraft_dataset import Planes
from .cub_dataset import CUB
from .car_dataset import Cars
from .dtd_dataset import DTDataset
from .arch_dataset import ArchDataset
from .compcars_dataset import CompCars


from util import get_transform
from cutmix.cutmix import CutMix


def get_datasets(dataset, resize, train_sample_ratio=1.0, aug_json=None, aug_sample_ratio=None, limit_aug_per_image=None, special_aug=None, use_cutmix=False):
    train_transform = get_transform(resize=resize, phase='train', special_aug=special_aug)
    val_transform = get_transform(resize=resize, phase='val')
    if dataset == 'planes':
        train, val, test = Planes(split='train', transform=train_transform, train_sample_ratio=train_sample_ratio, aug_json=aug_json, aug_sample_ratio=aug_sample_ratio, limit_aug_per_image=limit_aug_per_image), Planes(split='val', transform=val_transform), Planes(split='test', transform=val_transform)
    elif dataset == 'cub':
        train, val, test =  CUB(split='train', transform=train_transform, train_sample_ratio=train_sample_ratio, aug_json=aug_json, aug_sample_ratio=aug_sample_ratio, limit_aug_per_image=limit_aug_per_image), CUB(split='val', transform=val_transform), None
    elif dataset == 'cars':
        train, val, test =  Cars(split='train', transform=train_transform, train_sample_ratio=train_sample_ratio, aug_json=aug_json, aug_sample_ratio=aug_sample_ratio, limit_aug_per_image=limit_aug_per_image), Cars(split='val', transform=val_transform), None
    elif dataset == 'dtd':
        train, val, test =  DTDataset(split='train', transform=train_transform, train_sample_ratio=train_sample_ratio, aug_json=aug_json, aug_sample_ratio=aug_sample_ratio, limit_aug_per_image=limit_aug_per_image), DTDataset(split='val', transform=val_transform), DTDataset(split='test', transform=val_transform)
    elif dataset == 'arch_dataset':
        train, val, test =  ArchDataset(split='train', transform=train_transform, train_sample_ratio=train_sample_ratio, aug_json=aug_json, aug_sample_ratio=aug_sample_ratio, limit_aug_per_image=limit_aug_per_image), ArchDataset(split='val', transform=val_transform), ArchDataset(split='test', transform=val_transform)
    elif dataset == 'compcars':
        train, val, test =  CompCars(split='train', transform=train_transform, train_sample_ratio=train_sample_ratio, aug_json=aug_json, aug_sample_ratio=aug_sample_ratio, limit_aug_per_image=limit_aug_per_image), CompCars(split='val', transform=val_transform), None
    elif dataset == 'compcars-parts':
        train, val, test =  CompCars(split='train', transform=train_transform, train_sample_ratio=train_sample_ratio, aug_json=aug_json, aug_sample_ratio=aug_sample_ratio, limit_aug_per_image=limit_aug_per_image, dataset_type='parts'), CompCars(split='val', transform=val_transform, dataset_type='parts'), None
    else:
        raise ValueError('Unsupported dataset {}'.format(dataset))

    if use_cutmix or special_aug == "cutmix":
        logging.info("Using CutMix augmentation")
        # we used the same params for cutmix as ALIA, DA-Fusion
        # DA-Fusion: https://github.com/brandontrabucco/da-fusion/blob/main/train_classifier.py#L134
        return CutMix(train, num_class=train.num_classes, beta=1.0, prob=0.5, num_mix=2).dataset, val, test
    else:
        return train, val, test