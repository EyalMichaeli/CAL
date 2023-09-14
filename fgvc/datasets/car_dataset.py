""" Stanford Cars (Car) Dataset """
import json
import logging
import os
from pathlib import Path
import pdb
import random
from typing import Callable, Optional
import warnings
from PIL import Image
import numpy as np
from scipy.io import loadmat
from torch.utils.data import Dataset
from utils import get_transform
from torchvision.datasets import StanfordCars


class Cars(StanfordCars):
    def __init__(self, root: str = "/mnt/raid/home/eyal_michaeli/datasets/", split: str = "train", transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, 
                 download: bool = False, train_sample_ratio: float = 1.0, aug_json: str = None, aug_sample_ratio: float = None, limit_aug_per_image: int = None):
        super().__init__(root=root, split="train" if split =="train" else "test", transform=transform, target_transform=target_transform, download=download)
        assert split in ['train', 'val', 'test']

        self.is_train = "train" in split
        self.original_data_length = len(self._samples)

        self._image_files = [sapmle[0] for sapmle in self._samples]
        self._labels = [sapmle[1] for sapmle in self._samples]

        # use only a subset of the images for training, if train_sample_ratio < 1
        if self.is_train and train_sample_ratio < 1:
            self._image_files, self._labels = self.use_subset(train_sample_ratio, self._image_files, self._labels)

        self.num_classes = len(set(self.classes))
        logging.info("CARS {}".format(split.upper()))
        logging.info("LEN DATASET: {}".format(len(self._image_files)))
        logging.info("NUM CLASSES: {}".format(self.num_classes))

        if self.is_train and aug_json:
            self.init_augmentation(aug_json, aug_sample_ratio, limit_aug_per_image)
        else:
            self.aug_json = None
            logging.info("Not using DiffusionAug images")
    

    def use_subset(self, sample_ratio, images_paths, labels):
        assert sample_ratio > 0 and sample_ratio <= 1
        subset_size = int(len(images_paths) * sample_ratio)
        indices_to_take = np.random.choice(len(images_paths), subset_size, replace=False)
        
        logging.info(f"With ratio {sample_ratio}, using only {subset_size} images for training, out of {len(images_paths)}")
        
        selected_images = np.array(images_paths)[indices_to_take]
        selected_labels = np.array(labels)[indices_to_take]
        
        return list(selected_images), list(selected_labels)
    


    def init_augmentation(self, aug_json, aug_sample_ratio, limit_aug_per_image):
        self.limit_aug_per_image = limit_aug_per_image
        assert aug_sample_ratio is not None
        assert aug_sample_ratio > 0 and aug_sample_ratio <= 1
        with open(aug_json, 'r') as f:
            self.aug_json = json.load(f)
        # leave only keys that thier values (which is a list) is not empty
        self.aug_json = {k: v[:self.limit_aug_per_image] for k, v in self.aug_json.items() if v}
        assert len(self.aug_json) > 0, "aug_json is empty"

        if self.limit_aug_per_image is not None:
            logging.info(f"Using only {self.limit_aug_per_image} augmented images per original image (or less if filtered out))")
            logging.info(f"For example: {list(self.aug_json.values())[0]}, which is indeed of length {len(list(self.aug_json.values())[0])}")
        assert self.limit_aug_per_image is None or self.limit_aug_per_image >= len(list(self.aug_json.values())[0]), "limit_aug_per_image must be >= the number of augmented images per original image"
        
        self.aug_sample_ratio = aug_sample_ratio
        self.times_used_orig_images = 0
        self.times_used_aug_images = 0

        logging.info(f"Using augmented images with ratio {aug_sample_ratio}")
        logging.info(f"There are {len(self.aug_json)} augmented images, out of {self.original_data_length} original images, \n which is {round(len(self.aug_json)/self.original_data_length, 2)*100}% of the original images")
        logging.info(f"json file: {aug_json}")


    def __len__(self):
        return len(self._image_files)


    def get_aug_image(self, image_path, idx):
        ratio_used_aug = 0
        if random.random() < self.aug_sample_ratio:
            original_image_path = image_path
            aug_img_files = self.aug_json.get(Path(image_path).name, [image_path])  # if image_path is not in aug_json, returns image_path
            aug_img_files = [image_path] if len(aug_img_files) == 0 else aug_img_files  # if image_path key in the json returns an enpty list, use current image_path
            image_path = random.choice(aug_img_files)
            if original_image_path == image_path:  # didn't use augmented image
                #print("Augmented image not found in aug_json")
                self.times_used_orig_images += 1

            else:  # used augmented image
                #print(f"Using Augmented image found in aug_json: {image_path}")
                self.times_used_aug_images += 1
            pass

        else:
            self.times_used_orig_images += 1

        ratio_used_aug = self.times_used_aug_images / (self.times_used_orig_images + self.times_used_aug_images)

        if idx % 100 == 0 and ratio_used_aug < self.aug_sample_ratio / 3:  # check every 100 iters. e.g, if aug_sample_ratio = 0.3, then ratio_used_aug should not be less than 0.1
            warn = f"Using augmented images might be lacking, ratio: {ratio_used_aug:.4f} when it should be around {self.aug_sample_ratio}"
            warnings.warn(warn)
            logging.info(f"self.times_used_aug_images = {self.times_used_aug_images}, self.times_used_orig_images = {self.times_used_orig_images}")
            
        # every 500 iters, print the ratio of original images to augmented images
        if idx % 1000 == 0:
            logging.info(f"Used augmented images {(ratio_used_aug*100):.4f}% of the time")
        return image_path


    def __getitem__(self, idx):
        image_path, label = self._image_files[idx], self._labels[idx]

        if self.is_train and self.aug_json:
            image_path = self.get_aug_image(image_path, idx).replace("instruct-pix2pix", "Eyal-ip2p")

        img = Image.open(image_path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        return img, label
