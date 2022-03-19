'''
downloaded and adapted from : https://github.com/facebookresearch/deit
'''

import os
import json

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader
from torchvision.transforms import InterpolationMode

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

from paths import DATAROOT

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

def build_dataset(dataset, is_train, input_size, erasing_aug):

    if dataset == 'imagenet':
        re_prob = 0.25 if erasing_aug else 0
        transform = build_transform(is_train, True, 0, input_size, re_prob)
        partition = 'train' if is_train else 'val'
        dataset = datasets.ImageFolder(DATAROOT/'imagenet'/partition, transform=transform)
        nb_classes = 1000

    elif dataset == 'fMoW':
        re_prob = 0.25 if erasing_aug else 0
        transform = build_transform(is_train, False, 0.5, input_size, re_prob)
        partition = 'train_bbox' if is_train else 'test_bbox'
        dataset = datasets.ImageFolder(DATAROOT/'fMoW'/partition, transform=transform)
        nb_classes = 62

    dataset.transform.transforms[0].interpolation=InterpolationMode.BICUBIC

    return dataset, nb_classes


def build_transform(is_train, center_crop, vflip, input_size, re_prob):
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=input_size,
            is_training=True,
            color_jitter=0.4,
            auto_augment='rand-m9-mstd0.5-inc1',
            interpolation='bicubic',
            re_prob=re_prob,
            re_mode='pixel',
            re_count=1,
            vflip=vflip,
        )
        return transform

    t = []
    if center_crop:
        size = int((256 / 224) * input_size)
        t.append(
            transforms.Resize(size, interpolation=InterpolationMode.BICUBIC),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))
    else:
        t.append(
            transforms.Resize(input_size, interpolation=InterpolationMode.BICUBIC),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))
 

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)

#imagenettrainset, nclass = build_dataset('imagenet', DATAROOT, is_train=True)
#imagenetvalset,   nclass = build_dataset('imagenet', DATAROOT, is_train=False)
