# Copyright (c) OpenMMLab. All rights reserved.
from .base_dataset import BaseDataset
from .builder import build_dataset
from .cifar import CIFAR10, CIFAR100
from .cub import CUB
from .custom import CustomDataset
from .dataset_wrappers import KFoldDataset
from .imagenet import ImageNet, ImageNet21k
from .inshop import InShop
from .mnist import MNIST, FashionMNIST
from .multi_label import MultiLabelDataset
from .multi_task import MultiTaskDataset
from .samplers import *  # noqa: F401,F403
from .transforms import *  # noqa: F401,F403
from .voc import VOC

__all__ = [
    'BaseDataset', 'ImageNet', 'CIFAR10', 'CIFAR100', 'MNIST', 'FashionMNIST',
    'VOC', 'build_dataset', 'ImageNet21k', 'KFoldDataset', 'CUB',
    'CustomDataset', 'MultiLabelDataset', 'MultiTaskDataset', 'InShop'
]
