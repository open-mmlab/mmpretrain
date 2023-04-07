# Copyright (c) OpenMMLab. All rights reserved.
from .base_dataset import BaseDataset
from .builder import build_dataset
from .caltech101 import Caltech101
from .cifar import CIFAR10, CIFAR100
from .cub import CUB
from .custom import CustomDataset
from .dataset_wrappers import KFoldDataset
from .dtd import DescribableTexture
from .fgvc_aircraft import FGVC_Aircraft
from .food101 import Food101
from .imagenet import ImageNet, ImageNet21k
from .inshop import InShop
from .mnist import MNIST, FashionMNIST
from .multi_label import MultiLabelDataset
from .multi_task import MultiTaskDataset
from .oxford102flowers import Oxford102Flowers
from .oxford_iiit_pets import Oxford_IIIT_Pets
from .places205 import Places205
from .samplers import *  # noqa: F401,F403
from .stanford_cars import StanfordCars
from .sun397 import SUN397
from .transforms import *  # noqa: F401,F403
from .voc import VOC

__all__ = [
    'BaseDataset', 'ImageNet', 'CIFAR10', 'CIFAR100', 'MNIST', 'FashionMNIST',
    'VOC', 'build_dataset', 'ImageNet21k', 'KFoldDataset', 'CUB',
    'CustomDataset', 'MultiLabelDataset', 'MultiTaskDataset', 'InShop',
    'Places205', 'Oxford102Flowers', 'Oxford_IIIT_Pets', 'DescribableTexture',
    'FGVC_Aircraft', 'StanfordCars', 'SUN397', 'Caltech101', 'Food101'
]
