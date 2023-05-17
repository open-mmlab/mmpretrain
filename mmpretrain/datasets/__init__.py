# Copyright (c) OpenMMLab. All rights reserved.
from .base_dataset import BaseDataset
from .builder import build_dataset
from .caltech101 import Caltech101
from .cifar import CIFAR10, CIFAR100
from .cub import CUB
from .custom import CustomDataset
from .dataset_wrappers import KFoldDataset
from .dsdl import DSDLClsDataset
from .dtd import DTD
from .fgvcaircraft import FGVCAircraft
from .flowers102 import Flowers102
from .food101 import Food101
from .imagenet import ImageNet, ImageNet21k
from .inshop import InShop
from .mnist import MNIST, FashionMNIST
from .multi_label import MultiLabelDataset
from .multi_task import MultiTaskDataset
from .oxfordiiitpet import OxfordIIITPet
from .places205 import Places205
from .samplers import *  # noqa: F401,F403
from .stanfordcars import StanfordCars
from .sun397 import SUN397
from .transforms import *  # noqa: F401,F403
from .voc import VOC

__all__ = [
    'BaseDataset', 'ImageNet', 'CIFAR10', 'CIFAR100', 'MNIST', 'FashionMNIST',
    'VOC', 'build_dataset', 'ImageNet21k', 'KFoldDataset', 'CUB',
    'CustomDataset', 'MultiLabelDataset', 'MultiTaskDataset', 'InShop',
    'Places205', 'Flowers102', 'OxfordIIITPet', 'DTD', 'FGVCAircraft',
    'StanfordCars', 'SUN397', 'Caltech101', 'Food101', 'Places205',
    'DSDLClsDataset'
]
