# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.transforms import (CenterCrop, LoadImageFromFile, Normalize,
                             RandomFlip, RandomGrayscale, RandomResize, Resize)

from mmpretrain.registry import TRANSFORMS
from .auto_augment import (AutoAugment, AutoContrast, BaseAugTransform,
                           Brightness, ColorTransform, Contrast, Cutout,
                           Equalize, GaussianBlur, Invert, Posterize,
                           RandAugment, Rotate, Sharpness, Shear, Solarize,
                           SolarizeAdd, Translate)
from .formatting import (Collect, NumpyToPIL, PackInputs, PackMultiTaskInputs,
                         PILToNumpy, Transpose)
from .processing import (Albumentations, BEiTMaskGenerator, CleanCaption,
                         ColorJitter, EfficientNetCenterCrop,
                         EfficientNetRandomCrop, Lighting, RandomCrop,
                         RandomErasing, RandomResizedCrop,
                         RandomResizedCropAndInterpolationWithTwoPic,
                         RandomTranslatePad, ResizeEdge, SimMIMMaskGenerator)
from .utils import get_transform_idx, remove_transform
from .wrappers import ApplyToList, MultiView

for t in (CenterCrop, LoadImageFromFile, Normalize, RandomFlip,
          RandomGrayscale, RandomResize, Resize):
    TRANSFORMS.register_module(module=t)

__all__ = [
    'NumpyToPIL', 'PILToNumpy', 'Transpose', 'Collect', 'RandomCrop',
    'RandomResizedCrop', 'Shear', 'Translate', 'Rotate', 'Invert',
    'ColorTransform', 'Solarize', 'Posterize', 'AutoContrast', 'Equalize',
    'Contrast', 'Brightness', 'Sharpness', 'AutoAugment', 'SolarizeAdd',
    'Cutout', 'RandAugment', 'Lighting', 'ColorJitter', 'RandomErasing',
    'PackInputs', 'Albumentations', 'EfficientNetRandomCrop',
    'EfficientNetCenterCrop', 'ResizeEdge', 'BaseAugTransform',
    'PackMultiTaskInputs', 'GaussianBlur', 'BEiTMaskGenerator',
    'SimMIMMaskGenerator', 'CenterCrop', 'LoadImageFromFile', 'Normalize',
    'RandomFlip', 'RandomGrayscale', 'RandomResize', 'Resize', 'MultiView',
    'ApplyToList', 'CleanCaption', 'RandomTranslatePad',
    'RandomResizedCropAndInterpolationWithTwoPic', 'get_transform_idx',
    'remove_transform'
]
