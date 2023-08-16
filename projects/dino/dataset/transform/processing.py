# Copyright (c) OpenMMLab. All rights reserved.
import random

from mmcv.transforms import RandomApply  # noqa: E501
from mmcv.transforms import BaseTransform, Compose, RandomFlip, RandomGrayscale

from mmpretrain.datasets.transforms import (ColorJitter, GaussianBlur,
                                            RandomResizedCrop, Solarize)
from mmpretrain.registry import TRANSFORMS


@TRANSFORMS.register_module()
class DINOMultiCrop(BaseTransform):
    """Multi-crop transform for DINO.

    This module applies the multi-crop transform for DINO.

    Args:
        global_crops_scale (int): Scale of global crops.
        local_crops_scale (int): Scale of local crops.
        local_crops_number (int): Number of local crops.
    """

    def __init__(self, global_crops_scale: int, local_crops_scale: int,
                 local_crops_number: int) -> None:
        super().__init__()
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale

        flip_and_color_jitter = Compose([
            RandomFlip(prob=0.5, direction='horizontal'),
            RandomApply([
                ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
            ],
                        prob=0.8),
            RandomGrayscale(
                prob=0.2,
                keep_channels=True,
                channel_weights=(0.114, 0.587, 0.2989),
            )
        ])

        self.global_transform_1 = Compose([
            RandomResizedCrop(
                224,
                crop_ratio_range=global_crops_scale,
                interpolation='bicubic'),
            flip_and_color_jitter,
            GaussianBlur(prob=1.0, radius=random.uniform(0.1, 2.0)),
        ])

        self.global_transform_2 = Compose([
            RandomResizedCrop(
                224,
                crop_ratio_range=global_crops_scale,
                interpolation='bicubic'),
            flip_and_color_jitter,
            GaussianBlur(prob=1.0, radius=random.uniform(0.1, 2.0)),
            Solarize(thr=128, prob=0.2),
        ])

        self.local_crops_number = local_crops_number
        self.local_transform = Compose([
            RandomResizedCrop(
                96,
                crop_ratio_range=local_crops_scale,
                interpolation='bicubic'),
            flip_and_color_jitter,
            GaussianBlur(prob=1.0, radius=random.uniform(0.1, 2.0)),
        ])

    def transform(self, results: dict) -> dict:
        ori_img = results['img']
        crops = []
        results['img'] = ori_img
        crops.append(self.global_transform_1(results)['img'])
        results['img'] = ori_img
        crops.append(self.global_transform_2(results)['img'])
        for _ in range(self.local_crops_number):
            results['img'] = ori_img
            crops.append(self.local_transform(results)['img'])
        results['img'] = crops
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(global_crops_scale = {self.global_crops_scale}, '
        repr_str += f'local_crops_scale = {self.local_crops_scale}, '
        repr_str += f'local_crop_number = {self.local_crops_number})'
        return repr_str
