# Copyright (c) OpenMMLab. All rights reserved.
import math
import random
from typing import Optional, Tuple

import numpy as np
from mmcv.transforms.base import BaseTransform

from mmpretrain.registry import TRANSFORMS


@TRANSFORMS.register_module()
class MaskFeatMaskGenerator3D(BaseTransform):
    """Generate mask for video.

    Added Keys:

    - mask

    This module is borrowed from
    https://github.com/facebookresearch/SlowFast/blob/main/slowfast/datasets/transform.py

    Args:
        input_size (int): The size of input video.
        num_masking_patches (int): The number of patches to be masked.
        min_num_patches (int): The minimum number of patches to be masked
            in the process of generating mask. Defaults to 4.
        max_num_patches (int, optional): The maximum number of patches to be
            masked in the process of generating mask. Defaults to None.
        min_aspect (float): The minimum aspect ratio of mask blocks. Defaults
            to 0.3.
        min_aspect (float, optional): The minimum aspect ratio of mask blocks.
            Defaults to None.
    """

    def __init__(self,
                 input_size: int,
                 num_masking_patches: int,
                 min_num_patches: int = 4,
                 max_num_patches: Optional[int] = None,
                 min_aspect: float = 0.3,
                 max_aspect: Optional[float] = None) -> None:

        self.temporal, self.height, self.width = input_size
        self.num_masking_patches = num_masking_patches
        self.min_num_patches = min_num_patches
        self.max_num_patches = (
            num_masking_patches
            if max_num_patches is None else max_num_patches)
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def get_shape(self) -> Tuple[int, int, int]:
        """Get the shape of mask.

        Returns:
            Tuple[int, int, int]: The shape of mask.
        """
        return self.temporal, self.height, self.width

    def _mask(self, mask: np.ndarray, max_mask_patches: int) -> int:
        """Generate mask recursively.

        Args:
            mask (np.ndarray): The mask to be generated.
            max_mask_patches (int): The maximum number of patches to be masked.

        Returns:
            int: The number of patches masked.
        """
        delta = 0
        for _ in range(100):
            target_area = random.uniform(self.min_num_patches,
                                         self.max_num_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            t = random.randint(1, self.temporal)  # !
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)
                front = random.randint(0, self.temporal - t)

                num_masked = mask[front:front + t, top:top + h,
                                  left:left + w].sum()
                # Overlap
                if 0 < h * w * t - num_masked <= max_mask_patches:
                    for i in range(front, front + t):
                        for j in range(top, top + h):
                            for k in range(left, left + w):
                                if mask[i, j, k] == 0:
                                    mask[i, j, k] = 1
                                    delta += 1

                if delta > 0:
                    break
        return delta

    def transform(self, results: dict) -> dict:
        """Method to generate random block mask.

        Args:
            results (dict): Result dict from previous pipeline.

        Returns:
            dict: Result dict with added key ``mask``.
        """
        mask = np.zeros(shape=self.get_shape(), dtype=np.int)
        mask_count = 0
        while mask_count < self.num_masking_patches:
            max_mask_patches = self.num_masking_patches - mask_count
            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta

        results.update({'mask': mask})
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(temporal={self.temporal}, '
        repr_str += f'height={self.height}, '
        repr_str += f'width={self.width}, '
        repr_str += f'num_masking_patches={self.num_masking_patches}, '
        repr_str += f'min_num_patches={self.min_num_patches}, '
        repr_str += f'max_num_patches={self.max_num_patches}, '
        repr_str += f'log_aspect_ratio={self.log_aspect_ratio})'
        return repr_str
