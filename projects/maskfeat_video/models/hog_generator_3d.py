# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmpretrain.models import HOGGenerator
from mmpretrain.registry import MODELS


@MODELS.register_module()
class HOGGenerator3d(HOGGenerator):
    """Generate HOG feature for videos.

    This module is used in MaskFeat to generate HOG feature.
    Here is the link of `HOG wikipedia
    <https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients>`_.

    Args:
        nbins (int): Number of bin. Defaults to 9.
        pool (float): Number of cell. Defaults to 8.
        gaussian_window (int): Size of gaussian kernel. Defaults to 16.
    """

    def __init__(self,
                 nbins: int = 9,
                 pool: int = 8,
                 gaussian_window: int = 16) -> None:
        super().__init__(
            nbins=nbins, pool=pool, gaussian_window=gaussian_window)

    def _reshape(self, hog_feat: torch.Tensor) -> torch.Tensor:
        """Reshape HOG Features for output."""
        hog_feat = hog_feat.flatten(1, 2)
        self.unfold_size = hog_feat.shape[-1] // 14
        hog_feat = hog_feat.permute(0, 2, 3, 1)
        hog_feat = hog_feat.unfold(1, self.unfold_size,
                                   self.unfold_size).unfold(
                                       2, self.unfold_size, self.unfold_size)
        hog_feat = hog_feat.flatten(3).view(self.B, self.T, 14, 14, -1)
        hog_feat = hog_feat.flatten(1, 3)  # B N C
        return hog_feat
