# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List

import torch
import torch.nn.functional as F

from mmpretrain.models import BaseSelfSupervisor
from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample


@MODELS.register_module()
class VideoMaskFeat(BaseSelfSupervisor):
    """MaskFeat.

    Implementation of `Masked Feature Prediction for Self-Supervised Visual
    Pre-Training <https://arxiv.org/abs/2112.09133>`_.
    """

    def loss(self, inputs: List[torch.Tensor], data_samples: List[DataSample],
             **kwargs) -> Dict[str, torch.Tensor]:
        """The forward function in training.

        Args:
            inputs (List[torch.Tensor]): The input images.
            data_samples (List[DataSample]): All elements required
                during the forward function.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of loss components.
        """
        mask = torch.stack(
            [data_sample.mask.value for data_sample in data_samples])
        mask = mask.to(torch.bool)

        video = inputs[0]
        video = video.view((-1, ) + video.shape[2:])  # B, C, T, H, W
        latent = self.backbone(video, mask)
        B, L, C = latent[0].shape
        pred = self.neck([latent[0].view(B * L, C)])
        pred = pred[0].view(B, L, -1)

        # generate hog target
        video = video[:, :, ::self.backbone.patch_stride[0], :, :]
        video = video.transpose(1, 2)  # B, T, C, H, W
        self.target_generator.B = video.size(0)
        self.target_generator.T = video.size(1)
        video = video.flatten(0, 1)  # B*T, C, H, W
        hog = self.target_generator(video)

        mask = self._get_output_mask(mask)
        loss = self.head(pred, hog, mask)
        losses = dict(loss=loss)
        return losses

    def _get_output_mask(self, mask: torch.Tensor) -> torch.Tensor:
        size = self.backbone.out_patch_resolution[-1][-1]
        output_mask = F.interpolate(mask.float(), size=size)
        return output_mask
