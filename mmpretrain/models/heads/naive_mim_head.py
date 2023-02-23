# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch

from mmpretrain.registry import MODELS
from .base_head import BaseHead


@MODELS.register_module()
class NaiveMIMHead(BaseHead):
    """Naive pretrain head for Masked Image Modeling

    Args:
        loss (dict): Config of loss.
    """

    def __init__(self, loss: dict) -> None:
        super().__init__()
        self.loss_module = MODELS.build(loss)

    def loss(self,
             pred: torch.Tensor,
             target: torch.Tensor,
             mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward function.

        Args:
            pred (torch.Tensor): Predicted features, of shape (N, L, D).
            target (torch.Tensor): Target features, of shape (N, L, D).
            mask (torch.Tensor): The mask of the target image of shape.

        Returns:
            torch.Tensor: the reconstructed loss.
        """
        loss = self.loss_module(pred, target, mask)
        return loss
