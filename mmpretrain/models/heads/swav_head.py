# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.model import BaseModule

from mmpretrain.registry import MODELS


@MODELS.register_module()
class SwAVHead(BaseModule):
    """Head for SwAV Pre-training.

    Args:
        loss (dict): Config dict for module of loss functions.
    """

    def __init__(self, loss: dict) -> None:
        super().__init__()
        self.loss_module = MODELS.build(loss)

    def loss(self, pred: torch.Tensor) -> torch.Tensor:
        """Generate loss.

        Args:
            pred (torch.Tensor): NxC input features.

        Returns:
            torch.Tensor: The SwAV loss.
        """
        loss = self.loss_module(pred)

        return loss
