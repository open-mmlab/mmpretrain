# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple, Optional, Union, List

import torch
import torch.nn as nn
from mmengine.dist import all_reduce, get_world_size

from mmpretrain.registry import MODELS
from .base_head import BaseHead


@MODELS.register_module()
class LatentPredictHead(BaseHead):
    """Head for latent feature prediction.

    This head builds a predictor, which can be any registered neck component.
    For example, BYOL and SimSiam call this head and build NonLinearNeck.
    It also implements similarity loss between two forward features.

    Args:
        loss (dict): Config dict for the loss.
        predictor (dict): Config dict for the predictor.
        init_cfg (dict or List[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 loss: dict,
                 predictor: dict,
                 init_cfg: Optional[Union[dict, List[dict]]] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.loss = MODELS.build(loss)
        self.predictor = MODELS.build(predictor)

    def forward(self, input: torch.Tensor,
                target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward head.

        Args:
            input (torch.Tensor): NxC input features.
            target (torch.Tensor): NxC target features.

        Returns:
            torch.Tensor: The latent predict loss.
        """
        pred = self.predictor([input])[0]
        target = target.detach()

        loss = self.loss(pred, target)

        return loss


@MODELS.register_module()
class LatentCrossCorrelationHead(BaseHead):
    """Head for latent feature cross correlation.

    Part of the code is borrowed from `script
    <https://github.com/facebookresearch/barlowtwins/blob/main/main.py>`_.

    Args:
        in_channels (int): Number of input channels.
        loss (dict): Config dict for module of loss functions.
        init_cfg (dict or List[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 loss: dict,
                 init_cfg: Optional[Union[dict, List[dict]]] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.world_size = get_world_size()
        self.bn = nn.BatchNorm1d(in_channels, affine=False)
        self.loss = MODELS.build(loss)

    def forward(self, input: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """Forward head.

        Args:
            input (torch.Tensor): NxC input features.
            target (torch.Tensor): NxC target features.

        Returns:
            torch.Tensor: The cross correlation loss.
        """
        # cross-correlation matrix
        cross_correlation_matrix = self.bn(input).T @ self.bn(target)
        cross_correlation_matrix.div_(input.size(0) * self.world_size)

        all_reduce(cross_correlation_matrix)

        loss = self.loss(cross_correlation_matrix)
        return loss
