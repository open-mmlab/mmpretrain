# Copyright (c) OpenMMLab. All rights reserved.
from math import cos, pi
from typing import Optional

import torch
import torch.nn as nn
from mmengine.logging import MessageHub
from mmengine.model import ExponentialMovingAverage

from mmpretrain.registry import MODELS


@MODELS.register_module()
class CosineEMA(ExponentialMovingAverage):
    """CosineEMA is implemented for updating momentum parameter, used in BYOL,
    MoCoV3, etc.

    The momentum parameter is updated with cosine annealing, including momentum
    adjustment following:

    .. math::
        m = m_1 - (m_1 - m_0) * (cos(pi * k / K) + 1) / 2

    where :math:`k` is the current step, :math:`K` is the total steps.

    Args:
        model (nn.Module): The model to be averaged.
        momentum (float): The momentum used for updating ema parameter.
            Ema's parameter are updated with the formula:
            `averaged_param = momentum * averaged_param + (1-momentum) *
            source_param`. Defaults to 0.996.
        end_momentum (float): The end momentum value for cosine annealing.
            Defaults to 1.
        interval (int): Interval between two updates. Defaults to 1.
        device (torch.device, optional): If provided, the averaged model will
            be stored on the :attr:`device`. Defaults to None.
        update_buffers (bool): if True, it will compute running averages for
            both the parameters and the buffers of the model. Defaults to
            False.
    """

    def __init__(self,
                 model: nn.Module,
                 momentum: float = 0.996,
                 end_momentum: float = 1.,
                 interval: int = 1,
                 device: Optional[torch.device] = None,
                 update_buffers: bool = False) -> None:
        super().__init__(
            model=model,
            momentum=momentum,
            interval=interval,
            device=device,
            update_buffers=update_buffers)
        self.end_momentum = end_momentum

    def avg_func(self, averaged_param: torch.Tensor,
                 source_param: torch.Tensor, steps: int) -> None:
        """Compute the moving average of the parameters using the cosine
        momentum strategy.

        Args:
            averaged_param (Tensor): The averaged parameters.
            source_param (Tensor): The source parameters.
            steps (int): The number of times the parameters have been
                updated.
        Returns:
            Tensor: The averaged parameters.
        """
        message_hub = MessageHub.get_current_instance()
        max_iters = message_hub.get_info('max_iters')
        momentum = self.end_momentum - (self.end_momentum - self.momentum) * (
            cos(pi * steps / float(max_iters)) + 1) / 2
        averaged_param.mul_(momentum).add_(source_param, alpha=(1 - momentum))
