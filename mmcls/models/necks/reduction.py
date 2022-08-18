# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmcls.registry import MODELS


@MODELS.register_module()
class Reduction(nn.Module):
    """Neck with Dimension reduction.

    Args:
        in_channels (int): Number of channels in the input.
        out_channels (int): Number of channels in the output.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(Reduction, self).__init__()

        self.reduction = nn.Linear(
            in_features=in_channels, out_features=out_channels)
        self.act = nn.SiLU(inplace=True)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, inputs):
        if isinstance(inputs, tuple):
            inputs = inputs[-1]
            out = self.reduction(inputs)
            out = self.act(out)
            out = self.bn(out)
        elif isinstance(inputs, torch.Tensor):
            out = self.reduction(inputs)
            out = self.act(out)
            out = self.bn(out)
        else:
            raise TypeError('neck inputs '
                            'should be torch.tensor or tuple of tensors')
        return out
