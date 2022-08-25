# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch
import torch.nn as nn
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmengine.model.utils import initialize

from mmcls.registry import MODELS


@MODELS.register_module()
class LinearReduction(nn.Module):
    """Neck with Dimension reduction.

    Args:
        in_channels (int): Number of channels in the input.
        out_channels (int): Number of channels in the output.
        act_cfg (dict, optional): dictionary to construct and
            config activate layer. Default: dict(type='ReLU')
        norm_cfg (dict, optional): dictionary to construct and
            config norm layer. Default: dict(type='BN1d')
        init_cfg (dict, optional): dictionary to initialize weights.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 act_cfg=dict(type='ReLU'),
                 norm_cfg=dict(type='BN1d'),
                 init_cfg=None):
        super(LinearReduction, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act_cfg = copy.deepcopy(act_cfg)
        self.norm_cfg = copy.deepcopy(norm_cfg)
        self.init_cfg = copy.deepcopy(init_cfg)

        self.reduction = nn.Linear(
            in_features=in_channels, out_features=out_channels)
        if act_cfg:
            self.act = build_activation_layer(act_cfg)
        else:
            self.act = None

        if norm_cfg:
            self.norm_name, layer = build_norm_layer(norm_cfg, out_channels)
            self.add_module(self.norm_name, layer)
        else:
            self.norm_name = None

    @property
    def norm(self):
        if self.norm_name:
            return getattr(self, self.norm_name)
        else:
            return None

    def init_weights(self):
        if self.init_cfg:
            initialize(self, self.init_cfg)

    def forward(self, inputs):
        if isinstance(inputs, tuple):
            inputs = inputs[-1]
        elif isinstance(inputs, torch.Tensor):
            pass
        else:
            raise TypeError(f'neck inputs must be tuple or torch.tensor,'
                            f' but get {type(inputs)}')

        out = self.reduction(inputs)
        if self.act_cfg:
            out = self.act(out)
        if self.norm_cfg:
            out = self.norm(out)
        return out
