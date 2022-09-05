# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Optional

import torch
import torch.nn as nn
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmengine.model import BaseModule

from mmcls.registry import MODELS


@MODELS.register_module()
class LinearReduction(BaseModule):
    """Neck with Dimension reduction.

    Args:
        in_channels (int): Number of channels in the input.
        out_channels (int): Number of channels in the output.
        norm_cfg (dict, optional): dictionary to construct and
            config norm layer. Defaults to None.
        act_cfg (dict, optional): dictionary to construct and
            config activate layer. Defaults to None.
        init_cfg (dict, optional): dictionary to initialize weights.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 norm_cfg: Optional[dict] = None,
                 act_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):
        super(LinearReduction, self).__init__(init_cfg=init_cfg)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_cfg = copy.deepcopy(norm_cfg)
        self.act_cfg = copy.deepcopy(act_cfg)

        self.reduction = nn.Linear(
            in_features=in_channels, out_features=out_channels)
        if norm_cfg:
            self.norm = build_norm_layer(norm_cfg, out_channels)[1]
        else:
            self.norm = nn.Identity()
        if act_cfg:
            self.act = build_activation_layer(act_cfg)
        else:
            self.act = nn.Identity()

    def forward(self, inputs):
        if isinstance(inputs, tuple):
            inputs = inputs[-1]
        elif isinstance(inputs, torch.Tensor):
            pass
        else:
            raise TypeError(f'neck inputs must be tuple or torch.tensor,'
                            f' but get {type(inputs)}')

        out = self.reduction(inputs)
        return self.act(self.norm(out))
