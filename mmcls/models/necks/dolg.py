# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmengine.utils import is_seq_of

from mmcls.registry import MODELS
from .gem import GeneralizedMeanPooling


class MultiAtrous(BaseModule):
    """Multiple atrous convolution layers.

    Args:
        in_channel (int): Number of channels in the input.
        out_channel (int): Number of channels in the output.
        dilation_rates (Sequence[int]): The list of the dilation rates
            of multiple atrous convolution. Defaults to `[3, 6, 9]`.
        init_cfg (dict, optional): dictionary to initialize weights.
            Defaults to None.
    """

    def __init__(self,
                 in_channel: int,
                 out_channel: int,
                 dilation_rates: Sequence[int] = [3, 6, 9],
                 init_cfg: Optional[dict] = None):
        super(MultiAtrous, self).__init__(init_cfg)
        assert is_seq_of(
            dilation_rates,
            int), ('``dilation_rates`` must be a sequence of int.')
        assert out_channel % 2 == 0

        # In the original implementation, here hidden-dim was set to a
        # fixed 512 as it was designed for resnet101. We set it here to
        # half of out_c for the sake of generality, which is consistent
        #  when using resnet101.
        hidden_channel = out_channel // 2
        self.dilated_conv_list = nn.ModuleList([
            nn.Conv2d(
                in_channel,
                hidden_channel,
                kernel_size=(3, 3),
                padding=rate,
                dilation=rate) for rate in dilation_rates
        ])

        # global convolution
        self.gap_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channel, hidden_channel, kernel_size=(1, 1)),
            nn.ReLU())

        # convolution after concat for smoothing
        num = len(dilation_rates)
        self.conv_after = nn.Sequential(
            nn.Conv2d(hidden_channel * num, out_channel, kernel_size=(1, 1)),
            nn.ReLU())

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        local_feat_list = [
            F.interpolate(
                self.gap_conv(x),
                scale_factor=(h, w),
                mode='bilinear',
                align_corners=False)
        ]
        for dilated_conv in self.dilated_conv_list:
            local_feat_list.append(dilated_conv(x))
        local_feat = torch.cat(local_feat_list, dim=1)
        local_feat = self.conv_after(local_feat)
        return local_feat


class ATT(BaseModule):
    """self-ATT module in the DOLG local branch.

    Args:
        in_channel (int): Number of channels in the input.
        out_channel (int): Number of channels in the output.
        init_cfg (dict, optional): dictionary to initialize weights.
            Defaults to None.
    """

    def __init__(self,
                 in_channel: int,
                 out_channel: int,
                 init_cfg: Optional[dict] = None):
        super(ATT, self).__init__(init_cfg)
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1))
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=(1, 1))

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channel)
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)

        feature_map_norm = F.normalize(x, p=2, dim=1)

        attention_map = self.relu(x)
        attention_map = self.conv2(attention_map)
        attention_map = self.softplus(attention_map)

        return attention_map * feature_map_norm


class LocalBranch(BaseModule):
    """The local branch in DOLG, which consists of a ``MultiAtrous`` module and
    a ``ATT`` module.

    Args:
        in_channel (int): Number of channels in the input.
        out_channel (int): Number of channels in the output.
        dilation_rates (Sequence[int]): The list of the dilation rates
            of multiple atrous convolution. Defaults to `[3, 6, 9]`.
        init_cfg (dict, optional): dictionary to initialize weights.
            Defaults to None.
    """

    def __init__(self,
                 in_channel: int,
                 out_channel: int,
                 dilation_rates: Sequence[int] = [3, 6, 9],
                 init_cfg: Optional[dict] = None):
        super(LocalBranch, self).__init__(init_cfg)
        self.multi_atrous = MultiAtrous(in_channel, in_channel, dilation_rates,
                                        init_cfg)
        self.att = ATT(in_channel, out_channel, init_cfg)

    def forward(self, x):
        x = self.multi_atrous(x)
        x = self.att(x)
        return x


class OrthogonalFusion(nn.Module):
    """Orthogonal Fusion Module."""

    def __init__(self):
        super().__init__()

    def forward(self, local_feat, global_feat):
        global_feat_norm = torch.norm(global_feat, p=2, dim=1)

        proj = torch.bmm(
            global_feat.unsqueeze(1), torch.flatten(local_feat, start_dim=2))

        proj = torch.bmm(global_feat.unsqueeze(2),
                         proj).view(local_feat.size())

        g_norm = global_feat_norm * global_feat_norm
        proj = proj / g_norm.view(-1, 1, 1, 1)
        # Orthogonal component
        orthogonal_comp = local_feat - proj
        global_feat = global_feat.unsqueeze(-1).unsqueeze(-1)
        # concat the orthogonal components and global features
        fused_feat = torch.cat(
            [global_feat.expand(orthogonal_comp.size()), orthogonal_comp],
            dim=1)

        return fused_feat


@MODELS.register_module()
class DOLG(BaseModule):
    """Deep Orthogonal Local and Global.

    Args:
        local_dim (int): Dimension of local features
        global_dim (int): Dimension of global features
        hidden_dim (int): Dimension of the joint mapping of local
            and global features
        dilation_rates (Union[list, tuple]): The list of the dilation
            rates of multiple atrous convolution in the local branch.
        init_cfg (dict, optional): dictionary to initialize weights.
            Defaults to None.
    """

    def __init__(self,
                 local_dim: int,
                 global_dim: int,
                 hidden_dim: int,
                 dilation_rates: Union[list, tuple],
                 init_cfg: Optional[dict] = None):
        super(DOLG, self).__init__(init_cfg)
        self.local_dim = local_dim
        self.global_dim = global_dim
        self.hidden_dim = hidden_dim

        self.local_branch = LocalBranch(local_dim, hidden_dim, dilation_rates)
        self.global_branch = nn.Sequential(GeneralizedMeanPooling(),
                                           nn.Flatten(),
                                           nn.Linear(global_dim, hidden_dim))

        self.orthogonal_fusion = OrthogonalFusion()
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, inputs):
        if not (isinstance(inputs, tuple) or isinstance(inputs, list)):
            raise TypeError('inputs of dolg should be tuple or list')
        assert len(inputs) >= 2, \
            f'the length of inputs should be greater than 2, ' \
            f'but got {len(inputs)}.'

        *_, local_feat, global_feat = inputs
        local_feat = self.local_branch(local_feat)
        global_feat = self.global_branch(global_feat)

        fused_feat = self.orthogonal_fusion(local_feat, global_feat)
        fused_feat = self.gap(fused_feat).flatten(1)
        return fused_feat
