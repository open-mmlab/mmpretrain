# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

from mmcls.models.necks import GeneralizedMeanPooling
from mmcls.registry import MODELS


class MultiAtrous(nn.Module):
    """Multiple atrous convolution layers.

    Args:
        in_channel (int): Number of channels in the input.
        out_channel (int): Number of channels in the output.
        dilation_rates (Union[list, tuple]): The list of
            the dilation rates of multiple atrous convolution.
    """

    def __init__(self, in_channel: int, out_channel: int,
                 dilation_rates: Union[list, tuple]):
        super().__init__()

        # multiple atrous convolution
        hidden_channel = out_channel // 4
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
        num = len(dilation_rates) + 1
        self.conv_after = nn.Sequential(
            nn.Conv2d(hidden_channel * num, out_channel, kernel_size=(1, 1)),
            nn.ReLU())

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        local_feat = [
            F.interpolate(
                self.gap_conv(x),
                scale_factor=(h, w),
                mode='bilinear',
                align_corners=False)
        ]
        for dilated_conv in self.dilated_conv_list:
            local_feat.append(dilated_conv(x))
        local_feat = torch.cat(local_feat, dim=1)
        local_feat = self.conv_after(local_feat)
        return local_feat


class LocalBranch(nn.Module):
    """The local branch in DOLG, which consists of multiple atrous convolution
    layers and a self-attention module.

    Args:
        in_channel (int): Number of channels in the input.
        out_channel (int): Number of channels in the output.
        dilation_rates (Union[list, tuple]): The list of
            the dilation rates of multiple atrous convolution.
    """

    def __init__(self, in_channel: int, out_channel: int,
                 dilation_rates: Union[list, tuple]):
        super().__init__()
        self.multi_atrous = MultiAtrous(in_channel, in_channel, dilation_rates)

        # self-attention module
        self.conv1x1_1 = nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1))
        self.conv1x1_2 = nn.Conv2d(
            out_channel, out_channel, kernel_size=(1, 1), bias=False)
        self.conv1x1_3 = nn.Conv2d(
            out_channel, out_channel, kernel_size=(1, 1))

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channel)
        self.softplus = nn.Softplus()

    def forward(self, x):
        local_feat = self.multi_atrous(x)

        local_feat = self.conv1x1_1(local_feat)
        local_feat = self.relu(local_feat)
        local_feat = self.conv1x1_2(local_feat)
        local_feat = self.bn(local_feat)

        attention_map = self.relu(local_feat)
        attention_map = self.conv1x1_3(attention_map)
        attention_map = self.softplus(attention_map)

        local_feat = F.normalize(local_feat, p=2, dim=1)
        local_feat = local_feat * attention_map

        return local_feat


class OrthogonalFusion(nn.Module):
    """Orthogonal Fusion Module."""

    def __init__(self):
        super().__init__()

    def forward(self, local_feat, global_feat):
        global_feat_norm = torch.norm(global_feat, p=2, dim=1)

        projection = torch.bmm(
            global_feat.unsqueeze(1), torch.flatten(local_feat, start_dim=2))

        projection = torch.bmm(global_feat.unsqueeze(2),
                               projection).view(local_feat.size())

        g_norm = global_feat_norm * global_feat_norm
        projection = projection / g_norm.view(-1, 1, 1, 1)
        # Orthogonal component
        orthogonal_comp = local_feat - projection
        global_feat = global_feat.unsqueeze(-1).unsqueeze(-1)
        # concat the orthogonal components and global features
        return torch.cat(
            [global_feat.expand(orthogonal_comp.size()), orthogonal_comp],
            dim=1)


@MODELS.register_module()
class DolgNet(BaseModule):
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
        super(DolgNet, self).__init__(init_cfg)
        # local branch
        self.local_branch = LocalBranch(local_dim, hidden_dim, dilation_rates)
        # global branch
        self.global_branch = nn.Sequential(GeneralizedMeanPooling(),
                                           nn.Flatten(),
                                           nn.Linear(global_dim, hidden_dim))
        # orthogonal fusion module
        self.orthogonal_fusion = OrthogonalFusion()
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, inputs):
        if not (isinstance(inputs, tuple) or isinstance(inputs, list)):
            raise TypeError('inputs of dolg should be tuple or list')
        assert len(inputs) >= 2, \
            f'the length of inputs should be greater than 2, ' \
            f'but got {len(inputs)}, {inputs[0].shape}'

        local_feat, global_feat = inputs[-2], inputs[-1]
        local_feat = self.local_branch(local_feat)
        global_feat = self.global_branch(global_feat)
        feat = self.orthogonal_fusion(local_feat, global_feat)
        feat = self.gap(feat).flatten(1)
        return feat
