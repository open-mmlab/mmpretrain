# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

from mmcls.models.necks import GeneralizedMeanPooling
from mmcls.registry import MODELS


class MultiAtrous(nn.Module):

    def __init__(self, in_channel, out_channel, dilation_rates):
        super().__init__()
        # hidden_channel 是 out_channel 的四分之一
        hidden_channel = int(out_channel / 4)
        self.dilated_conv_list = nn.ModuleList([
            nn.Conv2d(
                in_channel,
                hidden_channel,
                kernel_size=(3, 3),
                padding=rate,
                dilation=rate) for rate in dilation_rates
        ])
        self.gap_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channel, hidden_channel, kernel_size=(1, 1)),
            nn.ReLU())

        # 平滑
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

    def __init__(self, in_channel, out_channel, dilation_rates):
        super().__init__()
        # 多簇
        self.multi_atrous = MultiAtrous(in_channel, in_channel, dilation_rates)
        self.conv1x1_1 = nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1))
        self.conv1x1_2 = nn.Conv2d(
            out_channel, out_channel, kernel_size=(1, 1), bias=False)
        self.conv1x1_3 = nn.Conv2d(
            out_channel, out_channel, kernel_size=(1, 1))

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channel)
        self.softplus = nn.Softplus()

    def forward(self, x):
        # 多簇
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

    def __init__(self):
        super().__init__()

    def forward(self, local_feat, global_feat):
        global_feat_norm = torch.norm(global_feat, p=2, dim=1)
        # f_l * f_g 矩阵乘法
        projection = torch.bmm(
            global_feat.unsqueeze(1), torch.flatten(local_feat, start_dim=2))
        # f_l * f_g * f_g
        projection = torch.bmm(global_feat.unsqueeze(2),
                               projection).view(local_feat.size())
        # (f_l * f_g * f_g) / (f_g * f_g)
        projection = projection / (global_feat_norm * global_feat_norm).view(
            -1, 1, 1, 1)

        orthogonal_comp = local_feat - projection
        global_feat = global_feat.unsqueeze(-1).unsqueeze(-1)

        return torch.cat(
            [global_feat.expand(orthogonal_comp.size()), orthogonal_comp],
            dim=1)


@MODELS.register_module()
class DolgNet(BaseModule):

    def __init__(self,
                 local_dim,
                 global_dim,
                 hidden_dim,
                 output_dim,
                 dilation_rates,
                 init_cfg=None):
        super().__init__(init_cfg)
        # 正交融合
        self.orthogonal_fusion = OrthogonalFusion()
        # local 分支
        self.local_branch = LocalBranch(local_dim, hidden_dim, dilation_rates)
        # global 分支
        self.global_branch = nn.Sequential(GeneralizedMeanPooling(),
                                           nn.Flatten(),
                                           nn.Linear(global_dim, hidden_dim))

        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.fc = nn.Linear(2 * hidden_dim, output_dim)

    def forward(self, inputs):
        if not (isinstance(inputs, tuple) or isinstance(inputs, list)):
            raise TypeError('inputs of dolg should be tuple or list')
        assert len(inputs) >= 2, \
            f'the length of inputs should be greater than 2, ' \
            f'but got {len(inputs)}, {inputs[0].shape}'

        local_feat, global_feat = inputs[-2], inputs[-1]

        # 局部特征
        local_feat = self.local_branch(local_feat)
        global_feat = self.global_branch(global_feat)

        feat = self.orthogonal_fusion(local_feat, global_feat)
        feat = self.gap(feat).squeeze()
        # feat = self.fc(feat)
        return feat
