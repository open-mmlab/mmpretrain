# Copyright (c) OpenMMLab. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, ModuleList, Sequential

from mmcls.models.builder import BACKBONES


def fuse_bn(conv_or_fc, bn):
    std = (bn.running_var + bn.eps).sqrt()
    t = bn.weight / std
    t = t.reshape(-1, 1, 1, 1)

    if len(t) == conv_or_fc.weight.size(0):
        return (conv_or_fc.weight * t,
                bn.bias - bn.running_mean * bn.weight / std)
    else:
        repeat_times = conv_or_fc.weight.size(0) // len(t)
        repeated = t.repeat_interleave(repeat_times, 0)
        return conv_or_fc.weight * repeated, (
            bn.bias - bn.running_mean * bn.weight / std).repeat_interleave(
                repeat_times, 0)


class GlobalPerceptron(nn.Module):

    def __init__(self, input_channels, internal_neurons):
        super(GlobalPerceptron, self).__init__()
        self.fc1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=internal_neurons,
            kernel_size=1,
            stride=1,
            bias=True)
        self.fc2 = nn.Conv2d(
            in_channels=internal_neurons,
            out_channels=input_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        x = F.sigmoid(x)
        x = x.view(-1, self.input_channels, 1, 1)
        return x


class RepMLPBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 h,
                 w,
                 reparam_conv_k=None,
                 globalperceptron_reduce=4,
                 num_sharesets=1,
                 deploy=False):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_sharesets = num_sharesets

        self.h, self.w = h, w

        self.deploy = deploy

        assert in_channels == out_channels
        self.gp = GlobalPerceptron(
            input_channels=in_channels,
            internal_neurons=in_channels // globalperceptron_reduce)

        self.fc3 = nn.Conv2d(
            self.h * self.w * num_sharesets,
            self.h * self.w * num_sharesets,
            1,
            1,
            0,
            bias=deploy,
            groups=num_sharesets)
        if deploy:
            self.fc3_bn = nn.Identity()
        else:
            self.fc3_bn = nn.BatchNorm2d(num_sharesets)

        self.reparam_conv_k = reparam_conv_k
        if not deploy and reparam_conv_k is not None:
            for k in reparam_conv_k:
                conv_branch = ConvModule(
                    in_channels=num_sharesets,
                    out_channels=num_sharesets,
                    kernel_size=k,
                    stride=1,
                    padding=k // 2,
                    norm_cfg=dict(type='BN'),
                    groups=num_sharesets,
                    act_cfg=None)
                self.__setattr__('repconv{}'.format(k), conv_branch)

    def partition(self, x, h_parts, w_parts):
        x = x.reshape(-1, self.in_channels, h_parts, self.h, w_parts, self.w)
        x = x.permute(0, 2, 4, 1, 3, 5)
        return x

    def partition_affine(self, x, h_parts, w_parts):
        fc_inputs = x.reshape(-1, self.num_sharesets * self.h * self.w, 1, 1)
        out = self.fc3(fc_inputs)
        out = out.reshape(-1, self.num_sharesets, self.h, self.w)
        out = self.fc3_bn(out)
        out = out.reshape(-1, h_parts, w_parts, self.num_sharesets, self.h,
                          self.w)
        return out

    def forward(self, inputs):
        #   Global Perceptron
        global_vec = self.gp(inputs)

        origin_shape = inputs.size()
        h_parts = origin_shape[2] // self.h
        w_parts = origin_shape[3] // self.w

        partitions = self.partition(inputs, h_parts, w_parts)

        #  Channel Perceptron
        fc3_out = self.partition_affine(partitions, h_parts, w_parts)

        #   Local Perceptron
        if self.reparam_conv_k is not None and not self.deploy:
            conv_inputs = partitions.reshape(-1, self.num_sharesets, self.h,
                                             self.w)
            conv_out = 0
            for k in self.reparam_conv_k:
                conv_branch = self.__getattr__('repconv{}'.format(k))
                conv_out += conv_branch(conv_inputs)
            conv_out = conv_out.reshape(-1, h_parts, w_parts,
                                        self.num_sharesets, self.h, self.w)
            fc3_out += conv_out

        fc3_out = fc3_out.permute(0, 3, 1, 4, 2,
                                  5)  # N, O, h_parts, out_h, w_parts, out_w
        out = fc3_out.reshape(*origin_shape)
        out = out * global_vec
        return out

    def get_equivalent_fc3(self):
        fc_weight, fc_bias = fuse_bn(self.fc3, self.fc3_bn)
        if self.reparam_conv_k is not None:
            largest_k = max(self.reparam_conv_k)
            largest_branch = self.__getattr__('repconv{}'.format(largest_k))
            total_kernel, total_bias = fuse_bn(largest_branch.conv,
                                               largest_branch.bn)
            for k in self.reparam_conv_k:
                if k != largest_k:
                    k_branch = self.__getattr__('repconv{}'.format(k))
                    kernel, bias = fuse_bn(k_branch.conv, k_branch.bn)
                    total_kernel += F.pad(kernel, [(largest_k - k) // 2] * 4)
                    total_bias += bias
            rep_weight, rep_bias = self._convert_conv_to_fc(
                total_kernel, total_bias)
            final_fc3_weight = rep_weight.reshape_as(fc_weight) + fc_weight
            final_fc3_bias = rep_bias + fc_bias
        else:
            final_fc3_weight = fc_weight
            final_fc3_bias = fc_bias
        return final_fc3_weight, final_fc3_bias

    def local_inject(self):
        self.deploy = True
        #   Locality Injection
        fc3_weight, fc3_bias = self.get_equivalent_fc3()
        #   Remove Local Perceptron
        if self.reparam_conv_k is not None:
            for k in self.reparam_conv_k:
                self.__delattr__('repconv{}'.format(k))
        self.__delattr__('fc3')
        self.__delattr__('fc3_bn')
        self.fc3 = nn.Conv2d(
            self.num_sharesets * self.h * self.w,
            self.num_sharesets * self.h * self.w,
            1,
            1,
            0,
            bias=True,
            groups=self.num_sharesets)
        self.fc3_bn = nn.Identity()
        self.fc3.weight.data = fc3_weight
        self.fc3.bias.data = fc3_bias

    def _convert_conv_to_fc(self, conv_kernel, conv_bias):
        in_channels = torch.eye(self.h * self.w).repeat(
            1, self.num_sharesets).reshape(self.h * self.w, self.num_sharesets,
                                           self.h,
                                           self.w).to(conv_kernel.device)
        fc_k = F.conv2d(
            in_channels,
            conv_kernel,
            padding=(conv_kernel.size(2) // 2, conv_kernel.size(3) // 2),
            groups=self.num_sharesets)
        fc_k = fc_k.reshape(self.h * self.w,
                            self.num_sharesets * self.h * self.w).t()
        fc_bias = conv_bias.repeat_interleave(self.h * self.w)
        return fc_k, fc_bias


class RepMLPNetUnit(BaseModule):

    def __init__(self,
                 channels,
                 h,
                 w,
                 reparam_conv_k,
                 globalperceptron_reduce,
                 ffn_expand=4,
                 num_sharesets=1,
                 deploy=False):
        super().__init__()
        self.repmlp_block = RepMLPBlock(
            in_channels=channels,
            out_channels=channels,
            h=h,
            w=w,
            reparam_conv_k=reparam_conv_k,
            globalperceptron_reduce=globalperceptron_reduce,
            num_sharesets=num_sharesets,
            deploy=deploy)
        self.ffn_block = FFNBlock(channels, channels * ffn_expand)
        self.prebn1 = nn.BatchNorm2d(channels)
        self.prebn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        y = x + self.repmlp_block(self.prebn1(x))
        z = y + self.ffn_block(self.prebn2(y))
        return z


class FFNBlock(nn.Module):
    """FFNBlock implemented by using point-wise convs."""

    def __init__(self,
                 in_channels,
                 hidden_channels=None,
                 out_channels=None,
                 act_layer=nn.GELU):
        super().__init__()
        out_features = out_channels or in_channels
        hidden_features = hidden_channels or in_channels
        self.ffn_fc1 = ConvModule(
            in_channels,
            hidden_features,
            1,
            1,
            0,
            norm_cfg=dict(type='BN'),
            act_cfg=None)
        self.ffn_fc2 = ConvModule(
            hidden_features,
            out_features,
            1,
            1,
            0,
            norm_cfg=dict(type='BN'),
            act_cfg=None)
        self.act = act_layer()

    def forward(self, x):
        x = self.ffn_fc1(x)
        x = self.act(x)
        x = self.ffn_fc2(x)
        return x


@BACKBONES.register_module(force=True)
class RepMLPNet(BaseModule):
    arch_zoo = {
        **dict.fromkeys(['b224'],
                        {'channels':       [96, 192, 384, 768],
                         'hs':             [56, 28, 14, 7],
                         'ws':             [56, 28, 14, 7],
                         'num_blocks':     [2, 2, 12, 2],
                         'sharesets_nums': [1, 4, 32, 128]}),
        **dict.fromkeys(['b256'],
                        {'channels':       [96, 192, 384, 768],
                         'hs':             [64, 32, 16, 8],
                         'ws':             [64, 32, 16, 8],
                         'num_blocks':     [2, 2, 12, 2],
                         'sharesets_nums': [1, 4, 32, 128]}),
    }  # yapf: disable

    essential_keys = {'channels', 'hs', 'ws', 'num_blocks', 'sharesets_nums'}

    def __init__(self,
                 arch,
                 in_channels=3,
                 out_indices=(3, ),
                 patch_size=(4, 4),
                 reparam_conv_k=(3, ),
                 globalperceptron_reduce=4,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 deploy=False,
                 init_cfg=None):
        super(RepMLPNet, self).__init__(init_cfg=init_cfg)
        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
        else:
            assert isinstance(arch, dict) and (
                set(arch) == self.essential_keys
            ), f'Custom arch needs a dict with keys {self.essential_keys}.'
            self.arch_settings = arch

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.num_extra_tokens = 0  # there is no cls-token in Twins
        self.num_stage = len(self.arch_settings['channels'])
        for key, value in self.arch_settings.items():
            assert isinstance(value, list) and len(value) == self.num_stage, (
                'Length of setting item in arch dict must be type of list and'
                ' have the same length.')

        self.channels = self.arch_settings['channels']
        self.hs = self.arch_settings['hs']
        self.ws = self.arch_settings['ws']
        self.num_blocks = self.arch_settings['num_blocks']
        self.sharesets_nums = self.arch_settings['sharesets_nums']

        self.path_embed = ConvModule(
            in_channels,
            self.channels[0],
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            inplace=True)

        self.stages, self.embeds = ModuleList(), ModuleList()
        for stage_idx in range(self.num_stage):
            stage_blocks = [
                RepMLPNetUnit(
                    channels=self.channels[stage_idx],
                    h=self.hs[stage_idx],
                    w=self.ws[stage_idx],
                    reparam_conv_k=reparam_conv_k,
                    globalperceptron_reduce=globalperceptron_reduce,
                    ffn_expand=4,
                    num_sharesets=self.sharesets_nums[stage_idx],
                    deploy=deploy) for _ in range(self.num_blocks[stage_idx])
            ]
            self.stages.append(Sequential(*stage_blocks))
            if stage_idx < self.num_stage - 1:
                self.embeds.append(
                    ConvModule(
                        in_channels=self.channels[stage_idx],
                        out_channels=self.channels[stage_idx + 1],
                        kernel_size=2,
                        stride=2,
                        padding=0,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        inplace=True))

        self.head_norm = nn.BatchNorm2d(self.channels[-1])
        self.out_indice = out_indices
        channels = tuple(list(self.channels) + [self.channels[-1]])
        for i in out_indices:
            norm_layer = nn.BatchNorm2d(channels[i + 1])
            self.add_module(f'norm{i}', norm_layer)

    def forward(self, x):
        outs = []

        x = self.path_embed(x)
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < len(self.stages) - 1:
                embed = self.embeds[i]
                x = embed(x)

            if i in self.out_indice:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(x)
                outs.append(out)
        return tuple(outs)

    def locality_injection(self):
        for m in self.modules():
            if hasattr(m, 'local_inject'):
                m.local_inject()


if __name__ == '__main__':
    # model settings
    repmlp_224_cfg = dict(
        type='ImageClassifier',
        backbone=dict(
            type='RepMLPNet',
            arch='B224',
            out_indices=(
                0,
                1,
                2,
                3,
            ),
            reparam_conv_k=(1, 3),
            deploy=False),
        neck=dict(type='GlobalAveragePooling'),
        head=dict(
            type='LinearClsHead',
            num_classes=1000,
            in_channels=768,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
            topk=(1, 5),
        ))

    from mmcls.models import CLASSIFIERS
    model = CLASSIFIERS.build(repmlp_224_cfg)
    x = torch.rand((1, 3, 224, 224))
    y = model(x, return_loss=False)
    print(y[0].shape)
    outs = model.extract_feat(x, stage='backbone')
    for i, out in enumerate(outs):
        print(i, out.shape)
