# Copyright (c) OpenMMLab. All rights reserved.
import itertools
from typing import Sequence

import torch
import torch.nn as nn
from mmcv.cnn import Linear, build_norm_layer, fuse_conv_bn
from mmengine.model import BaseModule

from mmcls.models.backbones.base_backbone import BaseBackbone
from mmcls.registry import MODELS


class HybridBackbone(BaseModule):

    def __init__(
            self,
            embed_dim,
            kernel_size=3,
            stride=2,
            pad=1,
            dilation=1,
            groups=1,
            bn_weight_init=1,
            activation=nn.Hardswish,
            conv_cfg=None,
            norm_cfg=dict(type='BN'),
            init_cfg=None,
    ):
        super(HybridBackbone, self).__init__(init_cfg=init_cfg)

        self.input_channels = [
            3, embed_dim // 8, embed_dim // 4, embed_dim // 2
        ]
        self.output_channels = [
            embed_dim // 8, embed_dim // 4, embed_dim // 2, embed_dim
        ]
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.patch_embed = nn.Sequential()

        for i in range(len(self.input_channels)):
            conv_bn = ConvolutionBatchNorm(
                self.input_channels[i],
                self.output_channels[i],
                kernel_size=kernel_size,
                stride=stride,
                pad=pad,
                dilation=dilation,
                groups=groups,
                bn_weight_init=bn_weight_init,
                norm_cfg=norm_cfg,
            )
            self.patch_embed.add_module('%d' % (2 * i), conv_bn)
            if i < len(self.input_channels) - 1:
                self.patch_embed.add_module('%d' % (i * 2 + 1), activation())

    def forward(self, x):
        x = self.patch_embed(x)
        return x


class ConvolutionBatchNorm(nn.Sequential):

    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size=3,
            stride=2,
            pad=1,
            dilation=1,
            groups=1,
            bn_weight_init=1,
            norm_cfg=dict(type='BN'),
    ):
        super(ConvolutionBatchNorm, self).__init__()
        self.norm_cfg = norm_cfg
        self.input_channel = in_channel
        self.output_channel = out_channel
        _, bn = build_norm_layer(norm_cfg, out_channel)
        self.bn_weight_init = bn_weight_init
        self.conv = nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=pad,
            dilation=dilation,
            groups=groups,
            bias=False)
        self.bn = bn

    @torch.no_grad()
    def fuse(self):
        return fuse_conv_bn(self).conv


class LinearBatchNorm(nn.Sequential):

    def __init__(self,
                 in_feature,
                 out_feature,
                 bn_weight_init=1,
                 norm_cfg=dict(type='BN1d')):
        super(LinearBatchNorm, self).__init__()
        linear = Linear(in_feature, out_feature, bias=False)
        _, bn = build_norm_layer(norm_cfg, out_feature)
        self.bn_weight_init = bn_weight_init
        self.linear = linear
        self.bn = bn

    @torch.no_grad()
    def fuse(self):
        device = next(self.linear.parameters()).device
        w = self.bn.weight / (self.bn.running_var + self.bn.eps)**0.5
        w = self.linear.weight * w[:, None]
        b = self.bn.bias - self.bn.running_mean * self.bn.weight / \
            (self.bn.running_var + self.bn.eps) ** 0.5
        m = Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        m.to(device)
        return m

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x.flatten(0, 1)).reshape_as(x)
        return x


class Residual(BaseModule):

    def __init__(self, block, drop):
        super(Residual, self).__init__()
        self.block = block
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.block(x) * torch.rand(
                x.size(0), 1, 1, device=x.device).ge_(
                    self.drop).div(1 - self.drop).detach()
        else:
            return x + self.block(x)  # add


class Attention(BaseModule):

    def __init__(
        self,
        dim,
        key_dim,
        num_heads=8,
        attn_ratio=4,
        activation=None,
        resolution=14,
    ):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.scale = key_dim**-0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2
        self.qkv = LinearBatchNorm(dim, h)
        self.proj = nn.Sequential(
            activation(), LinearBatchNorm(self.dh, dim, bn_weight_init=0))

        points = list(itertools.product(range(resolution), range(resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N, N))

    @torch.no_grad()
    def train(self, mode=True):
        """change the mode of model."""
        super(Attention, self).train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):  # x (B,N,C)
        B, N, C = x.shape  # 2 196 128
        qkv = self.qkv(x)  # 2 196 128
        q, k, v = qkv.view(B, N, self.num_heads, -1).split(
            [self.key_dim, self.key_dim, self.d],
            dim=3)  # q 2 196 4 16 ; k 2 196 4 16; v 2 196 4 32
        q = q.permute(0, 2, 1, 3)  # 2 4 196 16
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = ((q @ k.transpose(-2, -1)) *
                self.scale  # 2 4 196 16 * 2 4 16 196 -> 2 4 196 196
                + (self.attention_biases[:, self.attention_bias_idxs]
                   if self.training else self.ab))
        attn = attn.softmax(dim=-1)  # 2 4 196 196 -> 2 4 196 196
        x = (attn @ v).transpose(1, 2).reshape(
            B, N,
            self.dh)  # 2 4 196 196 * 2 4 196 32 -> 2 4 196 32 -> 2 196 128
        x = self.proj(x)
        return x


class MLP(nn.Sequential):

    def __init__(
        self,
        embed_dim,
        mlp_ratio,
        mlp_activation,
    ):
        super(MLP, self).__init__()
        h = embed_dim * mlp_ratio
        self.linear1 = LinearBatchNorm(embed_dim, h)
        self.activation = mlp_activation()
        self.linear2 = LinearBatchNorm(h, embed_dim, bn_weight_init=0)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


class Subsample(BaseModule):

    def __init__(self, stride, resolution):
        super(Subsample, self).__init__()
        self.stride = stride
        self.resolution = resolution

    def forward(self, x):
        B, _, C = x.shape
        # B, N, C -> B, H, W, C
        x = x.view(B, self.resolution, self.resolution, C)
        x = x[:, ::self.stride, ::self.stride]
        x = x.reshape(B, -1, C)  # B, H', W', C -> B, N', C
        return x


class AttentionSubsample(nn.Sequential):

    def __init__(self,
                 in_dim,
                 out_dim,
                 key_dim,
                 num_heads=8,
                 attn_ratio=2,
                 activation=None,
                 stride=2,
                 resolution=14,
                 sub_resolution=7):
        super(AttentionSubsample, self).__init__()
        self.num_heads = num_heads
        self.scale = key_dim**-0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * self.num_heads
        self.attn_ratio = attn_ratio
        self.sub_resolution = sub_resolution
        h = self.dh + nh_kd
        self.kv = LinearBatchNorm(in_dim, h)

        self.q = nn.Sequential(
            Subsample(stride, resolution), LinearBatchNorm(in_dim, nh_kd))
        self.proj = nn.Sequential(activation(),
                                  LinearBatchNorm(self.dh, out_dim))

        self.stride = stride
        self.resolution = resolution
        points = list(itertools.product(range(resolution), range(resolution)))
        sub_points = list(
            itertools.product(range(sub_resolution), range(sub_resolution)))
        N = len(points)
        N_sub = len(sub_points)
        attention_offsets = {}
        idxs = []
        for p1 in sub_points:
            for p2 in points:
                size = 1
                offset = (abs(p1[0] * stride - p2[0] + (size - 1) / 2),
                          abs(p1[1] * stride - p2[1] + (size - 1) / 2))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N_sub, N))

    @torch.no_grad()
    def train(self, mode=True):
        super(AttentionSubsample, self).train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):
        B, N, C = x.shape
        k, v = self.kv(x).view(B, N, self.num_heads,
                               -1).split([self.key_dim, self.d], dim=3)
        k = k.permute(0, 2, 1, 3)  # BHNC
        v = v.permute(0, 2, 1, 3)  # BHNC
        q = self.q(x).view(B, self.sub_resolution**2, self.num_heads,
                           self.key_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale + \
               (self.attention_biases[:, self.attention_bias_idxs]
                if self.training else self.ab)
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, -1, self.dh)
        x = self.proj(x)
        return x


@MODELS.register_module()
class LeViT(BaseBackbone):
    """Vision Transformer with support for patch or hybrid CNN input stage."""
    arch_zoo = {
        '128s': {
            'embed_dim': [128, 256, 384],
            'num_heads': [4, 6, 8],
            'depth': [2, 3, 4],
            'key_dim': [16, 16, 16],
            'down_ops': [[16, 4, 2, 2], [16, 4, 2, 2]]
        },
        '128': {
            'embed_dim': [128, 256, 384],
            'num_heads': [4, 8, 12],
            'depth': [4, 4, 4],
            'key_dim': [16, 16, 16],
            'down_ops': [[16, 4, 2, 2], [16, 4, 2, 2]]
        },
        '192': {
            'embed_dim': [192, 288, 384],
            'num_heads': [3, 5, 6],
            'depth': [4, 4, 4],
            'key_dim': [32, 32, 32],
            'down_ops': [[32, 4, 2, 2], [32, 4, 2, 2]]
        },
        '256': {
            'embed_dim': [256, 384, 512],
            'num_heads': [4, 6, 8],
            'depth': [4, 4, 4],
            'key_dim': [32, 32, 32],
            'down_ops': [[32, 4, 2, 2], [32, 4, 2, 2]]
        },
        '384': {
            'embed_dim': [384, 512, 768],
            'num_heads': [6, 9, 12],
            'depth': [4, 4, 4],
            'key_dim': [32, 32, 32],
            'down_ops': [
                [32, 4, 2, 2],
                [32, 4, 2, 2],
            ]
        },
    }

    def __init__(self,
                 arch,
                 img_size=224,
                 patch_size=16,
                 attn_ratio=2,
                 mlp_ratio=2,
                 hybrid_backbone=None,
                 attention_activation=torch.nn.Hardswish,
                 mlp_activation=torch.nn.Hardswish,
                 out_indices=(2, ),
                 deploy=False,
                 drop_path_rate=0):
        super(LeViT, self).__init__()

        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
        elif isinstance(arch, dict):
            essential_keys = {
                'embed_dim', 'num_heads', 'depth', 'key_dim', 'down_ops'
            }
            assert isinstance(arch, dict) and set(arch) == essential_keys, \
                f'Custom arch needs a dict with keys {essential_keys}'
            self.arch_settings = arch
        else:
            raise TypeError('Expect "arch" to be either a string '
                            f'or a dict, got {type(arch)}')

        self.embed_dims = self.arch_settings['embed_dim']
        self.num_heads = self.arch_settings['num_heads']
        self.depth = self.arch_settings['depth']
        self.key_dim = self.arch_settings['key_dim']
        self.down_ops = self.arch_settings['down_ops']
        self.drop_path_rate = drop_path_rate

        self.num_features = self.embed_dims[-1]
        if not hybrid_backbone:
            hybrid_backbone = HybridBackbone(self.embed_dims[0])
        self.patch_embed = hybrid_backbone
        self.blocks = [[]]
        self.size = []
        self.deploy = deploy

        self.down_ops.append([])
        resolution = img_size // patch_size
        for i, (embed_dim, key_dim, depth, num_heads, down_ops) in \
                enumerate(zip(self.embed_dims, self.key_dim, self.depth,
                              self.num_heads, self.down_ops)):
            # print('-----------------', i, '-------------------')
            for _ in range(depth):
                self.blocks[-1].append(
                    Residual(
                        Attention(
                            embed_dim,
                            key_dim,
                            num_heads,
                            attn_ratio=attn_ratio,
                            activation=attention_activation,
                            resolution=resolution,
                        ), self.drop_path_rate))
                if mlp_ratio > 0:
                    # h = int(embed_dim * mlp_ratio)
                    self.blocks[-1].append(
                        Residual(
                            MLP(embed_dim, mlp_ratio, mlp_activation),
                            self.drop_path_rate))
            if i < len(self.depth) - 1:
                self.size.append(resolution)
                sub_resolution = (resolution - 1) // down_ops[3] + 1
                self.blocks.append([])
                self.blocks[-1].append(
                    AttentionSubsample(
                        *self.embed_dims[i:i + 2],
                        key_dim=down_ops[0],
                        num_heads=embed_dim // down_ops[0],
                        attn_ratio=down_ops[1],
                        activation=attention_activation,
                        stride=down_ops[3],
                        resolution=resolution,
                        sub_resolution=sub_resolution))
                resolution = sub_resolution
                if down_ops[2] > 0:  # mlp_ratio
                    self.blocks[-1].append(
                        Residual(
                            nn.Sequential(
                                MLP(self.embed_dims[i + 1], down_ops[2],
                                    mlp_activation)), self.drop_path_rate))
        self.blocks = [nn.Sequential(*i) for i in self.blocks]
        self.size.append(resolution)
        self.stages = nn.Sequential()
        for i in range(len(self.blocks)):
            self.stages.add_module('%d' % i, self.blocks[i])

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            assert 0 <= out_indices[i] < len(self.stages), \
                f'Invalid out_indices {index}.'
        out_indices = list(out_indices)

        self.out_indices = out_indices

        if self.deploy:
            replace_batchnorm(self)

    def switch_to_deploy(self):
        if self.deploy:
            return
        replace_batchnorm(self)
        self.deploy = True

    def forward(self, x):
        x = self.patch_embed(x)  # 2 3 224 224 -> 2 128 14 14
        x = x.flatten(2).transpose(1,
                                   2)  # 2 128 14 14 -> 2 128 196 -> 2 196 128
        outs = []
        for i, layer_name in enumerate(self.stages):
            x = layer_name(x)
            B, _, C = x.shape
            if i in self.out_indices:
                out = x.reshape(B, self.size[i], self.size[i],
                                C).permute(0, 3, 1, 2)
                outs.append(out)
        return tuple(outs)


def replace_batchnorm(net):
    for child_name, child in net.named_children():
        if hasattr(child, 'fuse'):
            setattr(net, child_name, child.fuse())
        elif isinstance(child, torch.nn.Conv2d):
            child.bias = torch.nn.Parameter(torch.zeros(child.weight.size(0)))
        elif isinstance(child, torch.nn.BatchNorm2d):
            setattr(net, child_name, torch.nn.Identity())
        else:
            replace_batchnorm(child)
