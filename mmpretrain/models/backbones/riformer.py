# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

import torch
import torch.nn as nn
from mmcv.cnn.bricks import DropPath, build_norm_layer
from mmengine.model import BaseModule

from mmpretrain.registry import MODELS
from .base_backbone import BaseBackbone
from .poolformer import Mlp, PatchEmbed


class Affine(nn.Module):
    """Affine Transformation module.

    Args:
        in_features (int): Input dimension.
    """

    def __init__(self, in_features):
        super().__init__()
        self.affine = nn.Conv2d(
            in_features,
            in_features,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=in_features,
            bias=True)

    def forward(self, x):
        return self.affine(x) - x


class RIFormerBlock(BaseModule):
    """RIFormer Block.

    Args:
        dim (int): Embedding dim.
        mlp_ratio (float): Mlp expansion ratio. Defaults to 4.
        norm_cfg (dict): The config dict for norm layers.
            Defaults to ``dict(type='GN', num_groups=1)``.
        act_cfg (dict): The config dict for activation between pointwise
            convolution. Defaults to ``dict(type='GELU')``.
        drop (float): Dropout rate. Defaults to 0.
        drop_path (float): Stochastic depth rate. Defaults to 0.
        layer_scale_init_value (float): Init value for Layer Scale.
            Defaults to 1e-5.
        deploy (bool): Whether to switch the model structure to
            deployment mode. Default: False.
    """

    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 norm_cfg=dict(type='GN', num_groups=1),
                 act_cfg=dict(type='GELU'),
                 drop=0.,
                 drop_path=0.,
                 layer_scale_init_value=1e-5,
                 deploy=False):

        super().__init__()

        if deploy:
            self.norm_reparam = build_norm_layer(norm_cfg, dim)[1]
        else:
            self.norm1 = build_norm_layer(norm_cfg, dim)[1]
            self.token_mixer = Affine(in_features=dim)
        self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_cfg=act_cfg,
            drop=drop)

        # The following two techniques are useful to train deep RIFormers.
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.norm_cfg = norm_cfg
        self.dim = dim
        self.deploy = deploy

    def forward(self, x):
        if hasattr(self, 'norm_reparam'):
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) *
                self.norm_reparam(x))
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) *
                self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) *
                self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) *
                self.mlp(self.norm2(x)))
        return x

    def fuse_affine(self, norm, token_mixer):
        gamma_affn = token_mixer.affine.weight.reshape(-1)
        gamma_affn = gamma_affn - torch.ones_like(gamma_affn)
        beta_affn = token_mixer.affine.bias
        gamma_ln = norm.weight
        beta_ln = norm.bias
        return (gamma_ln * gamma_affn), (beta_ln * gamma_affn + beta_affn)

    def get_equivalent_scale_bias(self):
        eq_s, eq_b = self.fuse_affine(self.norm1, self.token_mixer)
        return eq_s, eq_b

    def switch_to_deploy(self):
        if self.deploy:
            return
        eq_s, eq_b = self.get_equivalent_scale_bias()
        self.norm_reparam = build_norm_layer(self.norm_cfg, self.dim)[1]
        self.norm_reparam.weight.data = eq_s
        self.norm_reparam.bias.data = eq_b
        self.__delattr__('norm1')
        if hasattr(self, 'token_mixer'):
            self.__delattr__('token_mixer')
        self.deploy = True


def basic_blocks(dim,
                 index,
                 layers,
                 mlp_ratio=4.,
                 norm_cfg=dict(type='GN', num_groups=1),
                 act_cfg=dict(type='GELU'),
                 drop_rate=.0,
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-5,
                 deploy=False):
    """generate RIFormer blocks for a stage."""
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (
            sum(layers) - 1)
        blocks.append(
            RIFormerBlock(
                dim,
                mlp_ratio=mlp_ratio,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                drop=drop_rate,
                drop_path=block_dpr,
                layer_scale_init_value=layer_scale_init_value,
                deploy=deploy,
            ))
    blocks = nn.Sequential(*blocks)

    return blocks


@MODELS.register_module()
class RIFormer(BaseBackbone):
    """RIFormer.

    A PyTorch implementation of RIFormer introduced by:
    `RIFormer: Keep Your Vision Backbone Effective But Removing Token Mixer <https://arxiv.org/abs/xxxx.xxxxx>`_

    Args:
        arch (str | dict): The model's architecture. If string, it should be
            one of architecture in ``RIFormer.arch_settings``. And if dict, it
            should include the following two keys:

            - layers (list[int]): Number of blocks at each stage.
            - embed_dims (list[int]): The number of channels at each stage.
            - mlp_ratios (list[int]): Expansion ratio of MLPs.
            - layer_scale_init_value (float): Init value for Layer Scale.

            Defaults to 'S12'.

        norm_cfg (dict): The config dict for norm layers.
            Defaults to ``dict(type='LN2d', eps=1e-6)``.
        act_cfg (dict): The config dict for activation between pointwise
            convolution. Defaults to ``dict(type='GELU')``.
        in_patch_size (int): The patch size of/? input image patch embedding.
            Defaults to 7.
        in_stride (int): The stride of input image patch embedding.
            Defaults to 4.
        in_pad (int): The padding of input image patch embedding.
            Defaults to 2.
        down_patch_size (int): The patch size of downsampling patch embedding.
            Defaults to 3.
        down_stride (int): The stride of downsampling patch embedding.
            Defaults to 2.
        down_pad (int): The padding of downsampling patch embedding.
            Defaults to 1.
        drop_rate (float): Dropout rate. Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        out_indices (Sequence | int): Output from which network position.
            Index 0-6 respectively corresponds to
            [stage1, downsampling, stage2, downsampling, stage3, downsampling, stage4]
            Defaults to -1, means the last stage.
        frozen_stages (int): Stages to be frozen (all param fixed).
            Defaults to -1, which means not freezing any parameters.
        deploy (bool): Whether to switch the model structure to
            deployment mode. Default: False.
        init_cfg (dict, optional): Initialization config dict
    """  # noqa: E501

    # --layers: [x,x,x,x], numbers of layers for the four stages
    # --embed_dims, --mlp_ratios:
    #     embedding dims and mlp ratios for the four stages
    # --downsamples: flags to apply downsampling or not in four blocks
    arch_settings = {
        's12': {
            'layers': [2, 2, 6, 2],
            'embed_dims': [64, 128, 320, 512],
            'mlp_ratios': [4, 4, 4, 4],
            'layer_scale_init_value': 1e-5,
        },
        's24': {
            'layers': [4, 4, 12, 4],
            'embed_dims': [64, 128, 320, 512],
            'mlp_ratios': [4, 4, 4, 4],
            'layer_scale_init_value': 1e-5,
        },
        's36': {
            'layers': [6, 6, 18, 6],
            'embed_dims': [64, 128, 320, 512],
            'mlp_ratios': [4, 4, 4, 4],
            'layer_scale_init_value': 1e-6,
        },
        'm36': {
            'layers': [6, 6, 18, 6],
            'embed_dims': [96, 192, 384, 768],
            'mlp_ratios': [4, 4, 4, 4],
            'layer_scale_init_value': 1e-6,
        },
        'm48': {
            'layers': [8, 8, 24, 8],
            'embed_dims': [96, 192, 384, 768],
            'mlp_ratios': [4, 4, 4, 4],
            'layer_scale_init_value': 1e-6,
        },
    }

    def __init__(self,
                 arch='s12',
                 in_channels=3,
                 norm_cfg=dict(type='GN', num_groups=1),
                 act_cfg=dict(type='GELU'),
                 in_patch_size=7,
                 in_stride=4,
                 in_pad=2,
                 down_patch_size=3,
                 down_stride=2,
                 down_pad=1,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 out_indices=-1,
                 frozen_stages=-1,
                 init_cfg=None,
                 deploy=False):

        super().__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'Unavailable arch, please choose from ' \
                f'({set(self.arch_settings)}) or pass a dict.'
            arch = self.arch_settings[arch]
        elif isinstance(arch, dict):
            assert 'layers' in arch and 'embed_dims' in arch, \
                f'The arch dict must have "layers" and "embed_dims", ' \
                f'but got {list(arch.keys())}.'

        layers = arch['layers']
        embed_dims = arch['embed_dims']
        mlp_ratios = arch['mlp_ratios'] \
            if 'mlp_ratios' in arch else [4, 4, 4, 4]
        layer_scale_init_value = arch['layer_scale_init_value'] \
            if 'layer_scale_init_value' in arch else 1e-5

        self.patch_embed = PatchEmbed(
            patch_size=in_patch_size,
            stride=in_stride,
            padding=in_pad,
            in_chans=in_channels,
            embed_dim=embed_dims[0])

        # set the main block in network
        network = []
        for i in range(len(layers)):
            stage = basic_blocks(
                embed_dims[i],
                i,
                layers,
                mlp_ratio=mlp_ratios[i],
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                layer_scale_init_value=layer_scale_init_value,
                deploy=deploy)
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if embed_dims[i] != embed_dims[i + 1]:
                # downsampling between two stages
                network.append(
                    PatchEmbed(
                        patch_size=down_patch_size,
                        stride=down_stride,
                        padding=down_pad,
                        in_chans=embed_dims[i],
                        embed_dim=embed_dims[i + 1]))

        self.network = nn.ModuleList(network)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = 7 + index
                assert out_indices[i] >= 0, f'Invalid out_indices {index}'
        self.out_indices = out_indices
        if self.out_indices:
            for i_layer in self.out_indices:
                layer = build_norm_layer(norm_cfg,
                                         embed_dims[(i_layer + 1) // 2])[1]
                layer_name = f'norm{i_layer}'
                self.add_module(layer_name, layer)

        self.frozen_stages = frozen_stages
        self._freeze_stages()
        self.deploy = deploy

    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        return x

    def forward_tokens(self, x):
        outs = []
        for idx, block in enumerate(self.network):
            x = block(x)
            if idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outs.append(x_out)
        return tuple(outs)

    def forward(self, x):
        # input embedding
        x = self.forward_embeddings(x)
        # through backbone
        x = self.forward_tokens(x)
        return x

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        for i in range(0, self.frozen_stages + 1):
            # Include both block and downsample layer.
            module = self.network[i]
            module.eval()
            for param in module.parameters():
                param.requires_grad = False
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                norm_layer.eval()
                for param in norm_layer.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        super(RIFormer, self).train(mode)
        self._freeze_stages()
        return self

    def switch_to_deploy(self):
        for m in self.modules():
            if isinstance(m, RIFormerBlock):
                m.switch_to_deploy()
        self.deploy = True
