# Copyright (c) OpenMMLab. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks import Conv2d, DropPath, build_activation_layer, build_norm_layer
from mmcv.cnn.bricks.transformer import PatchEmbed
from mmengine.model import BaseModule
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm

from mmpretrain.registry import MODELS
from .base_backbone import BaseBackbone


class ElementScale(nn.Module):
    """A learnable element-wise scaler."""

    def __init__(self, embed_dims, init_value=0., requires_grad=True):
        super(ElementScale, self).__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)),
            requires_grad=requires_grad
        )

    def forward(self, x):
        return x * self.scale


class ChannelAggregationFFN(BaseModule):
    """An implementation of FFN with Channel Aggregation.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`.
        feedforward_channels (int): The hidden dimension of FFNs.
        kernel_size (int): The depth-wise conv kernel size as the
            depth-wise convolution. Defaults to 3.
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 kernel_size=3,
                 act_cfg=dict(type='GELU'),
                 ffn_drop=0.,
                 init_cfg=None):
        super(ChannelAggregationFFN, self).__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.act_cfg = act_cfg

        self.fc1 = Conv2d(
            in_channels=embed_dims,
            out_channels=self.feedforward_channels,
            kernel_size=1)
        self.dwconv = Conv2d(
            in_channels=self.feedforward_channels,
            out_channels=self.feedforward_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=True,
            groups=self.feedforward_channels)
        self.act = build_activation_layer(act_cfg)
        self.fc2 = Conv2d(
            in_channels=feedforward_channels,
            out_channels=embed_dims,
            kernel_size=1)
        self.drop = nn.Dropout(ffn_drop)

        self.decompose = Conv2d(
            in_channels=self.feedforward_channels,  # C -> 1
            out_channels=1, kernel_size=1,
        )
        self.sigma = ElementScale(
            self.feedforward_channels, init_value=1e-5, requires_grad=True)
        self.decompose_act = build_activation_layer(act_cfg)

    def feat_decompose(self, x):
        # x_d: [B, C, H, W] -> [B, 1, H, W]
        x = x + self.sigma(x - self.decompose_act(self.decompose(x)))
        return x

    def forward(self, x):
        # proj 1
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        # proj 2
        x = self.feat_decompose(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MultiOrderDWConv(BaseModule):
    """Multi-order Features with Dilated DWConv Kernel.

    Args:
        embed_dims (int): Number of input channels.
        dw_dilation (list): Dilations of three DWConv layers.
        channel_split (list): The raletive ratio of three splited channels.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 dw_dilation=[1, 2, 3,],
                 channel_split=[1, 3, 4,],
                 init_cfg=None):
        super(MultiOrderDWConv, self).__init__(init_cfg=init_cfg)

        self.split_ratio = [i / sum(channel_split) for i in channel_split]
        self.embed_dims_1 = int(self.split_ratio[1] * embed_dims)
        self.embed_dims_2 = int(self.split_ratio[2] * embed_dims)
        self.embed_dims_0 = embed_dims - self.embed_dims_1 - self.embed_dims_2
        self.embed_dims = embed_dims
        assert len(dw_dilation) == len(channel_split) == 3
        assert 1 <= min(dw_dilation) and max(dw_dilation) <= 3
        assert embed_dims % sum(channel_split) == 0

        # basic DW conv
        self.DW_conv0 = Conv2d(
            in_channels=self.embed_dims,
            out_channels=self.embed_dims,
            kernel_size=5,
            padding=(1 + 4 * dw_dilation[0]) // 2,
            groups=self.embed_dims,
            stride=1, dilation=dw_dilation[0],
        )
        # DW conv 1
        self.DW_conv1 = Conv2d(
            in_channels=self.embed_dims_1,
            out_channels=self.embed_dims_1,
            kernel_size=5,
            padding=(1 + 4 * dw_dilation[1]) // 2,
            groups=self.embed_dims_1,
            stride=1, dilation=dw_dilation[1],
        )
        # DW conv 2
        self.DW_conv2 = Conv2d(
            in_channels=self.embed_dims_2,
            out_channels=self.embed_dims_2,
            kernel_size=7,
            padding=(1 + 6 * dw_dilation[2]) // 2,
            groups=self.embed_dims_2,
            stride=1, dilation=dw_dilation[2],
        )
        # a channel convolution
        self.PW_conv = Conv2d(  # point-wise convolution
            in_channels=embed_dims,
            out_channels=embed_dims,
            kernel_size=1)

    def forward(self, x):
        x_0 = self.DW_conv0(x)
        x_1 = self.DW_conv1(
            x_0[:, self.embed_dims_0: self.embed_dims_0+self.embed_dims_1, ...])
        x_2 = self.DW_conv2(
            x_0[:, self.embed_dims-self.embed_dims_2:, ...])
        x = torch.cat([
            x_0[:, :self.embed_dims_0, ...], x_1, x_2], dim=1)
        x = self.PW_conv(x)
        return x


class MultiOrderGatedAggregation(BaseModule):
    """Spatial Block with Multi-order Gated Aggregation.

    Args:
        embed_dims (int): Number of input channels.
        attn_dw_dilation (list): Dilations of three DWConv layers.
        attn_channel_split (list): The raletive ratio of splited channels.
        attn_act_cfg (dict, optional): The activation config for Spatial Block.
            Default: dict(type='SiLU').
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 attn_dw_dilation=[1, 2, 3],
                 attn_channel_split=[1, 3, 4],
                 attn_act_cfg=dict(type='SiLU'),
                 attn_force_fp32=False,
                 init_cfg=None):
        super(MultiOrderGatedAggregation, self).__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims
        self.attn_force_fp32 = attn_force_fp32
        self.proj_1 = Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)
        self.gate = Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)
        self.value = MultiOrderDWConv(
            embed_dims=embed_dims,
            dw_dilation=attn_dw_dilation,
            channel_split=attn_channel_split,
        )
        self.proj_2 = Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)

        # activation for gating and value
        self.act_value = build_activation_layer(attn_act_cfg)
        self.act_gate = build_activation_layer(attn_act_cfg)

        # decompose
        self.sigma = ElementScale(
            embed_dims, init_value=1e-5, requires_grad=True)

    def feat_decompose(self, x):
        x = self.proj_1(x)
        # x_d: [B, C, H, W] -> [B, C, 1, 1]
        x_d = F.adaptive_avg_pool2d(x, output_size=1)
        x = x + self.sigma(x - x_d)
        x = self.act_value(x)
        return x

    def forward_gating(self, g, v):
        """ Force to computing gating with fp32
        
        Warning: If you use `attn_force_fp32=True` during training, you
            should also keep it during evaluation, because the output results
            of whether to use `attn_force_fp32` are slightly different.
        """
        g = g.to(torch.float32)
        v = v.to(torch.float32)
        return self.proj_2(self.act_gate(g) * self.act_gate(v))

    def forward(self, x):
        shortcut = x.clone()
        # proj 1x1
        x = self.feat_decompose(x)
        # gating and value branch
        g = self.gate(x)
        v = self.value(x)
        # aggregation
        if not self.attn_force_fp32:
            x = self.proj_2(self.act_gate(g) * self.act_gate(v))
        else:
            x = self.forward_gating(self.act_gate(g), self.act_gate(v))
        x = x + shortcut
        return x


class MogaBlock(BaseModule):
    """A block of MogaNet.

    Args:
        embed_dims (int): Number of input channels.
        ffn_ratio (float): The expansion ratio of feedforward network hidden
            layer channels. Defaults to 4.
        drop_rate (float): Dropout rate after embedding. Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.1.
        act_cfg (dict, optional): The activation config for projections and FFNs.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        init_value (float): Init value for Layer Scale. Defaults to 1e-5.
        attn_dw_dilation (list): Dilations of three DWConv layers.
        attn_channel_split (list): The raletive ratio of splited channels.
        attn_act_cfg (dict): The activation config for the gating branch.
            Default: dict(type='SiLU').
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 ffn_ratio=4.,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='BN', eps=1e-5),
                 init_value=1e-5,
                 attn_dw_dilation=[1, 2, 3],
                 attn_channel_split=[1, 3, 4],
                 attn_act_cfg=dict(type='SiLU'),
                 attn_force_fp32=False,
                 init_cfg=None):
        super(MogaBlock, self).__init__(init_cfg=init_cfg)
        self.out_channels = embed_dims

        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]

        # spatial attention
        self.attn = MultiOrderGatedAggregation(
            embed_dims,
            attn_dw_dilation=attn_dw_dilation,
            attn_channel_split=attn_channel_split,
            attn_act_cfg=attn_act_cfg,
            attn_force_fp32=attn_force_fp32,
        )
        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]

        # channel MLP
        mlp_hidden_dim = int(embed_dims * ffn_ratio)
        self.mlp = ChannelAggregationFFN(  # DWConv + Channel Aggregation FFN
            embed_dims=embed_dims,
            feedforward_channels=mlp_hidden_dim,
            act_cfg=act_cfg,
            ffn_drop=drop_rate,
        )

        # init layer scale
        self.layer_scale_1 = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)), requires_grad=True)

    def forward(self, x):
        # spatial
        identity = x
        x = self.layer_scale_1 * self.attn(self.norm1(x))
        x = identity + self.drop_path(x)
        # channel
        identity = x
        x = self.layer_scale_2 * self.mlp(self.norm2(x))
        x = identity + self.drop_path(x)
        return x


class ConvPatchEmbed(PatchEmbed):
    """An implementation of Conv patch embedding layer.

    The differences between ConvPatchEmbed & ViT PatchEmbed:
        1. Use BN.
        2. Do not use 'flatten' and 'transpose'.
    """

    def __init__(self, *args, norm_cfg=dict(type='BN'), **kwargs):
        super(ConvPatchEmbed, self).__init__(*args, norm_cfg=norm_cfg, **kwargs)

    def forward(self, x):
        """
        Args:
            x (Tensor): Has shape (B, C, H, W). In most case, C is 3.
        Returns:
            tuple: Contains merged results and its spatial shape.
            - x (Tensor): Has shape (B, out_h * out_w, embed_dims)
            - out_size (tuple[int]): Spatial shape of x, arrange as
              (out_h, out_w).
        """

        if self.adaptive_padding:
            x = self.adaptive_padding(x)
        x = self.projection(x)
        out_size = (x.shape[2], x.shape[3])
        if self.norm is not None:
            x = self.norm(x)
        return x, out_size


class StackConvPatchEmbed(BaseModule):
    """An implementation of Stack Conv patch embedding layer.

    Args:
        in_features (int): The feature dimension.
        embed_dims (int): The output dimension of PatchEmbed.
        kernel_size (int): The conv kernel size of stack patch embedding.
            Defaults to 3.
        stride (int): The conv stride of stack patch embedding.
            Defaults to 2.
        act_cfg (dict, optional): The activation config in PatchEmbed.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer in PatchEmbed.
            Defaults: dict(type='BN').
    """

    def __init__(self,
                 in_channels,
                 embed_dims,
                 kernel_size=3,
                 stride=2,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='BN'),
                 init_cfg=None,
                ):
        super(StackConvPatchEmbed, self).__init__(init_cfg)

        self.projection = nn.Sequential(
            Conv2d(in_channels, embed_dims // 2, kernel_size=kernel_size,
                stride=stride, padding=kernel_size // 2),
            build_norm_layer(norm_cfg, embed_dims // 2)[1],
            build_activation_layer(act_cfg),
            Conv2d(embed_dims // 2, embed_dims, kernel_size=kernel_size,
                stride=stride, padding=kernel_size // 2),
            build_norm_layer(norm_cfg, embed_dims)[1],
        )

    def forward(self, x):
        x = self.projection(x)
        out_size = (x.shape[2], x.shape[3])
        return x, out_size


@MODELS.register_module()
class MogaNet(BaseBackbone):
    """MogaNet.

    A PyTorch implementation of MogaNet introduced by:
    `Efficient Multi-order Gated Aggregation Network <https://arxiv.org/abs/2211.03295>`_

    Modified from the `official repo
    <https://github.com/Westlake-AI/openmixup/blob/main/openmixup/models/backbones/moganet.py>`.

    Args:
        arch (str | dict): MogaNet architecture.
            If use string, choose from 'xtiny', 'tiny', 'small', 'base' and 'large'.
            If use dict, it should have below keys:

            - **embed_dims** (List[int]): The dimensions of embedding.
            - **depths** (List[int]): The number of blocks in each stage.
            - **ffn_ratios** (List[int]): The number of expansion ratio of
            feedforward network hidden layer channels.

            Defaults to 'tiny'.
        patch_sizes (List[int | tuple]): The patch size in patch embeddings.
            Defaults to [3, 3, 3, 3].
        in_channels (int): The num of input channels. Defaults to 3.
        drop_rate (float): Dropout rate after embedding. Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.1.
        init_value (float): Init value for Layer Scale. Defaults to 1e-5.
        out_indices (Sequence | int): Output from which network position.
            Index 0-6 respectively corresponds to
            [stage1, downsampling, stage2, downsampling, stage3, downsampling, stage4]
            Defaults to -1, means the last stage.
        frozen_stages (int): Stages to be frozen (all param fixed).
            Defaults to 0, which means not freezing any parameters.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Defaults to False.
        stem_norm_cfg (dict): Config dict for normalization layer for stems.
            Defaults to ``dict(type='LN')``.
        conv_norm_cfg (dict): Config dict for convolution normalization layer.
            Defaults to ``dict(type='BN')``.
        patchembed_types (list): The type of PatchEmbedding in each stage.
            Defaults to ``['ConvEmbed', 'Conv', 'Conv', 'Conv',]``.
        attn_dw_dilation (list): The dilate rate of depth-wise convolutions in
            Moga Blocks. Defaults to ``[1, 2, 3]``.
        attn_channel_split (list): The channel split rate of three depth-wise
            convolutions in Moga Blocks. Defaults to ``[1, 3, 4]``, i.e.,
            divided into ``[1/8, 3/8, 4/8]``.
        attn_act_cfg (dict): Config dict for activation of gating in Moga
            Blocks. Defaults to ``dict(type='SiLU')``.
        attn_final_dilation (bool): Whether to adopt dilated depth-wise
            convolutions in the final stage. Defaults to True.
        attn_force_fp32 (bool): Whether to force the gating running with fp32.
            Warning: If you use `attn_force_fp32=True` during training, you
            should also keep it during evaluation, because the output results
            of whether to use `attn_force_fp32` are different. Defaults to True.
        block_cfgs (Sequence[dict] | dict): The extra config of each block.
            Defaults to empty dicts.
        init_cfg (dict, optional): Initialization config dict
    """

    arch_settings = {
        **dict.fromkeys(['xt', 'x-tiny'],
                        {'embed_dims': [32, 64, 96, 192],
                         'depths': [3, 3, 10, 2],
                         'ffn_ratios': [8, 8, 4, 4]}),
        **dict.fromkeys(['t', 'tiny'],
                        {'embed_dims': [32, 64, 128, 256],
                         'depths': [3, 3, 12, 2],
                         'ffn_ratios': [8, 8, 4, 4]}),
        **dict.fromkeys(['s', 'small'],
                        {'embed_dims': [64, 128, 320, 512],
                         'depths': [2, 3, 12, 2],
                         'ffn_ratios': [8, 8, 4, 4]}),
        **dict.fromkeys(['b', 'base'],
                        {'embed_dims': [64, 160, 320, 512],
                         'depths': [4, 6, 22, 3],
                         'ffn_ratios': [8, 8, 4, 4]}),
        **dict.fromkeys(['l', 'large'],
                        {'embed_dims': [64, 160, 320, 640],
                         'depths': [4, 6, 44, 4],
                         'ffn_ratios': [8, 8, 4, 4]}),
        **dict.fromkeys(['xl', 'x-large'],
                        {'embed_dims': [96, 192, 480, 960],
                         'depths': [6, 6, 44, 4],
                         'ffn_ratios': [8, 8, 4, 4]}),
    }  # yapf: disable

    def __init__(self,
                 arch='tiny',
                 patch_sizes=[3, 3, 3, 3],
                 in_channels=3,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 init_value=1e-5,
                 out_indices=(3, ),
                 frozen_stages=-1,
                 norm_eval=False,
                 stem_norm_cfg=dict(type='BN', eps=1e-5),
                 conv_norm_cfg=dict(type='BN', eps=1e-5),
                 patchembed_types=['ConvEmbed', 'Conv', 'Conv', 'Conv',],
                 attn_dw_dilation=[1, 2, 3],
                 attn_channel_split=[1, 3, 4],
                 attn_act_cfg=dict(type='SiLU'),
                 attn_final_dilation=True,
                 attn_force_fp32=False,
                 block_cfgs=dict(),
                 init_cfg=None):

        super().__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in self.arch_settings, \
                f'Unavailable arch, please choose from ' \
                f'({set(self.arch_settings)}) or pass a dict.'
            self.arch_settings = self.arch_settings[arch]
        else:
            essential_keys = {'embed_dims', 'depths', 'ffn_ratios'}
            assert isinstance(arch, dict) and set(arch) == essential_keys, \
                f'Custom arch needs a dict with keys {essential_keys}'
            self.arch_settings = arch

        self.embed_dims = self.arch_settings['embed_dims']
        self.depths = self.arch_settings['depths']
        self.ffn_ratios = self.arch_settings['ffn_ratios']
        self.num_stages = len(self.depths)
        self.out_indices = out_indices
        self.norm_eval = norm_eval
        self.attn_force_fp32 = attn_force_fp32
        self.use_layer_norm = stem_norm_cfg['type'] == 'LN'
        assert stem_norm_cfg['type'] in ['BN', 'SyncBN', 'LN', 'LN2d',]
        assert len(patchembed_types) == self.num_stages

        total_depth = sum(self.depths)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]  # stochastic depth decay rule

        cur_block_idx = 0
        for i, depth in enumerate(self.depths):
            if i == 0 and patchembed_types[i] == "ConvEmbed":
                assert patch_sizes[i] <= 3
                patch_embed = StackConvPatchEmbed(
                    in_channels=in_channels,
                    embed_dims=self.embed_dims[i],
                    kernel_size=patch_sizes[i],
                    stride=patch_sizes[i] // 2 + 1,
                    norm_cfg=conv_norm_cfg,
                )
            else:
                patch_embed = ConvPatchEmbed(
                    in_channels=in_channels if i == 0 else self.embed_dims[i - 1],
                    input_size=None,
                    embed_dims=self.embed_dims[i],
                    kernel_size=patch_sizes[i],
                    stride=patch_sizes[i] // 2 + 1,
                    padding=(patch_sizes[i] // 2, patch_sizes[i] // 2),
                    norm_cfg=conv_norm_cfg)

            if i == self.num_stages - 1 and not attn_final_dilation:
                attn_dw_dilation = [1, 2, 1]
            blocks = nn.ModuleList([
                MogaBlock(
                    embed_dims=self.embed_dims[i],
                    ffn_ratio=self.ffn_ratios[i],
                    drop_rate=drop_rate,
                    drop_path_rate=dpr[cur_block_idx + j],
                    norm_cfg=conv_norm_cfg,
                    init_value=init_value,
                    attn_dw_dilation=attn_dw_dilation,
                    attn_channel_split=attn_channel_split,
                    attn_act_cfg=attn_act_cfg,
                    attn_force_fp32=attn_force_fp32,
                    **block_cfgs) for j in range(depth)
            ])
            cur_block_idx += depth
            norm = build_norm_layer(stem_norm_cfg, self.embed_dims[i])[1]

            self.add_module(f'patch_embed{i + 1}', patch_embed)
            self.add_module(f'blocks{i + 1}', blocks)
            self.add_module(f'norm{i + 1}', norm)

        self.frozen_stages = frozen_stages
        self._freeze_stages()

    def forward(self, x):
        outs = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f'patch_embed{i + 1}')
            blocks = getattr(self, f'blocks{i + 1}')
            norm = getattr(self, f'norm{i + 1}')

            x, hw_shape = patch_embed(x)
            for block in blocks:
                x = block(x)
            if self.use_layer_norm:
                x = x.flatten(2).transpose(1, 2)
                x = norm(x)
                x = x.reshape(-1, *hw_shape,
                              blocks.out_channels).permute(0, 3, 1, 2).contiguous()
            else:
                x = norm(x)

            if i in self.out_indices:
                outs.append(x)

        return tuple(outs)

    def _freeze_stages(self):
        for i in range(0, self.frozen_stages + 1):
            # freeze patch embed
            m = getattr(self, f'patch_embed{i + 1}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
            # freeze blocks
            m = getattr(self, f'blocks{i + 1}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
            # freeze norm
            m = getattr(self, f'norm{i + 1}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super(MogaNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, (_BatchNorm, nn.SyncBatchNorm)):
                    m.eval()
