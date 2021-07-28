from copy import deepcopy
from typing import Sequence

import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmcv.runner.base_module import BaseModule, ModuleList

from ..builder import BACKBONES
from ..utils import PatchEmbed, PatchMerging, ShiftWindowMSA
from .base_backbone import BaseBackbone


class SwinBlock(BaseModule):
    """Swin Transformer block.

    Args:
        embed_dims (int): Number of input channels.
        input_resolution (Tuple[int, int]): The resolution of the input feature
            map.
        num_heads (int): Number of attention heads.
        window_size (int, optional): The height and width of the window.
            Defaults to 7.
        shift (bool, optional): Shift the attention window or not.
            Defaults to False.
        ffn_ratio (float, optional): The expansion ratio of feedforward network
            hidden layer channels. Defaults to 4.
        drop_path (float, optional): The drop path rate after attention and
            ffn. Defaults to 0.
        attn_cfgs (dict, optional): The extra config of Shift Window-MSA.
            Defaults to empty dict.
        ffn_cfgs (dict, optional): The extra config of FFN.
            Defaults to empty dict.
        norm_cfg (dict, optional): The config of norm layers.
            Defaults to dict(type='LN').
        auto_pad (bool, optional): Auto pad the feature map to be divisible by
            window_size, Defaults to False.
        init_cfg (dict, optional): The extra config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 input_resolution,
                 num_heads,
                 window_size=7,
                 shift=False,
                 ffn_ratio=4.,
                 drop_path=0.,
                 attn_cfgs=dict(),
                 ffn_cfgs=dict(),
                 norm_cfg=dict(type='LN'),
                 auto_pad=False,
                 init_cfg=None):

        super(SwinBlock, self).__init__(init_cfg)

        _attn_cfgs = {
            'embed_dims': embed_dims,
            'input_resolution': input_resolution,
            'num_heads': num_heads,
            'shift_size': window_size // 2 if shift else 0,
            'window_size': window_size,
            'dropout_layer': dict(type='DropPath', drop_prob=drop_path),
            'auto_pad': auto_pad,
            **attn_cfgs
        }
        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.attn = ShiftWindowMSA(**_attn_cfgs)

        _ffn_cfgs = {
            'embed_dims': embed_dims,
            'feedforward_channels': int(embed_dims * ffn_ratio),
            'num_fcs': 2,
            'ffn_drop': 0,
            'dropout_layer': dict(type='DropPath', drop_prob=drop_path),
            'act_cfg': dict(type='GELU'),
            **ffn_cfgs
        }
        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.ffn = FFN(**_ffn_cfgs)

    def forward(self, x):
        identity = x
        x = self.norm1(x)
        x = self.attn(x)
        x = x + identity

        identity = x
        x = self.norm2(x)
        x = self.ffn(x, identity=identity)
        return x


class SwinBlockSequence(BaseModule):
    """Module with successive Swin Transformer blocks and downsample layer.

    Args:
        embed_dims (int): Number of input channels.
        input_resolution (Tuple[int, int]): The resolution of the input feature
            map.
        depth (int): Number of successive swin transformer blocks.
        num_heads (int): Number of attention heads.
        downsample (bool, optional): Downsample the output of blocks by patch
            merging. Defaults to False.
        downsample_cfg (dict, optional): The extra config of the patch merging
            layer. Defaults to empty dict.
        drop_paths (Sequence[float] | float, optional): The drop path rate in
            each block. Defaults to 0.
        block_cfgs (Sequence[dict] | dict, optional): The extra config of each
            block. Defaults to empty dicts.
        auto_pad (bool, optional): Auto pad the feature map to be divisible by
            window_size, Defaults to False.
        init_cfg (dict, optional): The extra config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 input_resolution,
                 depth,
                 num_heads,
                 downsample=False,
                 downsample_cfg=dict(),
                 drop_paths=0.,
                 block_cfgs=dict(),
                 auto_pad=False,
                 init_cfg=None):
        super().__init__(init_cfg)

        if not isinstance(drop_paths, Sequence):
            drop_paths = [drop_paths] * depth

        if not isinstance(block_cfgs, Sequence):
            block_cfgs = [deepcopy(block_cfgs) for _ in range(depth)]

        self.blocks = ModuleList()
        for i in range(depth):
            _block_cfg = {
                'embed_dims': embed_dims,
                'input_resolution': input_resolution,
                'num_heads': num_heads,
                'shift': False if i % 2 == 0 else True,
                'drop_path': drop_paths[i],
                'auto_pad': auto_pad,
                **block_cfgs[i]
            }
            block = SwinBlock(**_block_cfg)
            self.blocks.append(block)

        if downsample:
            _downsample_cfg = {
                'input_resolution': input_resolution,
                'in_channels': embed_dims,
                'expansion_ratio': 2,
                'norm_cfg': dict(type='LN'),
                **downsample_cfg
            }
            self.downsample = PatchMerging(**_downsample_cfg)
        else:
            self.downsample = None

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        if self.downsample:
            x = self.downsample(x)
        return x


@BACKBONES.register_module()
class SwinTransformer(BaseBackbone):
    """ Swin Transformer
    A PyTorch implement of : `Swin Transformer:
    Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>`_

    Inspiration from
    https://github.com/microsoft/Swin-Transformer

    Args:
        arch (str | dict): Swin Transformer architecture
            Defaults to 'T'.
        img_size (int | tuple): The size of input image.
            Defaults to 224.
        in_channels (int): The num of input channels.
            Defaults to 3.
        drop_rate (float): Dropout rate after embedding.
            Defaults to 0.
        drop_path_rate (float): Stochastic depth rate.
            Defaults to 0.1.
        use_abs_pos_embed (bool): If True, add absolute position embedding to
            the patch embedding. Defaults to False.
        auto_pad (bool): If True, auto pad feature map to fit window_size.
            Defaults to False.
        norm_cfg (dict, optional): Config dict for normalization layer at end
            of backone. Defaults to dict(type='LN')
        stage_cfgs (Sequence | dict, optional): Extra config dict for each
            stage. Defaults to empty dict.
        patch_cfg (dict, optional): Extra config dict for patch embedding.
            Defaults to empty dict.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.

    Examples:
        >>> from mmcls.models import SwinTransformer
        >>> import torch
        >>> extra_config = dict(
        >>>     arch='tiny',
        >>>     stage_cfgs=dict(downsample_cfg={'kernel_size': 3,
        >>>                                     'expansion_ratio': 3}),
        >>>     auto_pad=True)
        >>> self = SwinTransformer(**extra_config)
        >>> inputs = torch.rand(1, 3, 224, 224)
        >>> output = self.forward(inputs)
        >>> print(output.shape)
        (1, 2592, 4)
    """
    arch_zoo = {
        **dict.fromkeys(['t', 'tiny'],
                        {'embed_dims': 96,
                         'depths':     [2, 2,  6,  2],
                         'num_heads':  [3, 6, 12, 24]}),
        **dict.fromkeys(['s', 'small'],
                        {'embed_dims': 96,
                         'depths':     [2, 2, 18,  2],
                         'num_heads':  [3, 6, 12, 24]}),
        **dict.fromkeys(['b', 'base'],
                        {'embed_dims': 128,
                         'depths':     [2, 2, 18,  2],
                         'num_heads':  [4, 8, 16, 32]}),
        **dict.fromkeys(['l', 'large'],
                        {'embed_dims': 192,
                         'depths':     [2,  2, 18,  2],
                         'num_heads':  [6, 12, 24, 48]}),
    }  # yapf: disable

    def __init__(self,
                 arch='T',
                 img_size=224,
                 in_channels=3,
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 use_abs_pos_embed=False,
                 auto_pad=False,
                 norm_cfg=dict(type='LN'),
                 stage_cfgs=dict(),
                 patch_cfg=dict(),
                 init_cfg=None):
        super(SwinTransformer, self).__init__(init_cfg)

        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
        else:
            essential_keys = {'embed_dims', 'depths', 'num_head'}
            assert isinstance(arch, dict) and set(arch) == essential_keys, \
                f'Custom arch needs a dict with keys {essential_keys}'
            self.arch_settings = arch

        self.embed_dims = self.arch_settings['embed_dims']
        self.depths = self.arch_settings['depths']
        self.num_heads = self.arch_settings['num_heads']
        self.num_layers = len(self.depths)
        self.use_abs_pos_embed = use_abs_pos_embed
        self.auto_pad = auto_pad

        _patch_cfg = {
            'img_size': img_size,
            'in_channels': in_channels,
            'embed_dims': self.embed_dims,
            'conv_cfg': dict(type='Conv2d', kernel_size=4, stride=4),
            'norm_cfg': dict(type='LN'),
            **patch_cfg
        }
        self.patch_embed = PatchEmbed(**_patch_cfg)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        if self.use_abs_pos_embed:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, self.embed_dims))

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        # stochastic depth
        total_depth = sum(self.depths)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]  # stochastic depth decay rule

        self.stages = ModuleList()
        embed_dims = self.embed_dims
        input_resolution = patches_resolution
        for i, (depth,
                num_heads) in enumerate(zip(self.depths, self.num_heads)):
            if isinstance(stage_cfgs, Sequence):
                stage_cfg = stage_cfgs[i]
            else:
                stage_cfg = deepcopy(stage_cfgs)
            downsample = True if i < self.num_layers - 1 else False
            _stage_cfg = {
                'embed_dims': embed_dims,
                'depth': depth,
                'num_heads': num_heads,
                'downsample': downsample,
                'input_resolution': input_resolution,
                'drop_paths': dpr[:depth],
                'auto_pad': auto_pad,
                **stage_cfg
            }

            stage = SwinBlockSequence(**_stage_cfg)
            self.stages.append(stage)

            dpr = dpr[depth:]
            if downsample:
                embed_dims = stage.downsample.out_channels
                input_resolution = stage.downsample.output_resolution

        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        else:
            self.norm = None

    def init_weights(self):
        super(SwinTransformer, self).init_weights()

        if self.use_abs_pos_embed:
            trunc_normal_(self.absolute_pos_embed, std=0.02)

    def forward(self, x):
        x = self.patch_embed(x)
        if self.use_abs_pos_embed:
            x = x + self.absolute_pos_embed
        x = self.drop_after_pos(x)

        for stage in self.stages:
            x = stage(x)

        x = self.norm(x) if self.norm else x

        return x.transpose(1, 2)
