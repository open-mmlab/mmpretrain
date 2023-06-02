# Copyright (c) 2022 OpenGVLab
# Copyright (c) OpenMMLab. All rights reserved.
# modified from
# https://github.com/OpenGVLab/InternImage/blob/master/classification/models/intern_image.py
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn.bricks import DropPath, build_activation_layer
from mmcv.cnn.bricks.transformer import FFN
from mmengine.model.weight_init import trunc_normal_
from ops_dcnv3 import modules as opsm

from mmpretrain.models.backbones.base_backbone import BaseBackbone
from mmpretrain.models.utils import CrossMultiheadAttention
from mmpretrain.registry import MODELS


class to_channels_first(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 3, 1, 2)


class to_channels_last(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 2, 3, 1)


def build_norm_layer(dim,
                     norm_layer,
                     in_format='channels_last',
                     out_format='channels_last',
                     eps=1e-6):
    layers = []
    if norm_layer == 'BN':
        if in_format == 'channels_last':
            layers.append(to_channels_first())
        layers.append(nn.BatchNorm2d(dim))
        if out_format == 'channels_last':
            layers.append(to_channels_last())
    elif norm_layer == 'LN':
        if in_format == 'channels_first':
            layers.append(to_channels_last())
        layers.append(nn.LayerNorm(dim, eps=eps))
        if out_format == 'channels_first':
            layers.append(to_channels_first())
    else:
        raise NotImplementedError(
            f'build_norm_layer does not support {norm_layer}')
    return nn.Sequential(*layers)


class AttentiveBlock(nn.Module):
    """Attentive Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to q, k, v.
            Default: False.
        qk_scale (float, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop (float, optional): Dropout rate. Default: 0.0.
        attn_drop (float, optional): Attention dropout rate. Default: 0.0.
        drop_path (float, optional): Stochastic depth rate. Default: 0.0.
        norm_cfg (dict, optional): Normalization layer.
            Default: dict(type='LN')
        out_dim (int, optional): Dimension of output. Default: None.
    """

    def __init__(self,
                 dim,
                 num_heads,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_cfg=dict(type='LN'),
                 out_dim=None):
        super().__init__()
        norm_layer = norm_cfg['type']
        self.norm1_q = build_norm_layer(dim, norm_layer, eps=1e-6)
        self.norm1_k = build_norm_layer(dim, norm_layer, eps=1e-6)
        self.norm1_v = build_norm_layer(dim, norm_layer, eps=1e-6)

        self.cross_dcn = CrossMultiheadAttention(
            embed_dims=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        if out_dim and out_dim != dim:
            self.cross_dcn.proj = nn.Linear(dim, out_dim)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x_q, x_kv, pos_q, pos_k):
        x_q = self.norm1_q(x_q + pos_q)
        x_k = self.norm1_k(x_kv + pos_k)
        x_v = self.norm1_v(x_kv)
        x = self.cross_dcn(x_q, k=x_k, v=x_v)
        return x


class AttentionPoolingBlock(AttentiveBlock):

    def forward(self, x):
        x_q = x.mean(1, keepdim=True)
        x_kv = x
        pos_q, pos_k = 0, 0
        x = super().forward(x_q, x_kv, pos_q, pos_k)
        x = x.squeeze(1)
        return x


class DownsampleLayer(nn.Module):
    """Downsample layer of InternImage.

    Args:
        channels (int): number of input channels
        norm_layer (str): normalization layer
    """

    def __init__(self, channels, norm_layer='LN'):
        super().__init__()
        self.conv = nn.Conv2d(
            channels,
            2 * channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)
        self.norm = build_norm_layer(2 * channels, norm_layer,
                                     'channels_first', 'channels_last')

    def forward(self, x):
        x = self.conv(x.permute(0, 3, 1, 2))
        x = self.norm(x)
        return x


class InternImageLayer(nn.Module):
    """Basic layer of InternImage.

    Args:
        core_op (nn.Module): core operation of InternImage
        channels (int): number of input channels
        groups (list): Groups of each block.
        mlp_ratio (float): ratio of mlp hidden features to input channels
        drop (float): dropout rate
        drop_path (float): drop path rate
        act_cfg (dict): activation layer
        norm_cfg (dict): normalization layer
        post_norm (bool): whether to use post normalization
        layer_scale (float): layer scale
        offset_scale (float): offset scale
        with_cp (bool): whether to use checkpoint
    """

    def __init__(
        self,
        core_op,
        channels,
        groups,
        mlp_ratio=4.,
        drop=0.,
        drop_path=0.,
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='LN'),
        post_norm=False,
        layer_scale=None,
        offset_scale=1.0,
        with_cp=False,
        dw_kernel_size=None,
        res_post_norm=False,
        center_feature_scale=False,
        remove_center=False,
    ):
        super().__init__()
        self.channels = channels
        self.groups = groups
        self.mlp_ratio = mlp_ratio
        self.with_cp = with_cp

        self.norm1 = build_norm_layer(channels, 'LN')
        self.post_norm = post_norm
        self.dcn = core_op(
            channels=channels,
            kernel_size=3,
            stride=1,
            pad=1,
            dilation=1,
            group=groups,
            offset_scale=offset_scale,
            act_layer=act_cfg['type'],
            norm_layer=norm_cfg['type'],
            dw_kernel_size=dw_kernel_size,
            center_feature_scale=center_feature_scale,
            remove_center=remove_center,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.norm2 = build_norm_layer(channels, 'LN')

        self.mlp = FFN(
            embed_dims=channels,
            feedforward_channels=int(channels * mlp_ratio),
            act_cfg=act_cfg,
            ffn_drop=drop,
            add_identity=False)

        self.layer_scale = layer_scale is not None
        if self.layer_scale:
            self.gamma1 = nn.Parameter(
                layer_scale * torch.ones(channels), requires_grad=True)
            self.gamma2 = nn.Parameter(
                layer_scale * torch.ones(channels), requires_grad=True)
        self.res_post_norm = res_post_norm
        if res_post_norm:
            self.res_post_norm1 = build_norm_layer(channels, 'LN')
            self.res_post_norm2 = build_norm_layer(channels, 'LN')

    def forward(self, x):

        def _inner_forward(x):
            if not self.layer_scale:
                if self.post_norm:
                    x = x + self.drop_path(self.norm1(self.dcn(x)))
                    x = x + self.drop_path(self.norm2(self.mlp(x)))
                elif self.res_post_norm:
                    x = x + self.drop_path(
                        self.res_post_norm1(self.dcn(self.norm1(x))))
                    x = x + self.drop_path(
                        self.res_post_norm2(self.mlp(self.norm2(x))))
                else:
                    x = x + self.drop_path(self.dcn(self.norm1(x)))
                    x = x + self.drop_path(self.mlp(self.norm2(x)))
                return x
            if self.post_norm:
                x = x + self.drop_path(self.gamma1 * self.norm1(self.dcn(x)))
                x = x + self.drop_path(self.gamma2 * self.norm2(self.mlp(x)))
            else:
                x = x + self.drop_path(self.gamma1 * self.dcn(self.norm1(x)))
                x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


class InternImageBlock(nn.Module):
    """Block of InternImage.

    Args:
        core_op (nn.Module): core operation of InternImage
        channels (int): number of input channels
        depths (list): Depth of each block.
        groups (list): Groups of each block.
        mlp_ratio (float): ratio of mlp hidden features to input channels
        drop (float): dropout rate
        drop_path (float): drop path rate
        act_cfg (dict): activation layer
        norm_cfg (dict): normalization layer
        post_norm (bool): whether to use post normalization
        layer_scale (float): layer scale
        offset_scale (float): offset scale
        with_cp (bool): whether to use checkpoint
    """

    def __init__(
        self,
        core_op,
        channels,
        depth,
        groups,
        downsample=True,
        mlp_ratio=4.,
        drop=0.,
        drop_path=0.,
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='LN'),
        post_norm=False,
        offset_scale=1.0,
        layer_scale=None,
        with_cp=False,
        dw_kernel_size=None,
        post_norm_block_ids=None,
        res_post_norm=False,
        center_feature_scale=False,
        remove_center=False,
    ):
        super().__init__()
        self.channels = channels
        self.depth = depth
        self.post_norm = post_norm
        self.center_feature_scale = center_feature_scale

        self.blocks = nn.ModuleList([
            InternImageLayer(
                core_op=core_op,
                channels=channels,
                groups=groups,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path[i]
                if isinstance(drop_path, list) else drop_path,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                post_norm=post_norm,
                layer_scale=layer_scale,
                offset_scale=offset_scale,
                with_cp=with_cp,
                dw_kernel_size=dw_kernel_size,
                res_post_norm=res_post_norm,
                center_feature_scale=center_feature_scale,
                remove_center=remove_center,
            ) for i in range(depth)
        ])
        if not self.post_norm or center_feature_scale:
            self.norm = build_norm_layer(channels, 'LN')
        self.post_norm_block_ids = post_norm_block_ids
        if post_norm_block_ids is not None:
            self.post_norms = nn.ModuleList([
                build_norm_layer(channels, 'LN', eps=1e-6)
                for _ in post_norm_block_ids
            ])
        self.downsample = DownsampleLayer(
            channels=channels,
            norm_layer=norm_cfg['type']) if downsample else None

    def forward(self, x, return_wo_downsample=False):
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if (self.post_norm_block_ids
                    is not None) and (i in self.post_norm_block_ids):
                index = self.post_norm_block_ids.index(i)
                x = self.post_norms[index](x)
        if not self.post_norm or self.center_feature_scale:
            x = self.norm(x)
        if return_wo_downsample:
            x_ = x
        if self.downsample is not None:
            x = self.downsample(x)

        if return_wo_downsample:
            return x, x_
        return x


@MODELS.register_module()
class InternImage(BaseBackbone):
    """ InternImage
        A PyTorch impl of : `InternImage: Exploring Large-Scale Vision Foundation Models with Deformable Convolutions`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        core_op (str): Core operator. Default: 'DCNv3'
        stem_channels (int): Number of the first stage. Default: 64
        stage_blocks (list): Depth of each block. Default: [3, 4, 18, 5]
        groups (list): Groups of each block. Default: [3, 6, 12, 24]
        num_classes (int): Number of classes. Default: 1000
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        drop_rate (float): Probability of an element to be zeroed. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        act_cfg (dict): Activation layer. Default: dict(type='GELU')
        norm_cfg (dict): Normalization layer. Default: dict(type='LN')
        layer_scale (bool): Whether to use layer scale. Default: False
        cls_scale (bool): Whether to use class scale. Default: False
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
        dw_kernel_size (int): Size of the dwconv. Default: None
        use_clip_projector (bool): Whether to use clip projector. Default: False
        level2_post_norm (bool): Whether to use level2 post norm. Default: False
        level2_post_norm_block_ids (list): Indexes of post norm blocks. Default: None
        res_post_norm (bool): Whether to use res post norm. Default: False
        center_feature_scale (bool): Whether to use center feature scale. Default: False
    """  # noqa: E501

    def __init__(self,
                 stem_channels=64,
                 stage_blocks=[3, 4, 18, 5],
                 groups=[3, 6, 12, 24],
                 mlp_ratio=4.,
                 drop_rate=0.,
                 drop_path_rate=0.2,
                 drop_path_type='linear',
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 layer_scale=None,
                 offset_scale=1.0,
                 post_norm=False,
                 cls_scale=1.5,
                 with_cp=False,
                 dw_kernel_size=None,
                 use_clip_projector=False,
                 level2_post_norm=False,
                 level2_post_norm_block_ids=None,
                 res_post_norm=False,
                 center_feature_scale=False,
                 remove_center=False,
                 init_cfg=None):
        super(InternImage, self).__init__(init_cfg)

        self.core_op = 'DCNv3'
        self.num_stages = len(stage_blocks)
        self.num_features = int(stem_channels * 2**(self.num_stages - 1))
        self.post_norm = post_norm
        self.mlp_ratio = mlp_ratio
        self.use_clip_projector = use_clip_projector
        self.level2_post_norm_block_ids = level2_post_norm_block_ids
        self.remove_center = remove_center
        self.act_cfg = act_cfg
        self.norm_cfg = norm_cfg

        # stem layer
        self._make_stem_layer(in_channels=3, stem_channels=stem_channels)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth decay rule
        total_depth = sum(stage_blocks)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]
        if drop_path_type == 'uniform':
            for i in range(len(dpr)):
                dpr[i] = drop_path_rate

        # InternImage Layers
        self.layers = nn.ModuleList()
        for i in range(self.num_stages):
            if level2_post_norm and i == 2:
                post_norm_block_ids = level2_post_norm_block_ids
            else:
                post_norm_block_ids = None

            layer = InternImageBlock(
                core_op=getattr(opsm, self.core_op),
                channels=int(stem_channels * 2**i),
                depth=stage_blocks[i],
                groups=groups[i],
                mlp_ratio=self.mlp_ratio,
                drop=drop_rate,
                drop_path=dpr[sum(stage_blocks[:i]):sum(stage_blocks[:i + 1])],
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                post_norm=post_norm,
                downsample=(i < self.num_stages - 1),
                layer_scale=layer_scale,
                offset_scale=offset_scale,
                with_cp=with_cp,
                dw_kernel_size=dw_kernel_size,
                post_norm_block_ids=post_norm_block_ids,
                res_post_norm=res_post_norm,
                center_feature_scale=center_feature_scale,
                remove_center=remove_center,
            )
            self.layers.append(layer)

        # Conv Head
        if not use_clip_projector:
            self.conv_head = nn.Sequential(
                nn.Conv2d(
                    self.num_features,
                    int(self.num_features * cls_scale),
                    kernel_size=1,
                    bias=False),
                build_norm_layer(
                    int(self.num_features * cls_scale), 'BN', 'channels_first',
                    'channels_first'), build_activation_layer(act_cfg))

        else:
            pretrain_embed_dim, _stride, attnpool_num_heads, clip_embed_dim \
                = 1024, 2, 16, 768
            self.dcnv3_head_x4 = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.num_features,
                    out_channels=pretrain_embed_dim * (_stride**2),
                    kernel_size=1), nn.PixelShuffle(_stride))
            self.dcnv3_head_x3 = nn.Conv2d(
                in_channels=self.num_features // 2,
                out_channels=pretrain_embed_dim,
                kernel_size=1)
            self.clip_projector = AttentionPoolingBlock(
                dim=pretrain_embed_dim,
                num_heads=attnpool_num_heads,
                qkv_bias=True,
                qk_scale=None,
                drop=0.,
                attn_drop=0.,
                norm_cfg=norm_cfg,
                out_dim=clip_embed_dim)
            norm_layer = norm_cfg['type']
            self.fc_norm = build_norm_layer(
                clip_embed_dim, norm_layer, eps=1e-6)

    def init_weights(self):
        super(InternImage, self).init_weights()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

            elif isinstance(m, getattr(opsm, self.core_op)):
                m._reset_parameters()

    def _make_stem_layer(self, in_channels, stem_channels):
        norm_layer = self.norm_cfg['type']
        self.patch_embed = nn.Sequential(
            nn.Conv2d(
                in_channels,
                stem_channels // 2,
                kernel_size=3,
                stride=2,
                padding=1),
            build_norm_layer(stem_channels // 2, norm_layer, 'channels_first',
                             'channels_first'),
            build_activation_layer(self.act_cfg),
            nn.Conv2d(
                stem_channels // 2,
                stem_channels,
                kernel_size=3,
                stride=2,
                padding=1),
            build_norm_layer(stem_channels, norm_layer, 'channels_first',
                             'channels_last'),
        )

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.conv_head(x.permute(0, 3, 1, 2))
        return (x, )

    def forward_features_seq_out(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        seq_out = []
        for layer in self.layers:
            x, x_ = layer(x, return_wo_downsample=True)
            seq_out.append(x_)
        return seq_out

    def forward_clip_projector(self, x):  # for InternImage-H/G
        xs = self.forward_features_seq_out(x)
        x1, x2, x3, x4 = xs

        x1 = x1.permute(0, 3, 1, 2)  # NHWC -> NCHW
        x2 = x2.permute(0, 3, 1, 2)  # NHWC -> NCHW
        x3 = x3.permute(0, 3, 1, 2)  # NHWC -> NCHW
        x4 = x4.permute(0, 3, 1, 2)  # NHWC -> NCHW

        x4 = self.dcnv3_head_x4(x4)
        x = x4
        x3 = self.dcnv3_head_x3(x3)
        x = x + x3

        x = x.flatten(-2).transpose(1, 2).contiguous()
        x = self.clip_projector(x)
        x = self.fc_norm(x)

        return (x, )

    def forward(self, x):
        if not self.use_clip_projector:
            # for InternImage-T/S/B/L/XL
            return self.forward_features(x)
        else:
            # for InternImage-H/G
            return self.forward_clip_projector(x)

    @staticmethod
    def _checkpoint_filter(state_dict, prefix, local_metadata, strict,
                           missing_keys, unexpected_keys, error_msgs):

        def internimage_to_mmpretrain():
            for k, v in state_dict['model'].items():
                if 'head.' in k and 'conv_head' not in k:
                    if 'weight' in k:
                        new_k = 'head.fc.weight'
                    else:
                        new_k = 'head.fc.bias'
                elif 'patch_embed' in k:
                    map_fun = {
                        'conv1': '0',
                        'norm1': '1',
                        'conv2': '3',
                        'norm2': '4'
                    }
                    new_k = k
                    for old, new in map_fun.items():
                        new_k = new_k.replace(old, new)
                    new_k = 'backbone.' + new_k

                elif 'levels' in k:
                    new_k = k.replace('levels', 'layers')
                    if 'mlp' in new_k:
                        new_k = new_k.replace('fc1', 'layers.0.0')
                        new_k = new_k.replace('fc2', 'layers.1')
                    new_k = 'backbone.' + new_k
                elif 'clip_projector.cross_dcn.k_bias' in k:
                    continue
                else:
                    new_k = 'backbone.' + k

                state_dict[new_k] = state_dict['model'][k]
            del state_dict['model']

        # The original weights need to be converted to mmpretrain format.
        # Some modules in the original weights starts with 'levels',
        # and in this implement they are replaced with 'layers'.
        if 'model' in state_dict and 'levels.0.blocks.0.norm1.0.weight'\
                in state_dict['model']:
            internimage_to_mmpretrain()
