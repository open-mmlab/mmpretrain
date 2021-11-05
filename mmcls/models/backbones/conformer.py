import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmcv.cnn.bricks.drop import DropPath
from mmcv.cnn.utils.weight_init import trunc_normal_

from mmcls.utils import get_root_logger
from ..builder import BACKBONES
from .base_backbone import BaseBackbone
from .vision_transformer import TransformerEncoderLayer


class ConvBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 res_conv=False,
                 groups=1,
                 drop_path=0.,
                 norm_cfg=dict(type='BN', eps=1e-6),
                 act_cfg=dict(type='ReLU', inplace=True)):
        super(ConvBlock, self).__init__()

        expansion = 4
        mid_channels = out_channels // expansion

        self.conv1 = nn.Conv2d(
            in_channels,
            mid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)
        self.bn1 = build_norm_layer(norm_cfg, mid_channels)[1]
        self.act1 = build_activation_layer(act_cfg)

        self.conv2 = nn.Conv2d(
            mid_channels,
            mid_channels,
            kernel_size=3,
            stride=stride,
            groups=groups,
            padding=1,
            bias=False)
        self.bn2 = build_norm_layer(norm_cfg, mid_channels)[1]
        self.act2 = build_activation_layer(act_cfg)

        self.conv3 = nn.Conv2d(
            mid_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)
        self.bn3 = build_norm_layer(norm_cfg, out_channels)[1]
        self.act3 = build_activation_layer(act_cfg)

        if res_conv:
            self.residual_conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False)
            self.residual_bn = build_norm_layer(norm_cfg, out_channels)[1]

        self.res_conv = res_conv
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x, fusion_features=None, out_conv2=True):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x) if fusion_features is None else self.conv2(
            x + fusion_features)
        x = self.bn2(x)
        x2 = self.act2(x)

        x = self.conv3(x2)
        x = self.bn3(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.res_conv:
            residual = self.residual_conv(residual)
            residual = self.residual_bn(residual)

        x += residual
        x = self.act3(x)

        if out_conv2:
            return x, x2
        else:
            return x


class FCUDown(nn.Module):
    """CNN feature maps -> Transformer patch embeddings."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 dw_stride,
                 with_cls_token=True,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 act_cfg=dict(type='GELU')):
        super(FCUDown, self).__init__()
        self.dw_stride = dw_stride
        self.with_cls_token = with_cls_token

        self.conv_project = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.sample_pooling = nn.AvgPool2d(
            kernel_size=dw_stride, stride=dw_stride)

        self.ln = build_norm_layer(norm_cfg, out_channels)[1]
        self.act = build_activation_layer(act_cfg)

    def forward(self, x, x_t):
        x = self.conv_project(x)  # [N, C, H, W]

        x = self.sample_pooling(x).flatten(2).transpose(1, 2)
        x = self.ln(x)
        x = self.act(x)

        if self.with_cls_token:
            x = torch.cat([x_t[:, 0][:, None, :], x], dim=1)

        return x


class FCUUp(nn.Module):
    """Transformer patch embeddings -> CNN feature maps."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 up_stride,
                 with_cls_token=True,
                 norm_cfg=dict(type='BN', eps=1e-6),
                 act_cfg=dict(type='ReLU', inplace=True)):
        super(FCUUp, self).__init__()

        self.up_stride = up_stride
        self.with_cls_token = with_cls_token

        self.conv_project = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn = build_norm_layer(norm_cfg, out_channels)[1]
        self.act = build_activation_layer(act_cfg)

    def forward(self, x, H, W):
        B, _, C = x.shape
        # [N, 197, 384] -> [N, 196, 384] -> [N, 384, 196] -> [N, 384, 14, 14]
        if self.with_cls_token:
            x_r = x[:, 1:].transpose(1, 2).reshape(B, C, H, W)
        else:
            x_r = x.transpose(1, 2).reshape(B, C, H, W)

        x_r = self.act(self.bn(self.conv_project(x_r)))

        return F.interpolate(
            x_r, size=(H * self.up_stride, W * self.up_stride))


class ConvTransBlock(nn.Module):
    """Basic module for Conformer, keep feature maps for CNN block and patch
    embeddings for transformer encoder block."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 res_conv,
                 stride,
                 dw_stride,
                 embed_dim,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 with_cls_token=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 last_fusion=False,
                 groups=1):

        super(ConvTransBlock, self).__init__()
        expansion = 4
        self.cnn_block = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            res_conv=res_conv,
            stride=stride,
            groups=groups)

        if last_fusion:
            self.fusion_block = ConvBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                stride=2,
                res_conv=True,
                groups=groups,
                drop_path=drop_path_rate)
        else:
            self.fusion_block = ConvBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                groups=groups,
                drop_path=drop_path_rate)

        self.squeeze_block = FCUDown(
            in_channels=out_channels // expansion,
            out_channels=embed_dim,
            dw_stride=dw_stride,
            with_cls_token=with_cls_token)

        self.expand_block = FCUUp(
            in_channels=embed_dim,
            out_channels=out_channels // expansion,
            up_stride=dw_stride,
            with_cls_token=with_cls_token)

        self.trans_block = TransformerEncoderLayer(
            embed_dims=embed_dim,
            num_heads=num_heads,
            feedforward_channels=int(embed_dim * mlp_ratio),
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            attn_drop_rate=attn_drop_rate,
            qkv_bias=qkv_bias,
            norm_cfg=dict(type='LN', eps=1e-6))

        self.dw_stride = dw_stride
        self.embed_dim = embed_dim
        self.last_fusion = last_fusion

    def forward(self, cnn_input, trans_input):
        x, x_conv2 = self.cnn_block(cnn_input, out_conv2=True)

        _, _, H, W = x_conv2.shape

        # Convert the feature map of conv2 to transformer embedding
        # and concat with class token.
        conv2_embedding = self.squeeze_block(x_conv2, trans_input)

        trans_output = self.trans_block(conv2_embedding + trans_input)

        # Convert the transformer output embedding to feature map
        trans_features = self.expand_block(trans_output, H // self.dw_stride,
                                           W // self.dw_stride)
        x = self.fusion_block(
            x, fusion_features=trans_features, out_conv2=False)

        return x, trans_output


@BACKBONES.register_module()
class Conformer(BaseBackbone):
    """Vision Transformer with support for patch or hybrid CNN input stage."""
    arch_zoo = {
        **dict.fromkeys(['t', 'tiny'],
                        {'embed_dims': 384,
                         'channel_ratio': 1,
                         'num_heads': 6,
                         'depths': 12
                         }),
        **dict.fromkeys(['s', 'small'],
                        {'embed_dims': 384,
                         'channel_ratio': 4,
                         'num_heads': 6,
                         'depths': 12
                         }),
        **dict.fromkeys(['b', 'base'],
                        {'embed_dims': 576,
                         'channel_ratio': 6,
                         'num_heads': 9,
                         'depths': 12
                         }),
    }  # yapf: disable

    _version = 1

    def __init__(self,
                 arch='T',
                 patch_size=16,
                 base_channel=64,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 with_cls_token=True,
                 drop_path_rate=0.,
                 norm_eval=True,
                 frozen_stages=0,
                 out_indices=(12, ),
                 init_cfg=None):

        # Transformer
        super().__init__(init_cfg)

        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
        else:
            essential_keys = {
                'embed_dims', 'depths', 'num_heads', 'channel_ratio'
            }
            assert isinstance(arch, dict) and set(arch) == essential_keys, \
                f'Custom arch needs a dict with keys {essential_keys}'
            self.arch_settings = arch

        self.num_features = self.embed_dims = self.arch_settings['embed_dims']
        self.depths = self.arch_settings['depths']
        self.num_heads = self.arch_settings['num_heads']
        self.channel_ratio = self.arch_settings['channel_ratio']
        self.out_indices = out_indices
        self.init_cfg = init_cfg

        self.norm_eval = norm_eval
        self.frozen_stages = frozen_stages

        self.with_cls_token = with_cls_token
        if self.with_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims))

        self.trans_dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, self.depths)
        ]  # stochastic depth decay rule

        # Stem stage: get the feature maps by conv block (copied resnet.py)
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3,
            bias=False)  # 1 / 2 [112, 112]
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1)  # 1 / 4 [56, 56]

        # 1 stage
        stage_1_channel = int(base_channel * self.channel_ratio)
        trans_dw_stride = patch_size // 4
        self.conv_1 = ConvBlock(
            in_channels=64,
            out_channels=stage_1_channel,
            res_conv=True,
            stride=1)
        self.trans_patch_conv = nn.Conv2d(
            64,
            self.embed_dims,
            kernel_size=trans_dw_stride,
            stride=trans_dw_stride,
            padding=0)

        self.trans_1 = TransformerEncoderLayer(
            embed_dims=self.embed_dims,
            num_heads=self.num_heads,
            feedforward_channels=int(self.embed_dims * mlp_ratio),
            drop_path_rate=self.trans_dpr[0],
            qkv_bias=qkv_bias,
            norm_cfg=dict(type='LN', eps=1e-6))

        # 2~4 stage
        init_stage = 2
        fin_stage = self.depths // 3 + 1
        for i in range(init_stage, fin_stage):
            self.add_module(
                f'conv_trans_{i}',
                ConvTransBlock(
                    stage_1_channel,
                    stage_1_channel,
                    False,
                    1,
                    dw_stride=trans_dw_stride,
                    embed_dim=self.embed_dims,
                    num_heads=self.num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_path_rate=self.trans_dpr[i - 1],
                    with_cls_token=self.with_cls_token))

        stage_2_channel = int(base_channel * self.channel_ratio * 2)
        # 5~8 stage
        init_stage = fin_stage  # 5
        fin_stage = fin_stage + self.depths // 3  # 9
        for i in range(init_stage, fin_stage):
            stride = 2 if i == init_stage else 1
            in_channel = stage_1_channel if i == init_stage \
                else stage_2_channel
            res_conv = True if i == init_stage else False
            self.add_module(
                f'conv_trans_{i}',
                ConvTransBlock(
                    in_channel,
                    stage_2_channel,
                    res_conv,
                    stride,
                    dw_stride=trans_dw_stride // 2,
                    embed_dim=self.embed_dims,
                    num_heads=self.num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_path_rate=self.trans_dpr[i - 1],
                    with_cls_token=self.with_cls_token))

        stage_3_channel = int(base_channel * self.channel_ratio * 2 * 2)
        # 9~12 stage
        init_stage = fin_stage  # 9
        fin_stage = fin_stage + self.depths // 3  # 13
        for i in range(init_stage, fin_stage):
            stride = 2 if i == init_stage else 1
            in_channel = stage_2_channel if i == init_stage \
                else stage_3_channel
            res_conv = True if i == init_stage else False
            last_fusion = True if i == self.depths else False
            self.add_module(
                f'conv_trans_{i}',
                ConvTransBlock(
                    in_channel,
                    stage_3_channel,
                    res_conv,
                    stride,
                    dw_stride=trans_dw_stride // 4,
                    embed_dim=self.embed_dims,
                    num_heads=self.num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_path_rate=self.trans_dpr[i - 1],
                    with_cls_token=self.with_cls_token,
                    last_fusion=last_fusion))
        self.fin_stage = fin_stage

        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.trans_norm = nn.LayerNorm(self.embed_dims)

        if self.with_cls_token:
            trunc_normal_(self.cls_token, std=.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)

        if hasattr(m, 'zero_init_last_bn'):
            m.zero_init_last_bn()

    def init_weights(self):
        super(Conformer, self).init_weights()
        logger = get_root_logger()

        if (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            # Suppress default init if use pretrained model.
            return
        else:
            logger.info(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            self.apply(self._init_weights)

    def forward(self, x):
        output = []
        B = x.shape[0]
        if self.with_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)

        # stem
        x_base = self.maxpool(self.act1(self.bn1(self.conv1(x))))

        # 1 stage [N, 64, 56, 56] -> [N, 128, 56, 56]
        x = self.conv_1(x_base, out_conv2=False)
        x_t = self.trans_patch_conv(x_base).flatten(2).transpose(1, 2)
        if self.with_cls_token:
            x_t = torch.cat([cls_tokens, x_t], dim=1)
        x_t = self.trans_1(x_t)

        # 2 ~ final
        for i in range(2, self.fin_stage):
            stage = getattr(self, f'conv_trans_{i}')
            x, x_t = stage(x, x_t)
            if i in self.out_indices:
                if self.with_cls_token:
                    output.append([
                        self.pooling(x).flatten(1),
                        self.trans_norm(x_t)[:, 0]
                    ])
                else:
                    # if no class token, use the mean patch token
                    # as the transformer feature.
                    output.append([
                        self.pooling(x).flatten(1),
                        self.trans_norm(x_t).mean(dim=1)
                    ])

        return tuple(output)
