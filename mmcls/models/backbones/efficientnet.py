import copy
import logging
import math

import torch.nn as nn
from mmcv.cnn import ConvModule, constant_init, kaiming_init
from mmcv.runner import load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm

from mmcls.models.utils import make_divisible
from ..builder import BACKBONES
from ..utils import EdgeResidual, InvertedResidual
from .base_backbone import BaseBackbone


def model_scaling(layer_setting, arch_setting):
    """Scaling operation to the layer's parameters according to the
    arch_setting."""
    new_layer_setting = copy.deepcopy(layer_setting)
    for layer_cfg in new_layer_setting:
        for block_cfg in layer_cfg:
            block_cfg[1] = make_divisible(block_cfg[1] * arch_setting[0], 8)

    split_layer_setting = [new_layer_setting[0]]
    for layer_cfg in new_layer_setting[1:-1]:
        tmp_index = [0]
        for i in range(len(layer_cfg) - 1):
            if layer_cfg[i + 1][1] != layer_cfg[i][1]:
                tmp_index.append(i + 1)
        tmp_index.append(len(layer_cfg))
        for i in range(len(tmp_index) - 1):
            split_layer_setting.append(layer_cfg[tmp_index[i]:tmp_index[i +
                                                                        1]])
    split_layer_setting.append(new_layer_setting[-1])

    num_of_layers = [len(layer_cfg) for layer_cfg in split_layer_setting[1:-1]]
    new_layers = [
        int(math.ceil(arch_setting[1] * num)) for num in num_of_layers
    ]

    merge_layer_setting = [split_layer_setting[0]]
    for i, layer_cfg in enumerate(split_layer_setting[1:-1]):
        if new_layers[i] <= num_of_layers[i]:
            tmp_layer_cfg = layer_cfg[:new_layers[i]]
        else:
            tmp_layer_cfg = copy.deepcopy(layer_cfg) + [layer_cfg[-1]] * (
                new_layers[i] - num_of_layers[i])
        if tmp_layer_cfg[0][4] == 1 and i != 0:
            merge_layer_setting[-1] += tmp_layer_cfg.copy()
        else:
            merge_layer_setting.append(tmp_layer_cfg.copy())
    merge_layer_setting.append(split_layer_setting[-1])

    return merge_layer_setting


@BACKBONES.register_module()
class EfficientNet(BaseBackbone):
    """EfficientNet backbone.

    Args:
        arch (str): Architecture of efficientnet. Default: b0.
        out_indices (Sequence[int]): Output from which stages.
            Default: (6, ).
        frozen_stages (int): Stages to be frozen (all param fixed).
            Default: -1, which means not freezing any parameters.
        conv_cfg (dict): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
    """

    # Parameters to build layers.
    # 'b' represents the architecture of normal EfficientNet family includes
    # 'b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8'.
    # 'e' represents the architecture of EfficientNet-EdgeTPU including 'es',
    # 'em', 'el'.
    # 9 parameters are needed to construct a layer, From left to right:
    # kernel_size, out_channel, se_ratio, act, stride, expand_ratio,
    # block_type.
    layer_settings = {
        'b': [[[3, 32, 0, 1, 2, 0, -1]],
              [[3, 16, 4, 1, 1, 1, 0]],
              [[3, 24, 4, 1, 2, 6, 0],
               [3, 24, 4, 1, 1, 6, 0]],
              [[5, 40, 4, 1, 2, 6, 0],
               [5, 40, 4, 1, 1, 6, 0]],
              [[3, 80, 4, 1, 2, 6, 0],
               [3, 80, 4, 1, 1, 6, 0],
               [3, 80, 4, 1, 1, 6, 0],
               [5, 112, 4, 1, 1, 6, 0],
               [5, 112, 4, 1, 1, 6, 0],
               [5, 112, 4, 1, 1, 6, 0]],
              [[5, 192, 4, 1, 2, 6, 0],
               [5, 192, 4, 1, 1, 6, 0],
               [5, 192, 4, 1, 1, 6, 0],
               [5, 192, 4, 1, 1, 6, 0],
               [3, 320, 4, 1, 1, 6, 0]],
              [[1, 1280, 0, 1, 1, 0, -1]]
              ],
        'e': [[[3, 32, 0, 0, 2, 0, -1]],
              [[3, 24, 0, 0, 1, 3, 1]],
              [[3, 32, 0, 0, 2, 8, 1],
               [3, 32, 0, 0, 1, 8, 1]],
              [[3, 48, 0, 0, 2, 8, 1],
               [3, 48, 0, 0, 1, 8, 1],
               [3, 48, 0, 0, 1, 8, 1],
               [3, 48, 0, 0, 1, 8, 1]],
              [[5, 96, 0, 0, 2, 8, 0],
               [5, 96, 0, 0, 1, 8, 0],
               [5, 96, 0, 0, 1, 8, 0],
               [5, 96, 0, 0, 1, 8, 0],
               [5, 96, 0, 0, 1, 8, 0],
               [5, 144, 0, 0, 1, 8, 0],
               [5, 144, 0, 0, 1, 8, 0],
               [5, 144, 0, 0, 1, 8, 0],
               [5, 144, 0, 0, 1, 8, 0]],
              [[5, 192, 0, 0, 2, 8, 0],
               [5, 192, 0, 0, 1, 8, 0]],
              [[1, 1280, 0, 0, 1, 0, -1]]
              ]
    }   # yapf: disable

    # Parameters to build different kinds of architecture.
    # From left to right: scaling factor for width, scaling factor for depth,
    # resolution, dropout ratio.
    arch_settings = {
        'b0': (1.0, 1.0, 224, 0.2),
        'b1': (1.0, 1.1, 240, 0.2),
        'b2': (1.1, 1.2, 260, 0.3),
        'b3': (1.2, 1.4, 300, 0.3),
        'b4': (1.4, 1.8, 380, 0.4),
        'b5': (1.6, 2.2, 456, 0.4),
        'b6': (1.8, 2.6, 528, 0.5),
        'b7': (2.0, 3.1, 600, 0.5),
        'b8': (2.2, 3.6, 672, 0.5),
        'es': (1.0, 1.0, 224, 0.2),
        'em': (1.0, 1.1, 240, 0.2),
        'el': (1.2, 1.4, 300, 0.3)
    }

    def __init__(self,
                 arch='b0',
                 out_indices=(6, ),
                 frozen_stages=0,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='Swish'),
                 norm_eval=False,
                 with_cp=False):
        super(EfficientNet, self).__init__()
        assert arch in self.arch_settings
        self.arch_setting = self.arch_settings[arch]
        self.layer_setting = self.layer_settings[arch[:1]]
        for index in out_indices:
            if index not in range(0, len(self.layer_setting)):
                raise ValueError('the item in out_indices must in '
                                 f'range(0, {len(self.layer_setting)}). '
                                 f'But received {index}')

        if frozen_stages not in range(len(self.layer_setting) + 1):
            raise ValueError('frozen_stages must be in range(0, '
                             f'{len(self.layer_setting) + 1}). '
                             f'But received {frozen_stages}')
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp

        self.layer_setting = model_scaling(self.layer_setting,
                                           self.arch_setting)
        self.dropout_rate = self.arch_setting[3]
        self.in_channels = make_divisible(self.layer_setting[0][0][1], 8)
        self.out_channels = self.layer_setting[-1][-1][1]

        self.layers = nn.ModuleList()
        self.layers.append(
            ConvModule(
                in_channels=3,
                out_channels=self.in_channels,
                kernel_size=self.layer_setting[0][0][0],
                stride=self.layer_setting[0][0][4],
                padding=self.layer_setting[0][0][0] // 2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=(self.act_cfg if self.layer_setting[0][0][3] else dict(
                    type='ReLU'))))
        self.make_layer()
        self.layers.append(
            ConvModule(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.layer_setting[-1][-1][0],
                stride=self.layer_setting[-1][-1][4],
                padding=self.layer_setting[-1][-1][0] // 2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=(self.act_cfg if self.layer_setting[-1][-1][3] else
                         dict(type='ReLU'))))

    def make_layer(self):
        for layer_cfg in self.layer_setting[1:-1]:
            layer = []
            for i, block_cfg in enumerate(layer_cfg):
                (kernel_size, out_channels, se_ratio, use_swish, stride,
                 expand_ratio, block_type) = block_cfg
                act_cfg = self.act_cfg if use_swish else dict(type='ReLU')

                mid_channels = int(self.in_channels * expand_ratio)
                out_channels = make_divisible(out_channels, 8)
                if se_ratio <= 0:
                    se_cfg = None
                else:
                    se_cfg = dict(
                        channels=mid_channels,
                        base_channels=self.in_channels,
                        ratio=se_ratio,
                        act_cfg=(act_cfg, dict(type='Sigmoid')))
                if block_type == 1:
                    block = EdgeResidual
                    if i > 0 and expand_ratio == 3:
                        with_residual = False
                        expand_ratio = 4
                    else:
                        with_residual = True
                    mid_channels = int(self.in_channels * expand_ratio)
                    if se_cfg is not None:
                        se_cfg = dict(
                            channels=mid_channels,
                            base_channels=self.in_channels,
                            ratio=se_ratio,
                            act_cfg=(act_cfg, dict(type='Sigmoid')))
                else:
                    block = InvertedResidual
                    with_residual = i
                layer.append(
                    block(
                        self.in_channels,
                        out_channels,
                        mid_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        se_cfg=se_cfg,
                        with_residual=with_residual,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=act_cfg,
                        with_cp=self.with_cp))
                self.in_channels = out_channels
            self.layers.append(nn.Sequential(*layer))

    def _freeze_stages(self):
        for i in range(self.frozen_stages):
            m = self.layers[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)

        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

    def train(self, mode=True):
        super(EfficientNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
