import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import ConvModule, constant_init, kaiming_init
from torch.nn.modules.batchnorm import _BatchNorm

from mmcls.models.utils import make_divisible
from ..builder import BACKBONES
from .base_backbone import BaseBackbone


class InvertedResidual(nn.Module):
    """InvertedResidual block for MobileNetV2.

    Args:
        inplanes (int): The input channels of the InvertedResidual block.
        planes (int): The output channels of the InvertedResidual block.
        stride (int): Stride of the middle (first) 3x3 convolution.
        expand_ratio (int): adjusts number of channels of the hidden layer
            in InvertedResidual by this amount.
        conv_cfg (dict): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU6').
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.

    Returns:
        Tensor: The output tensor
    """

    def __init__(self,
                 inplanes,
                 planes,
                 stride,
                 expand_ratio,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU6'),
                 with_cp=False):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2], f'stride must in [1, 2]. ' \
            f'But received {stride}.'
        self.with_cp = with_cp
        self.use_res_connect = self.stride == 1 and inplanes == planes
        hidden_dim = int(round(inplanes * expand_ratio))

        layers = []
        if expand_ratio != 1:
            layers.append(
                ConvModule(
                    in_channels=inplanes,
                    out_channels=hidden_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
        layers.extend([
            ConvModule(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=hidden_dim,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                in_channels=hidden_dim,
                out_channels=planes,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):

        def _inner_forward(x):
            if self.use_res_connect:
                return x + self.conv(x)
            else:
                return self.conv(x)

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        return out


@BACKBONES.register_module()
class MobileNetV2(BaseBackbone):
    """MobileNetV2 backbone.

    Args:
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Default: 1.0.
        out_indices (None or Sequence[int]): Output from which stages.
            Default: None
        frozen_stages (int): Stages to be frozen (all param fixed).
            Default: -1, which means not freezing any parameters.
        conv_cfg (dict): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU6').
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
    """

    # Parameters to build layers. 4 parameters are needed to construct a
    # layer, from left to right: expand_ratio, channel, num_blocks, stride.
    arch_settings = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2],
                     [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2],
                     [6, 320, 1, 1]]

    def __init__(self,
                 widen_factor=1.,
                 out_indices=None,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU6'),
                 norm_eval=False,
                 with_cp=False):
        super(MobileNetV2, self).__init__()
        self.widen_factor = widen_factor
        self.out_indices = out_indices
        if out_indices is not None:
            assert max(out_indices) < len(self.arch_settings)
        self.frozen_stages = frozen_stages
        assert frozen_stages < len(self.arch_settings)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp

        self.inplanes = make_divisible(32 * widen_factor, 8)

        self.conv1 = ConvModule(
            in_channels=3,
            out_channels=self.inplanes,
            kernel_size=3,
            stride=2,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.inverted_res_layers = []

        for i, layer_cfg in enumerate(self.arch_settings):
            expand_ratio, channel, num_blocks, stride = layer_cfg
            planes = make_divisible(channel * widen_factor, 8)
            inverted_res_layer = self.make_layer(
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                expand_ratio=expand_ratio)
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, inverted_res_layer)
            self.inverted_res_layers.append(layer_name)

        if widen_factor > 1.0:
            self.out_channel = int(1280 * widen_factor)
        else:
            self.out_channel = 1280

        self.conv2 = ConvModule(
            in_channels=self.inplanes,
            out_channels=self.out_channel,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def make_layer(self, planes, num_blocks, stride, expand_ratio):
        """ Stack InvertedResidual blocks to build a layer for MobileNetV2.

        Args:
            planes (int): planes of block.
            num_blocks (int): number of blocks.
            stride (int): stride of the first block. Default: 1
            expand_ratio (int): Expand the number of channels of the
                hidden layer in InvertedResidual by this ratio. Default: 6.
        """
        layers = []
        for i in range(num_blocks):
            if i >= 1:
                stride = 1
            layers.append(
                InvertedResidual(
                    self.inplanes,
                    planes,
                    stride,
                    expand_ratio=expand_ratio,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    with_cp=self.with_cp))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def init_weights(self, pretrained=None):
        if pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        x = self.conv1(x)

        outs = []
        for i, layer_name in enumerate(self.inverted_res_layers):
            inverted_res_layer = getattr(self, layer_name)
            x = inverted_res_layer(x)
            if self.out_indices is not None and i in self.out_indices:
                outs.append(x)

        x = self.conv2(x)

        if self.out_indices is None:
            return x
        else:
            return tuple(outs)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for param in self.conv1.parameters():
                param.requires_grad = False
        for i in range(1, self.frozen_stages + 1):
            layer = getattr(self, f'layer{i}')
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super(MobileNetV2, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
