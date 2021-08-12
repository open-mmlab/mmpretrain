import mmcv
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule

from .make_divisible import make_divisible


# class SELayer(nn.Module):
class SELayer(BaseModule):
    """Squeeze-and-Excitation Module.

    Args:
        channels (int): The input (and output) channels of the SE layer.
        ratio (int): Squeeze ratio in SELayer, the intermediate channel will be
            ``int(channels/ratio)``. Default: 16.
        conv_cfg (None or dict): Config dict for convolution layer.
            Default: None, which means using conv2d.
        act_cfg (dict or Sequence[dict]): Config dict for activation layer.
            If act_cfg is a dict, two activation layers will be configurated
            by this dict. If act_cfg is a sequence of dicts, the first
            activation layer will be configurated by the first dict and the
            second activation layer will be configurated by the second dict.
            Default: (dict(type='ReLU'), dict(type='Sigmoid'))
        base_channels (None or int): The base channel number to calculate the
            channel number of the middle feature maps. Default: None, which
            means using `channels` to be the `base_channels`.
    """

    def __init__(self,
                 channels,
                 ratio=16,
                 bias='auto',
                 conv_cfg=None,
                 base_channels=None,
                 act_cfg=(dict(type='ReLU'), dict(type='Sigmoid')),
                 init_cfg=None):
        super(SELayer, self).__init__(init_cfg)
        if isinstance(act_cfg, dict):
            act_cfg = (act_cfg, act_cfg)
        assert len(act_cfg) == 2
        assert mmcv.is_tuple_of(act_cfg, dict)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        if base_channels is None:
            squeeze_channels = make_divisible(channels // ratio, 8)
        else:
            assert isinstance(base_channels, int)
            assert base_channels > 0
            squeeze_channels = max(1, round(base_channels / ratio))
        self.conv1 = ConvModule(
            in_channels=channels,
            out_channels=squeeze_channels,
            kernel_size=1,
            stride=1,
            bias=bias,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[0])
        self.conv2 = ConvModule(
            in_channels=squeeze_channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            bias=bias,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[1])

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.conv1(out)
        out = self.conv2(out)
        return x * out
