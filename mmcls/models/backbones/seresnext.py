# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import BACKBONES
from .resnet import ResLayer
from .resnext import Bottleneck
from .seresnet import SEBottleneck as _SEBottleneck
from .seresnet import SEResNet


class SEBottleneck(Bottleneck, _SEBottleneck):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(SEBottleneck, self).__init__(in_channels, out_channels, **kwargs)


@BACKBONES.register_module()
class SEResNeXt(SEResNet):
    """SEResNeXt backbone.

    Please refer to the `paper <https://arxiv.org/abs/1709.01507>`__ for
    details.

    Args:
        depth (int): Network depth, from {50, 101, 152}.
        groups (int): Groups of conv2 in Bottleneck. Default: 32.
        width_per_group (int): Width per group of conv2 in Bottleneck.
            Default: 4.
        se_ratio (int): Squeeze ratio in SELayer. Default: 16.
        in_channels (int): Number of input image channels. Default: 3.
        stem_channels (int): Output channels of the stem layer. Default: 64.
        num_stages (int): Stages of the network. Default: 4.
        strides (Sequence[int]): Strides of the first block of each stage.
            Default: ``(1, 2, 2, 2)``.
        dilations (Sequence[int]): Dilation of each stage.
            Default: ``(1, 1, 1, 1)``.
        out_indices (Sequence[int]): Output from which stages. If only one
            stage is specified, a single tensor (feature map) is returned,
            otherwise multiple stages are specified, a tuple of tensors will
            be returned. Default: ``(3, )``.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv.
            Default: False.
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Default: -1.
        conv_cfg (dict | None): The config dict for conv layers. Default: None.
        norm_cfg (dict): The config dict for norm layers.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity. Default: True.
    """

    arch_settings = {
        50: (SEBottleneck, (3, 4, 6, 3)),
        101: (SEBottleneck, (3, 4, 23, 3)),
        152: (SEBottleneck, (3, 8, 36, 3))
    }

    def __init__(self, depth, groups=32, width_per_group=4, **kwargs):
        self.groups = groups
        self.width_per_group = width_per_group
        super(SEResNeXt, self).__init__(depth, **kwargs)

    def make_res_layer(self, **kwargs):
        return ResLayer(
            groups=self.groups,
            width_per_group=self.width_per_group,
            base_channels=self.base_channels,
            **kwargs)
