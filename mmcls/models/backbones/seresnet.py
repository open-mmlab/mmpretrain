import torch.utils.checkpoint as cp

from ..builder import BACKBONES
from ..utils.se_layer import SELayer
from .resnet import Bottleneck, ResLayer, ResNet


class SEBottleneck(Bottleneck):
    """SEBottleneck block for SEResNet.

    Args:
        inplanes (int): The input channels of the SEBottleneck block.
        planes (int): The output channel base of the SEBottleneck block.
        se_ratio (int): Squeeze ratio in SELayer. Default: 16
    """
    expansion = 4

    def __init__(self, inplanes, planes, se_ratio=16, **kwargs):
        super(SEBottleneck, self).__init__(inplanes, planes, **kwargs)
        self.se_layer = SELayer(planes * self.expansion, ratio=se_ratio)

    def forward(self, x):

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.norm3(out)

            out = self.se_layer(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


@BACKBONES.register_module()
class SEResNet(ResNet):
    """SEResNet backbone.

    Args:
        depth (int): Depth of seresnet, from {50, 101, 152}.
        in_channels (int): Number of input image channels. Normally 3.
        base_channels (int): Number of base channels of hidden layer.
        num_stages (int): Resnet stages, normally 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        se_ratio (int): Squeeze ratio in SELayer. Default: 16
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        norm_cfg (dict): Dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity.

    Example:
        >>> from mmcls.models import SEResNet
        >>> import torch
        >>> self = SEResNet(depth=50)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 224, 224)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 64, 56, 56)
        (1, 128, 28, 28)
        (1, 256, 14, 14)
        (1, 512, 7, 7)
    """

    arch_settings = {
        50: (SEBottleneck, (3, 4, 6, 3)),
        101: (SEBottleneck, (3, 4, 23, 3)),
        152: (SEBottleneck, (3, 8, 36, 3))
    }

    def __init__(self, depth, se_ratio=16, **kwargs):
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
        self.se_ratio = se_ratio
        super(SEResNet, self).__init__(depth, **kwargs)

    def make_res_layer(self, **kwargs):
        return ResLayer(se_ratio=self.se_ratio, **kwargs)
