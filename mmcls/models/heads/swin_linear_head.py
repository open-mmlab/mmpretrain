import torch.nn as nn

from ..builder import HEADS
from .linear_head import LinearClsHead


@HEADS.register_module()
class SwinLinearClsHead(LinearClsHead):
    """Swin Linear classifier head.

    Compare with LinearClsHead, init weight method is modified.
    TODO: Use LinearClsHead with init_cfg to replace this class.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        loss (dict): Config of classification loss.
    """  # noqa: W605

    def __init__(self, *args, **kwargs):
        super(SwinLinearClsHead, self).__init__(*args, **kwargs)

    def init_weights(self):
        nn.init.trunc_normal_(self.fc.weight, std=0.02)
        nn.init.constant_(self.fc.bias, 0.)
