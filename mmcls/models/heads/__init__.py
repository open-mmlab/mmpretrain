# Copyright (c) OpenMMLab. All rights reserved.
from .cls_head import ClsHead
from .conformer_head import ConformerHead
from .deit_head import DeiTClsHead
from .linear_head import LinearClsHead
from .conv_head import ConvClsHead
from .multi_label_head import MultiLabelClsHead
from .multi_label_linear_head import MultiLabelLinearClsHead
from .stacked_head import StackedLinearClsHead
from .stackedconv_head import StackedConvClsHead
from .vision_transformer_head import VisionTransformerClsHead

__all__ = [
    "ClsHead",
    "LinearClsHead",
    "StackedLinearClsHead",
    "ConvClsHead",
    "StackedConvClsHead",
    "MultiLabelClsHead",
    "MultiLabelLinearClsHead",
    "VisionTransformerClsHead",
    "DeiTClsHead",
    "ConformerHead",
]
