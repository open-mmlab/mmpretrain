# Copyright (c) OpenMMLab. All rights reserved.
from .cls_head import ClsHead
from .conformer_head import ConformerHead
from .deit_head import DeiTClsHead
from .efficientformer_head import EfficientFormerClsHead
from .linear_head import LinearClsHead
from .margin_head import ArcFaceClsHead
from .multi_label_cls_head import MultiLabelClsHead
from .multi_label_csra_head import CSRAClsHead
from .multi_label_linear_head import MultiLabelLinearClsHead
from .stacked_head import StackedLinearClsHead
from .vision_transformer_head import VisionTransformerClsHead

__all__ = [
    'ClsHead', 'LinearClsHead', 'StackedLinearClsHead', 'MultiLabelClsHead',
    'MultiLabelLinearClsHead', 'VisionTransformerClsHead', 'DeiTClsHead',
    'ConformerHead', 'EfficientFormerClsHead', 'ArcFaceClsHead', 'CSRAClsHead'
]
