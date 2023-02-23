# Copyright (c) OpenMMLab. All rights reserved.
from .cls_head import ClsHead
from .conformer_head import ConformerHead
from .deit_head import DeiTClsHead
from .efficientformer_head import EfficientFormerClsHead
from .levit_head import LeViTClsHead
from .linear_head import LinearClsHead
from .margin_head import ArcFaceClsHead
from .multi_label_cls_head import MultiLabelClsHead
from .multi_label_csra_head import CSRAClsHead
from .multi_label_linear_head import MultiLabelLinearClsHead
from .multi_task_head import MultiTaskHead
from .stacked_head import StackedLinearClsHead
from .vig_head import VigClsHead
from .vision_transformer_head import VisionTransformerClsHead
from .beitv1_head import BEiTV1Head
from .beitv2_head import BEiTV2Head
from .cae_head import CAEHead
from .contrastive_head import ContrastiveHead
from .latent_heads import LatentCrossCorrelationHead, LatentPredictHead
from .mae_head import MAEPretrainHead
from .mixmim_head import MixMIMPretrainHead
from .naive_mim_head import NaiveMIMHead
from .swav_head import SwAVHead

__all__ = [
    'ClsHead',
    'LinearClsHead',
    'StackedLinearClsHead',
    'MultiLabelClsHead',
    'MultiLabelLinearClsHead',
    'VisionTransformerClsHead',
    'DeiTClsHead',
    'ConformerHead',
    'EfficientFormerClsHead',
    'ArcFaceClsHead',
    'CSRAClsHead',
    'MultiTaskHead',
    'LeViTClsHead',
    'VigClsHead',
    'BEiTV1Head',
    'BEiTV2Head',
    'CAEHead',
    'ContrastiveHead',
    'LatentCrossCorrelationHead',
    'LatentPredictHead',
    'MAEPretrainHead',
    'MixMIMPretrainHead',
    'NaiveMIMHead',
    'SwAVHead',
]
