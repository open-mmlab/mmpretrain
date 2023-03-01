# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseSelfSupervisor
from .beit import VQKD, BEiTPretrainViT, BEiT
from .byol import BYOL
from .cae import CAEPretrainViT, Encoder, CAE
from .mae import MAEViT, MAE
from .maskfeat import HOGGenerator, MaskFeatViT, MaskFeat
from .milan import CLIPGenerator, MILANViT
from .mixmim import MixMIMPretrainTransformer
from .moco import MoCo
from .mocov3 import MoCoV3ViT, MoCoV3
from .simclr import SimCLR
from .simmim import SimMIMSwinTransformer
from .simsiam import SimSiam

__all__ = [
    'BaseSelfSupervisor',
    'BEiTPretrainViT',
    'VQKD',
    'CAEPretrainViT',
    'Encoder',
    'MAEViT',
    'HOGGenerator',
    'MaskFeatViT',
    'CLIPGenerator',
    'MILANViT',
    'MixMIMPretrainTransformer',
    'MoCoV3ViT',
    'SimMIMSwinTransformer',
    'MoCo',
    'MoCoV3',
    'BYOL',
    'SimCLR',
    'SimSiam',
    'BEiT',
    'CAE',
    'MAE',
    'MaskFeat',
]
