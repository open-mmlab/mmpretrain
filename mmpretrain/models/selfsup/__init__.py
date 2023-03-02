# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseSelfSupervisor
from .beit import VQKD, BEiT, BEiTPretrainViT
from .byol import BYOL
from .cae import CAE, CAEPretrainViT, Encoder
from .eva import EVA
from .mae import MAE, MAEViT
from .maskfeat import HOGGenerator, MaskFeat, MaskFeatViT
from .milan import MILAN, CLIPGenerator, MILANViT
from .mixmim import MixMIM, MixMIMPretrainTransformer
from .moco import MoCo
from .mocov3 import MoCoV3, MoCoV3ViT
from .simclr import SimCLR
from .simmim import SimMIM, SimMIMSwinTransformer
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
    'MILAN',
    'MixMIM',
    'SimMIM',
    'EVA',
]
