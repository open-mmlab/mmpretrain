from .base import BaseSelfSupervisor
from .beit import VQKD, BEiTPretrainViT
from .cae import CAEViT, Encoder
from .mae import MAEViT
from .maskfeat import HOGGenerator, MaskFeatViT
from .milan import CLIPGenerator, MILANViT
from .mixmim import MixMIMPretrainTransformer
from .mocov3 import MoCoV3ViT
from .simmim import SimMIMSwinTransformer

__all__ = [
    'BaseSelfSupervisor',
    'BEiTPretrainViT',
    'VQKD',
    'CAEViT',
    'Encoder',
    'MAEViT',
    'HOGGenerator',
    'MaskFeatViT',
    'CLIPGenerator',
    'MILANViT',
    'MixMIMPretrainTransformer',
    'MoCoV3ViT',
    'SimMIMSwinTransformer',
]
