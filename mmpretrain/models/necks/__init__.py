# Copyright (c) OpenMMLab. All rights reserved.
from .gap import GlobalAveragePooling
from .gem import GeneralizedMeanPooling
from .hr_fuse import HRFuseScales
from .linear_neck import LinearNeck
from .beitv2_neck import BEiTV2Neck
from .cae_neck import CAENeck
from .densecl_neck import DenseCLNeck
from .mae_neck import MAEPretrainDecoder
from .milan_neck import MILANPretrainDecoder
from .mixmim_neck import MixMIMPretrainDecoder
from .mocov2_neck import MoCoV2Neck
from .nonlinear_neck import NonLinearNeck
from .simmim_neck import SimMIMLinearDecoder

__all__ = [
    'GlobalAveragePooling',
    'GeneralizedMeanPooling',
    'HRFuseScales',
    'LinearNeck',
    'BEiTV2Neck',
    'CAENeck',
    'DenseCLNeck',
    'MAEPretrainDecoder',
    'MILANPretrainDecoder',
    'MixMIMPretrainDecoder',
    'MoCoV2Neck',
    'NonLinearNeck',
    'SimMIMLinearDecoder',
]
