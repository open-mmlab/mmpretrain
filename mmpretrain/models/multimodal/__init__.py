# Copyright (c) OpenMMLab. All rights reserved.
from mmpretrain.utils.dependency import WITH_MULTIMODAL

if WITH_MULTIMODAL:
    from .blip import *  # noqa: F401,F403
    from .blip2 import *  # noqa: F401,F403
    from .flamingo import *  # noqa: F401, F403
    from .ofa import *  # noqa: F401, F403
else:
    from mmpretrain.registry import MODELS
    from mmpretrain.utils.dependency import register_multimodal_placeholder

    register_multimodal_placeholder([
        'BLIP2Captioner', 'BLIP2Retriever', 'BLIP2VQAModel', 'BLIPCaptioner',
        'BLIPNLVR', 'BLIPRetriever', 'BlipGroundingModel', 'BlipVQAModel',
        'Flamingo', 'OFA'
    ], MODELS)
