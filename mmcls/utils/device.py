# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch
from mmcv.utils import digit_version


def auto_select_device() -> str:
    mmcv_version = digit_version(mmcv.__version__)
    if mmcv_version >= digit_version('1.6.0'):
        from mmcv.device import get_device
        return get_device()
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'
