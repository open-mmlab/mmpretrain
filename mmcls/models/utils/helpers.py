# Copyright (c) OpenMMLab. All rights reserved.
import collections.abc
import warnings
from distutils.version import LooseVersion
from itertools import repeat

import torch


def is_tracing() -> bool:
    if LooseVersion(torch.__version__) >= LooseVersion('1.6.0'):
        on_trace = torch.jit.is_tracing()
        # In PyTorch 1.6, torch.jit.is_tracing has a bug.
        # Refers to https://github.com/pytorch/pytorch/issues/42448
        if isinstance(on_trace, bool):
            return on_trace
        else:
            return torch._C._is_tracing()
    else:
        warnings.warn(
            'torch.jit.is_tracing is only supported after v1.6.0. '
            'Therefore is_tracing returns False automatically. Please '
            'set on_trace manually if you are using trace.', UserWarning)
        return False


# From PyTorch internals
def _ntuple(n):

    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple
