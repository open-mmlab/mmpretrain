# Copyright (c) OpenMMLab. All rights reserved.

import pytest
import torch
from torch.nn.modules import GroupNorm
from torch.nn.modules.batchnorm import _BatchNorm

from mmcls.models.backbones import EfficientNet


def is_norm(modules):
    """Check if is one of the norms."""
    if isinstance(modules, (GroupNorm, _BatchNorm)):
        return True
    return False


def check_norm_state(modules, train_state):
    """Check if norm layer is in correct train state."""
    for mod in modules:
        if isinstance(mod, _BatchNorm):
            if mod.training != train_state:
                return False
    return True


def test_efficientnet():

    with pytest.raises(AssertionError):
        EfficientNet(arch='bad')

    model = EfficientNet(arch='b0')
    model.init_weights()
    model.train()

    assert check_norm_state(model.modules(), True)

    imgs = torch.randn(3, 3, 224, 224)
    feat = model(imgs)[-1]
    assert feat.shape == (3, 1280, 7, 7)
