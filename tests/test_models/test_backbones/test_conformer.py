# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy

import pytest
import torch
from torch.nn.modules import GroupNorm
from torch.nn.modules.batchnorm import _BatchNorm

from mmcls.models.backbones import Conformer


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


def test_conformer_backbone():

    cfg_ori = dict(
        arch='T',
        drop_path_rate=0.1,
    )

    with pytest.raises(AssertionError):
        # test invalid arch
        cfg = deepcopy(cfg_ori)
        cfg['arch'] = 'unknown'
        Conformer(**cfg)

    with pytest.raises(AssertionError):
        # test arch without essential keys
        cfg = deepcopy(cfg_ori)
        cfg['arch'] = {'embed_dims': 24, 'channel_ratio': 6, 'num_heads': 9}
        Conformer(**cfg)

    # Test Conformer small model with patch size of 16
    model = Conformer(**cfg_ori)
    model.init_weights()
    model.train()

    assert check_norm_state(model.modules(), True)

    imgs = torch.randn(3, 3, 224, 224)
    conv_feature, transformer_feature = model(imgs)[-1]
    assert conv_feature.shape == (3, 64 * 1 * 4
                                  )  # base_channels * channel_ratio * 4
    assert transformer_feature.shape == (3, 384)

    # Test Conformer with irregular input size.
    model = Conformer(**cfg_ori)
    model.init_weights()
    model.train()

    assert check_norm_state(model.modules(), True)

    imgs = torch.randn(3, 3, 241, 241)
    conv_feature, transformer_feature = model(imgs)[-1]
    assert conv_feature.shape == (3, 64 * 1 * 4
                                  )  # base_channels * channel_ratio * 4
    assert transformer_feature.shape == (3, 384)

    imgs = torch.randn(3, 3, 321, 221)
    conv_feature, transformer_feature = model(imgs)[-1]
    assert conv_feature.shape == (3, 64 * 1 * 4
                                  )  # base_channels * channel_ratio * 4
    assert transformer_feature.shape == (3, 384)

    # Test custom arch Conformer without output cls token
    cfg = deepcopy(cfg_ori)
    cfg['arch'] = {
        'embed_dims': 128,
        'depths': 15,
        'num_heads': 16,
        'channel_ratio': 3,
    }
    cfg['with_cls_token'] = False
    cfg['base_channels'] = 32
    model = Conformer(**cfg)
    conv_feature, transformer_feature = model(imgs)[-1]
    assert conv_feature.shape == (3, 32 * 3 * 4)
    assert transformer_feature.shape == (3, 128)

    # Test Conformer with multi out indices
    cfg = deepcopy(cfg_ori)
    cfg['out_indices'] = [4, 8, 12]
    model = Conformer(**cfg)
    outs = model(imgs)
    assert len(outs) == 3
    # stage 1
    conv_feature, transformer_feature = outs[0]
    assert conv_feature.shape == (3, 64 * 1)
    assert transformer_feature.shape == (3, 384)
    # stage 2
    conv_feature, transformer_feature = outs[1]
    assert conv_feature.shape == (3, 64 * 1 * 2)
    assert transformer_feature.shape == (3, 384)
    # stage 3
    conv_feature, transformer_feature = outs[2]
    assert conv_feature.shape == (3, 64 * 1 * 4)
    assert transformer_feature.shape == (3, 384)
