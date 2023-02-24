# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch

from mmpretrain.models import SimMIMSwinTransformer

backbone = dict(
    arch='B', img_size=192, stage_cfgs=dict(block_cfgs=dict(window_size=6)))


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_cae_vit():
    simmim_backbone = SimMIMSwinTransformer(**backbone)
    simmim_backbone.init_weights()
    fake_inputs = torch.randn((2, 3, 192, 192))
    fake_mask = torch.rand((2, 48, 48))
    fake_outputs = simmim_backbone(fake_inputs, fake_mask)[0]

    assert list(fake_outputs.shape) == [2, 1024, 6, 6]
