# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch

from mmpretrain.models import CAEViT

backbone = dict(arch='b', patch_size=16, layer_scale_init_value=0.1)


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_cae_vit():
    cae_backbone = CAEViT(**backbone)
    cae_backbone.init_weights()
    fake_inputs = torch.randn((2, 3, 224, 224))
    fake_mask = torch.zeros((2, 196)).bool()
    fake_mask[:, 75:150] = 1
    fake_outputs = cae_backbone(fake_inputs, fake_mask)

    assert list(fake_outputs.shape) == [2, 122, 768]
