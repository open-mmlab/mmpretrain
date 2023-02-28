# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch

from mmpretrain.models import BEiTPretrainViT

backbone = dict(
    arch='base',
    patch_size=16,
    drop_path_rate=0.1,
    final_norm=True,
    layer_scale_init_value=0.1,
)


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_beit_pretrain_vit():
    beit_backbone = BEiTPretrainViT(**backbone)
    beit_backbone.init_weights()

    fake_inputs = torch.randn((2, 3, 224, 224))
    fake_mask = torch.zeros((2, 196))
    fake_mask[:, 75:150] = 1
    fake_outputs = beit_backbone(fake_inputs, fake_mask)

    assert list(fake_outputs[0].shape) == [2, 197, 768]
