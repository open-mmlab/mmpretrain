# Copyright (c) OpenMMLab. All rights reserved.

import torch
from torch.nn.modules.batchnorm import _BatchNorm

from mmcls.models.backbones import DistilledVisionTransformer


def check_norm_state(modules, train_state):
    """Check if norm layer is in correct train state."""
    for mod in modules:
        if isinstance(mod, _BatchNorm):
            if mod.training != train_state:
                return False
    return True


def test_deit_backbone():
    cfg_ori = dict(arch='deit-b', img_size=224, patch_size=16)

    # Test structure
    model = DistilledVisionTransformer(**cfg_ori)
    model.init_weights()
    model.train()

    assert check_norm_state(model.modules(), True)
    assert model.dist_token.shape == (1, 1, 768)
    assert model.pos_embed.shape == (1, model.patch_embed.num_patches + 2, 768)

    # Test forward
    imgs = torch.rand(1, 3, 224, 224)
    outs = model(imgs)
    patch_token, cls_token, dist_token = outs[0]
    assert patch_token.shape == (1, 768, 14, 14)
    assert cls_token.shape == (1, 768)
    assert dist_token.shape == (1, 768)

    # Test multiple out_indices
    model = DistilledVisionTransformer(
        **cfg_ori, out_indices=(0, 1, 2, 3), output_cls_token=False)
    outs = model(imgs)
    for out in outs:
        assert out.shape == (1, 768, 14, 14)
