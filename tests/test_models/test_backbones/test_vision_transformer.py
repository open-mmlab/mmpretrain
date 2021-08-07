import pytest
import torch
from torch.nn.modules import GroupNorm
from torch.nn.modules.batchnorm import _BatchNorm

from mmcls.models.backbones import VGG, VisionTransformer


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


def test_vit_backbone():
    with pytest.raises(TypeError):
        # pretrained must be a string path
        model = VisionTransformer()
        model.init_weights(pretrained=0)

    # Test ViT base model with input size of 224
    # and patch size of 16
    model = VisionTransformer()
    model.init_weights()
    model.train()

    assert check_norm_state(model.modules(), True)

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert feat.shape == torch.Size((1, 768))


def test_vit_hybrid_backbone():

    # Test VGG11+ViT-B/16 hybrid model
    backbone = VGG(11, norm_eval=True)
    backbone.init_weights()
    model = VisionTransformer(hybrid_backbone=backbone)
    model.init_weights()
    model.train()

    assert check_norm_state(model.modules(), True)

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert feat.shape == torch.Size((1, 768))
