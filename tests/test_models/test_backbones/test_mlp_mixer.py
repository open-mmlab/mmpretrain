import torch
from torch.nn.modules.batchnorm import _BatchNorm

from mmcls.models.backbones import MlpMixer


def check_norm_state(modules, train_state):
    """Check if norm layer is in correct train state."""
    for mod in modules:
        if isinstance(mod, _BatchNorm):
            if mod.training != train_state:
                return False
    return True


def test_mlp_mixer_backbone():
    cfg_ori = dict(
        patch_size=16, num_blocks=12, embed_dims=768, out_indices=(-1, -2))
    model = MlpMixer(**cfg_ori)
    model.train()
    assert check_norm_state(model.modules(), True)
    imgs = torch.randn(3, 3, 224, 224)
    out = model(imgs)
    assert isinstance(out, tuple) and len(out) == 2
    assert out[-1].size(0) == 3 and out[-1].size(1) == 768
    assert out[0].size(0) == 3 and out[0].size(1) == 196 \
           and out[0].size(2) == 768
