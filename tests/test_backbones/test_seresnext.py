import pytest
import torch

from mmcls.models.backbones import SEResNeXt
from mmcls.models.backbones.seresnext import SEBottleneck as SEBottleneckX


def is_block(modules):
    """Check if is SEResNeXt building block."""
    if isinstance(modules, (SEBottleneckX)):
        return True
    return False


def test_seresnext_bottleneck():
    with pytest.raises(AssertionError):
        # Style must be in ['pytorch', 'caffe']
        SEBottleneckX(64, 64, groups=32, base_width=4, style='tensorflow')

    # Test SEResNeXt Bottleneck structure
    block = SEBottleneckX(
        64, 64, groups=32, base_width=4, stride=2, style='pytorch')
    assert block.conv2.stride == (2, 2)
    assert block.conv2.groups == 32
    assert block.conv2.out_channels == 128

    # Test SEResNeXt Bottleneck forward
    block = SEBottleneckX(64, 16, groups=32, base_width=4)
    x = torch.randn(1, 64, 56, 56)
    x_out = block(x)
    assert x_out.shape == torch.Size([1, 64, 56, 56])


def test_seresnext_backbone():
    with pytest.raises(KeyError):
        # SEResNeXt depth should be in [50, 101, 152]
        SEResNeXt(depth=18)

    # Test SEResNeXt with group 32, base_width 4
    model = SEResNeXt(
        depth=50, groups=32, base_width=4, out_indices=(0, 1, 2, 3))
    for m in model.modules():
        if is_block(m):
            assert m.conv2.groups == 32
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == torch.Size([1, 256, 56, 56])
    assert feat[1].shape == torch.Size([1, 512, 28, 28])
    assert feat[2].shape == torch.Size([1, 1024, 14, 14])
    assert feat[3].shape == torch.Size([1, 2048, 7, 7])

    # Test SEResNeXt with group 32, base_width 4 and layers 3 out forward
    model = SEResNeXt(depth=50, groups=32, base_width=4, out_indices=(3, ))
    for m in model.modules():
        if is_block(m):
            assert m.conv2.groups == 32
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert feat.shape == torch.Size([1, 2048, 7, 7])
