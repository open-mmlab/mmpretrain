import pytest
import torch
from torch.nn.modules import GroupNorm
from torch.nn.modules.batchnorm import _BatchNorm

from mmcls.models.utils import SELayer


def is_norm(modules):
    """Check if is one of the norms."""
    if isinstance(modules, (GroupNorm, _BatchNorm)):
        return True
    return False


def test_se():
    with pytest.raises(AssertionError):
        # base_channels must be a number
        SELayer(16, base_channels='32')

    with pytest.raises(AssertionError):
        # base_channels must be None or a number larger than 0
        SELayer(16, base_channels=-1)

    with pytest.raises(AssertionError):
        # act_cfg must be two dict tuple
        SELayer(
            16,
            act_cfg=(dict(type='ReLU'), dict(type='Sigmoid'),
                     dict(type='ReLU')))

    input = torch.randn((4, 64, 112, 112))
    se = SELayer(64)
    output = se(input)
    assert se.conv1.out_channels == 8
    assert se.conv2.in_channels == 8
    assert output.shape == torch.Size((4, 64, 112, 112))

    input = torch.randn((4, 128, 112, 112))
    se = SELayer(128, ratio=4)
    output = se(input)
    assert se.conv1.out_channels == 32
    assert se.conv2.in_channels == 32
    assert output.shape == torch.Size((4, 128, 112, 112))

    input = torch.randn((1, 54, 76, 103))
    se = SELayer(54, ratio=4)
    output = se(input)
    assert se.conv1.out_channels == 16
    assert se.conv2.in_channels == 16
    assert output.shape == torch.Size((1, 54, 76, 103))

    input = torch.randn((4, 128, 56, 56))
    se = SELayer(
        128,
        base_channels=100,
        ratio=4,
        act_cfg=(dict(type='ReLU'), dict(type='HSigmoid')))
    output = se(input)
    assert se.conv1.out_channels == 25
    assert se.conv2.in_channels == 25
    assert output.shape == torch.Size((4, 128, 56, 56))
