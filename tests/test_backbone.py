import pytest
import torch
import torch.nn as nn
from torch.nn.modules import AvgPool2d, GroupNorm
from torch.nn.modules.batchnorm import _BatchNorm

from mmcls.models.backbones import MobileNetv2


def test_mobilenetv2_backbone():
    # Test MobileNetv2 with widen_factor 1.0, activation nn.ReLU6
    model = MobileNetv2(widen_factor=1.0, activation=nn.ReLU6)
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 8
    assert feat[0].shape == torch.Size([1, 16, 112, 112])
    assert feat[1].shape == torch.Size([1, 24, 56, 56])
    assert feat[2].shape == torch.Size([1, 32, 28, 28])
    assert feat[3].shape == torch.Size([1, 64, 14, 14])
    assert feat[4].shape == torch.Size([1, 96, 14, 14])
    assert feat[5].shape == torch.Size([1, 160, 7, 7])
    assert feat[6].shape == torch.Size([1, 320, 7, 7])
