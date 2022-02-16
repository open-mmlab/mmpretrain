# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmcls.models.utils import ConditionalPositionEncoding


def test_conditional_position_encoding_module():
    CPE = ConditionalPositionEncoding(in_channels=32, embed_dims=32, stride=2)
    outs = CPE(torch.randn(1, 3136, 32), (56, 56))
    assert outs.shape == torch.Size([1, 784, 32])
