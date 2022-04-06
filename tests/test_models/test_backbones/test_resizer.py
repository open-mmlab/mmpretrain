# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import Tensor

from mmcls.models.backbones import LearnableResizer


def test_learnableresizer():
    imgs = torch.randn(16, 3, 48, 48)

    preprocessors = LearnableResizer(output_size=(32, 32))
    imgs = preprocessors(imgs)
    assert isinstance(imgs, Tensor) and imgs.shape == (16, 3, 32, 32)

    single_img = torch.randn(1, 3, 32, 32)
    single_img = preprocessors(single_img)
    assert isinstance(single_img,
                      Tensor) and single_img.shape == (1, 3, 32, 32)
