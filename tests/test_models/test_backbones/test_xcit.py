# Copyright (c) OpenMMLab. All rights reserved.
# The basic forward/backward tests are in ../test_models.py
import torch

from mmpretrain.apis import get_model


def test_out_type():
    inputs = torch.rand(1, 3, 224, 224)

    model = get_model(
        'xcit-nano-12-p16_3rdparty_in1k',
        backbone=dict(out_type='raw'),
        neck=None,
        head=None)
    outputs = model(inputs)[0]
    assert outputs.shape == (1, 197, 128)

    model = get_model(
        'xcit-nano-12-p16_3rdparty_in1k',
        backbone=dict(out_type='featmap'),
        neck=None,
        head=None)
    outputs = model(inputs)[0]
    assert outputs.shape == (1, 128, 14, 14)

    model = get_model(
        'xcit-nano-12-p16_3rdparty_in1k',
        backbone=dict(out_type='cls_token'),
        neck=None,
        head=None)
    outputs = model(inputs)[0]
    assert outputs.shape == (1, 128)

    model = get_model(
        'xcit-nano-12-p16_3rdparty_in1k',
        backbone=dict(out_type='avg_featmap'),
        neck=None,
        head=None)
    outputs = model(inputs)[0]
    assert outputs.shape == (1, 128)
