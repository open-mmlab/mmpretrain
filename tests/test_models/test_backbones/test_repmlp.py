# Copyright (c) OpenMMLab. All rights reserved.
import os
import tempfile

import pytest
import torch
from mmcv.runner import load_checkpoint, save_checkpoint

from mmcls.models.backbones import RepMLPNet


def test_repmlp_backbone():
    with pytest.raises(AssertionError):
        # arch must be str or dict
        RepMLPNet(arch=[4, 6, 16, 1])

    with pytest.raises(AssertionError):
        # arch must in arch_settings
        RepMLPNet(arch='A3')

    with pytest.raises(AssertionError):
        # arch must have num_blocks and width_factor
        arch = dict(channels=[96, 192, 384, 768])
        RepMLPNet(arch=arch)

    # test len(arch['num_blocks']) equals to len(arch['channels'])
    #  equals to len(arch['sharesets_nums'])
    with pytest.raises(AssertionError):
        arch = dict(
            channels=[2, 4, 14, 1],
            num_blocks=[0.75, 0.75, 0.75],
            sharesets_nums=[1, 4, 32])
        RepMLPNet(arch=arch)

    # Test RepMLPNet forward with layer 3 forward
    input_size = 224
    model = RepMLPNet(
        'b',
        img_size=input_size,
        out_indices=(3, ),
        reparam_conv_kernels=(1, 3),
        final_norm=True)
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert isinstance(feat, tuple)
    assert len(feat) == 1
    assert isinstance(feat[0], torch.Tensor)
    assert feat[0].shape == torch.Size((1, 768, 7, 7))

    # Test RepMLPNet downstream forward
    model_test_settings = [
        dict(model_name='B', out_sizes=(96, 192, 384, 768)),
    ]

    # Test RepMLPNet model forward
    for model_test_setting in model_test_settings:
        model = RepMLPNet(
            model_test_setting['model_name'],
            out_indices=(0, 1, 2, 3),
            final_norm=False)
        model.init_weights()

        model.train()
        imgs = torch.randn(1, 3, 224, 224)
        feat = model(imgs)
        assert feat[0].shape == torch.Size(
            (1, model_test_setting['out_sizes'][1], 28, 28))
        assert feat[1].shape == torch.Size(
            (1, model_test_setting['out_sizes'][2], 14, 14))
        assert feat[2].shape == torch.Size(
            (1, model_test_setting['out_sizes'][3], 7, 7))
        assert feat[3].shape == torch.Size(
            (1, model_test_setting['out_sizes'][3], 7, 7))


def test_repmlp_deploy():
    # Test output before and load from deploy checkpoint
    ckpt_path = os.path.join(tempfile.gettempdir(), 'ckpt.pth')
    model = RepMLPNet(
        'b', out_indices=(
            1,
            3,
        ), reparam_conv_kernels=(1, 3, 5))
    imgs = torch.randn((1, 3, 224, 224))
    model.eval()
    feats = model(imgs)
    model.switch_to_deploy()
    for m in model.modules():
        if hasattr(m, 'deploy'):
            assert m.deploy is True
    feats_ = model(imgs)
    assert len(feats) == len(feats_)
    for i in range(len(feats)):
        torch.allclose(feats[i], feats_[i])

    model_deploy = RepMLPNet(
        'b', out_indices=(
            1,
            3,
        ), reparam_conv_kernels=(1, 3, 5), deploy=True)
    save_checkpoint(model, ckpt_path)
    load_checkpoint(model_deploy, ckpt_path, strict=True)
    feats_ = model_deploy(imgs)

    assert len(feats) == len(feats_)
    for i in range(len(feats)):
        torch.allclose(feats[i], feats_[i])
