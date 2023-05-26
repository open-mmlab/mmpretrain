# Copyright (c) OpenMMLab. All rights reserved.
from dataclasses import dataclass

import pytest
import torch

import mmpretrain.models
from mmpretrain.apis import ModelHub, get_model


@dataclass
class Cfg:
    name: str
    backbone: type
    num_classes: int = 1000
    build: bool = True
    forward: bool = True
    backward: bool = True
    extract_feat: bool = True
    input_shape: tuple = (1, 3, 224, 224)


test_list = [
    Cfg(name='xcit-small-12-p16_3rdparty_in1k',
        backbone=mmpretrain.models.XCiT),
    Cfg(name='xcit-nano-12-p8_3rdparty-dist_in1k-384px',
        backbone=mmpretrain.models.XCiT,
        input_shape=(1, 3, 384, 384)),
    Cfg(name='vit-base-p16_sam-pre_3rdparty_sa1b-1024px',
        backbone=mmpretrain.models.ViTSAM,
        forward=False,
        backward=False),
    Cfg(name='vit-base-p14_dinov2-pre_3rdparty',
        backbone=mmpretrain.models.VisionTransformer,
        forward=False,
        backward=False),
    Cfg(name='hivit-tiny-p16_16xb64_in1k', backbone=mmpretrain.models.HiViT),
]


@pytest.mark.parametrize('cfg', test_list)
def test_build(cfg: Cfg):
    if not cfg.build:
        return

    model_name = cfg.name
    ModelHub._register_mmpretrain_models()
    assert ModelHub.has(model_name)

    model = get_model(model_name)
    backbone_class = cfg.backbone
    assert isinstance(model.backbone, backbone_class)


@pytest.mark.parametrize('cfg', test_list)
def test_forward(cfg: Cfg):
    if not cfg.forward:
        return

    model = get_model(cfg.name)
    inputs = torch.rand(*cfg.input_shape)
    outputs = model(inputs)
    assert outputs.shape == (1, cfg.num_classes)


@pytest.mark.parametrize('cfg', test_list)
def test_extract_feat(cfg: Cfg):
    if not cfg.extract_feat:
        return

    model = get_model(cfg.name)
    inputs = torch.rand(*cfg.input_shape)
    feats = model.extract_feat(inputs)
    assert isinstance(feats, tuple)
    assert len(feats) == 1


@pytest.mark.parametrize('cfg', test_list)
def test_backward(cfg: Cfg):
    if not cfg.backward:
        return

    model = get_model(cfg.name)
    inputs = torch.rand(*cfg.input_shape)
    outputs = model(inputs)
    outputs.mean().backward()

    for n, x in model.named_parameters():
        assert x.grad is not None, f'No gradient for {n}'
    num_grad = sum(
        [x.grad.numel() for x in model.parameters() if x.grad is not None])
    assert outputs.shape[-1] == cfg.num_classes
    num_params = sum([x.numel() for x in model.parameters()])
    assert num_params == num_grad, 'Some parameters are missing gradients'
    assert not torch.isnan(outputs).any(), 'Output included NaNs'
