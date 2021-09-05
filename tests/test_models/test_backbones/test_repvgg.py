import pytest
import torch
from torch import nn
from torch.nn.modules import GroupNorm
from torch.nn.modules.batchnorm import _BatchNorm

from mmcls.models.backbones import RepVGG
from mmcls.models.backbones.repvgg import RepVGGBlock
from mmcls.models.utils import SELayer


def check_norm_state(modules, train_state):
    """Check if norm layer is in correct train state."""
    for mod in modules:
        if isinstance(mod, _BatchNorm):
            if mod.training != train_state:
                return False
    return True


def is_norm(modules):
    """Check if is one of the norms."""
    if isinstance(modules, (GroupNorm, _BatchNorm)):
        return True
    return False


def is_repvgg_block(modules):
    if isinstance(modules, RepVGGBlock):
        return True
    return False


def test_repvgg_repvggblock():
    # Test RepVGGBlock with in_channels != out_channels, stride = 1
    block = RepVGGBlock(48, 96, stride=1)
    x = torch.randn(1, 48, 56, 56)
    x_out = block(x)
    assert block.branch_norm is None
    assert block.se_cfg is None
    assert x_out.shape == torch.Size((1, 96, 56, 56))
    block.switch_to_deploy()
    block.deploy = True
    x_out = block(x)
    assert x_out.shape == torch.Size((1, 96, 56, 56))

    # Test RepVGGBlock with in_channels == out_channels, stride = 1
    block = RepVGGBlock(48, 48, stride=1)
    x = torch.randn(1, 48, 56, 56)
    x_out = block(x)
    assert isinstance(block.branch_norm, nn.BatchNorm2d)
    assert x_out.shape == torch.Size((1, 48, 56, 56))
    block.switch_to_deploy()
    block.deploy = True
    x_out = block(x)
    assert x_out.shape == torch.Size((1, 48, 56, 56))

    # Test RepVGGBlock with in_channels == out_channels, stride = 2
    block = RepVGGBlock(48, 48, stride=2)
    x = torch.randn(1, 48, 56, 56)
    x_out = block(x)
    assert block.branch_norm is None
    assert x_out.shape == torch.Size((1, 48, 28, 28))

    # Test RepVGGBlock with padding == dilation == 2
    block = RepVGGBlock(48, 48, stride=1, padding=2, dilation=2)
    x = torch.randn(1, 48, 56, 56)
    x_out = block(x)
    assert isinstance(block.branch_norm, nn.BatchNorm2d)
    assert x_out.shape == torch.Size((1, 48, 56, 56))

    # Test RepVGGBlock with groups = 2
    block = RepVGGBlock(48, 48, stride=1, groups=2)
    x = torch.randn(1, 48, 56, 56)
    x_out = block(x)
    assert block.branch_3x3.conv.weight.data.shape == torch.Size(
        (48, 24, 3, 3))
    assert x_out.shape == torch.Size((1, 48, 56, 56))

    # Test RepVGGBlock with se
    se_cfg = dict(ratio=10, divisor=1)
    block = RepVGGBlock(40, 40, stride=1, se_cfg=se_cfg)
    x = torch.randn(1, 40, 56, 56)
    x_out = block(x)
    assert isinstance(block.se_layer, SELayer)
    assert x_out.shape == torch.Size((1, 40, 56, 56))
    assert block.se_layer.conv1.conv.weight.data.shape == torch.Size(
        (4, 40, 1, 1))

    # Test RepVGGBlock with checkpoint forward
    block = RepVGGBlock(48, 48, stride=1, with_cp=True)
    assert block.with_cp
    x = torch.randn(1, 48, 56, 56)
    x_out = block(x)
    assert x_out.shape == torch.Size((1, 48, 56, 56))

    # Test RepVGGBlock with deploy == True
    block = RepVGGBlock(48, 48, stride=1, deploy=True)
    assert isinstance(block.branch_reparam, nn.Conv2d)
    x = torch.randn(1, 48, 56, 56)
    x_out = block(x)
    assert x_out.shape == torch.Size((1, 48, 56, 56))


def test_repvgg_backbone():
    with pytest.raises(TypeError):
        # arch must be str or dict
        RepVGG(arch=[4, 6, 16, 1])

    with pytest.raises(AssertionError):
        # arch must in arch_settings
        RepVGG(arch='A3')

    with pytest.raises(KeyError):
        # arch must have num_blocks and width_factor
        arch = dict(num_blocks=[2, 4, 14, 1])
        RepVGG(arch=arch)

    # len(arch['num_blocks']) == len(arch['width_factor'])
    # == len(strides) == len(dilations)
    with pytest.raises(AssertionError):
        arch = dict(num_blocks=[2, 4, 14, 1], width_factor=[0.75, 0.75, 0.75])
        RepVGG(arch=arch)

    with pytest.raises(AssertionError):
        RepVGG('A0', strides=(1, 1, 1))

    with pytest.raises(AssertionError):
        RepVGG('A0', strides=(1, 1, 1, 1), dilations=(1, 1, 2))

    # max(out_indices) < len(arch['num_blocks'])
    with pytest.raises(AssertionError):
        RepVGG('A0', out_indices=(5, ))

    # max(arch['group_idx'].keys()) <= sum(arch['num_blocks'])
    with pytest.raises(AssertionError):
        arch = dict(
            num_blocks=[2, 4, 14, 1],
            width_factor=[0.75, 0.75, 0.75],
            group_idx={22: 2})
        RepVGG(arch=arch)

    # Test RepVGG norm state
    model = RepVGG('A0')
    model.init_weights()
    model.train()
    assert check_norm_state(model.modules(), True)

    # Test RepVGG with first stage frozen
    frozen_stages = 1
    model = RepVGG('A0', frozen_stages=frozen_stages)
    model.init_weights()
    model.train()
    for param in model.stage_0.parameters():
        assert param.requires_grad is False
    for i in range(0, frozen_stages):
        stage_name = model.stages[i]
        stage = model.__getattr__(stage_name)
        for mod in stage:
            if isinstance(mod, _BatchNorm):
                assert mod.training is False
        for param in stage.parameters():
            assert param.requires_grad is False

    # Test RepVGG with norm_eval
    model = RepVGG('A0', norm_eval=True)
    model.init_weights()
    model.train()
    assert check_norm_state(model.modules(), False)

    # Test RepVGG forward with arch='A0'
    model = RepVGG('A0', out_indices=(0, 1, 2, 3))
    model.init_weights()
    model.train()

    for m in model.modules():
        if is_norm(m):
            assert isinstance(m, _BatchNorm)

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == torch.Size((1, 48, 56, 56))
    assert feat[1].shape == torch.Size((1, 96, 28, 28))
    assert feat[2].shape == torch.Size((1, 192, 14, 14))
    assert feat[3].shape == torch.Size((1, 1280, 7, 7))

    # Test RepVGG forward
    model_test_settings = [
        dict(model_name='A0', out_sizes=(48, 96, 192, 1280)),
        dict(model_name='A1', out_sizes=(64, 128, 256, 1280)),
        dict(model_name='A2', out_sizes=(96, 192, 384, 1408)),
        dict(model_name='B0', out_sizes=(64, 128, 256, 1280)),
        dict(model_name='B1', out_sizes=(128, 256, 512, 2048)),
        dict(model_name='B1g2', out_sizes=(128, 256, 512, 2048)),
        dict(model_name='B1g4', out_sizes=(128, 256, 512, 2048)),
        dict(model_name='B2', out_sizes=(160, 320, 640, 2560)),
        dict(model_name='B2g2', out_sizes=(160, 320, 640, 2560)),
        dict(model_name='B2g4', out_sizes=(160, 320, 640, 2560)),
        dict(model_name='B3', out_sizes=(192, 384, 768, 2560)),
        dict(model_name='B3g2', out_sizes=(192, 384, 768, 2560)),
        dict(model_name='B3g4', out_sizes=(192, 384, 768, 2560)),
    ]

    # Test RepVGG model
    for model_test_setting in model_test_settings:
        model = RepVGG(
            model_test_setting['model_name'], out_indices=(0, 1, 2, 3))
        model.init_weights()
        model.train()

        for m in model.modules():
            if is_norm(m):
                assert isinstance(m, _BatchNorm)

        imgs = torch.randn(1, 3, 224, 224)
        feat = model(imgs)
        assert feat[0].shape == torch.Size(
            (1, model_test_setting['out_sizes'][0], 56, 56))
        assert feat[1].shape == torch.Size(
            (1, model_test_setting['out_sizes'][1], 28, 28))
        assert feat[2].shape == torch.Size(
            (1, model_test_setting['out_sizes'][2], 14, 14))
        assert feat[3].shape == torch.Size(
            (1, model_test_setting['out_sizes'][3], 7, 7))

        # Test RepVGG forward with arch='D2se'
        se_cfg = dict(ratio=10, divisor=1)
        model = RepVGG('D2se', out_indices=(0, 1, 2, 3), se_cfg=se_cfg)
        model.init_weights()
        model.train()

        for m in model.modules():
            if is_norm(m):
                assert isinstance(m, _BatchNorm)

        imgs = torch.randn(1, 3, 224, 224)
        feat = model(imgs)
        assert len(feat) == 4
        assert feat[0].shape == torch.Size((1, 160, 56, 56))
        assert feat[1].shape == torch.Size((1, 320, 28, 28))
        assert feat[2].shape == torch.Size((1, 640, 14, 14))
        assert feat[3].shape == torch.Size((1, 2560, 7, 7))

    # Test RepVGG forward with layer 3 forward
    model = RepVGG('A0', out_indices=(3, ))
    model.init_weights()
    model.train()

    for m in model.modules():
        if is_norm(m):
            assert isinstance(m, _BatchNorm)

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert isinstance(feat, torch.Tensor)
    assert feat.shape == torch.Size((1, 1280, 7, 7))

    # Test RepVGG forward with layer 1 2 forward
    model = RepVGG('A0', out_indices=(1, 2))
    model.init_weights()
    model.train()

    for m in model.modules():
        if is_norm(m):
            assert isinstance(m, _BatchNorm)

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 2
    assert feat[0].shape == torch.Size((1, 96, 28, 28))
    assert feat[1].shape == torch.Size((1, 192, 14, 14))

    # Test RepVGG with deploy=True
    model = RepVGG('A0', out_indices=(1, 2), deploy=True)
    model.init_weights()
    model.train()

    for m in model.modules():
        if is_norm(m):
            assert isinstance(m, _BatchNorm)
        if is_repvgg_block(m):
            assert isinstance(m.branch_reparam, nn.Conv2d)

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 2
    assert feat[0].shape == torch.Size((1, 96, 28, 28))
    assert feat[1].shape == torch.Size((1, 192, 14, 14))
