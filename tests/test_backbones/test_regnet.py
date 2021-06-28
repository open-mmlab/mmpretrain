import pytest
import torch

from mmcls.models.backbones import RegNet

regnet_test_data = [
    ('regnetx_400mf',
     dict(
         w0=24,
         wa=24.48,
         wm=2.54,
         group_w=16,
         depth=22,
         bot_mul=1.0,
         has_se=False), [32, 64, 160, 384]),
    ('regnetx_800mf',
     dict(
         w0=56,
         wa=35.73,
         wm=2.28,
         group_w=16,
         depth=16,
         bot_mul=1.0,
         has_se=False), [64, 128, 288, 672]),
    ('regnetx_1.6gf',
     dict(
         w0=80,
         wa=34.01,
         wm=2.25,
         group_w=24,
         depth=18,
         bot_mul=1.0,
         has_se=False), [72, 168, 408, 912]),
    ('regnetx_3.2gf',
     dict(
         w0=88,
         wa=26.31,
         wm=2.25,
         group_w=48,
         depth=25,
         bot_mul=1.0,
         has_se=False), [96, 192, 432, 1008]),
    ('regnetx_4.0gf',
     dict(
         w0=96,
         wa=38.65,
         wm=2.43,
         group_w=40,
         depth=23,
         bot_mul=1.0,
         has_se=False), [80, 240, 560, 1360]),
    ('regnetx_6.4gf',
     dict(
         w0=184,
         wa=60.83,
         wm=2.07,
         group_w=56,
         depth=17,
         bot_mul=1.0,
         has_se=False), [168, 392, 784, 1624]),
    ('regnetx_8.0gf',
     dict(
         w0=80,
         wa=49.56,
         wm=2.88,
         group_w=120,
         depth=23,
         bot_mul=1.0,
         has_se=False), [80, 240, 720, 1920]),
    ('regnetx_12gf',
     dict(
         w0=168,
         wa=73.36,
         wm=2.37,
         group_w=112,
         depth=19,
         bot_mul=1.0,
         has_se=False), [224, 448, 896, 2240]),
    ('regnety_400mf',
     dict(
         w0=48,
         wa=27.89,
         wm=2.09,
         group_w=8,
         depth=16,
         bot_mul=1.0,
         has_se=True), [48, 104, 208, 440]),
    ('regnety_600mf',
     dict(
         w0=48,
         wa=32.54,
         wm=2.32,
         group_w=16,
         depth=15,
         bot_mul=1.0,
         has_se=True), [48, 112, 256, 608]),
    ('regnety_800mf',
     dict(
         w0=56,
         wa=38.84,
         wm=2.4,
         group_w=16,
         depth=14,
         bot_mul=1.0,
         has_se=True), [64, 128, 320, 768]),
    ('regnety_1.6gf',
     dict(
         w0=48,
         wa=20.71,
         wm=2.65,
         group_w=24,
         depth=27,
         bot_mul=1.0,
         has_se=True), [48, 120, 336, 888]),
    ('regnety_3.2gf',
     dict(
         w0=80,
         wa=42.63,
         wm=2.66,
         group_w=24,
         depth=21,
         bot_mul=1.0,
         has_se=True), [72, 216, 576, 1512]),
    ('regnety_4.0gf',
     dict(
         w0=96,
         wa=31.41,
         wm=2.24,
         group_w=64,
         depth=22,
         bot_mul=1.0,
         has_se=True), [128, 192, 512, 1088]),
    ('regnety_6.4gf',
     dict(
         w0=112,
         wa=33.22,
         wm=2.27,
         group_w=72,
         depth=25,
         bot_mul=1.0,
         has_se=True), [144, 288, 576, 1296]),
    ('regnety_8.0gf',
     dict(
         w0=192,
         wa=76.82,
         wm=2.19,
         group_w=56,
         depth=17,
         bot_mul=1.0,
         has_se=True), [168, 448, 896, 2016]),
    ('regnety_12gf',
     dict(
         w0=168,
         wa=73.36,
         wm=2.37,
         group_w=112,
         depth=19,
         bot_mul=1.0,
         has_se=True), [224, 448, 896, 2240]),
]


@pytest.mark.parametrize('arch_name,arch,out_channels', regnet_test_data)
def test_regnet_backbone(arch_name, arch, out_channels):
    with pytest.raises(AssertionError):
        RegNet(arch_name + '233')

    # output the last feature map
    model = RegNet(arch_name)
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert isinstance(feat, torch.Tensor)
    assert feat.shape == (1, out_channels[-1], 7, 7)

    # output feature map of all stages
    model = RegNet(arch_name, out_indices=(0, 1, 2, 3))
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == (1, out_channels[0], 56, 56)
    assert feat[1].shape == (1, out_channels[1], 28, 28)
    assert feat[2].shape == (1, out_channels[2], 14, 14)
    assert feat[3].shape == (1, out_channels[3], 7, 7)


@pytest.mark.parametrize('arch_name,arch,out_channels', regnet_test_data)
def test_custom_arch(arch_name, arch, out_channels):
    # output the last feature map
    model = RegNet(arch)
    model.init_weights()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert isinstance(feat, torch.Tensor)
    assert feat.shape == (1, out_channels[-1], 7, 7)

    # output feature map of all stages
    model = RegNet(arch, out_indices=(0, 1, 2, 3))
    model.init_weights()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == (1, out_channels[0], 56, 56)
    assert feat[1].shape == (1, out_channels[1], 28, 28)
    assert feat[2].shape == (1, out_channels[2], 14, 14)
    assert feat[3].shape == (1, out_channels[3], 7, 7)


def test_exception():
    # arch must be a str or dict
    with pytest.raises(TypeError):
        _ = RegNet(50)
