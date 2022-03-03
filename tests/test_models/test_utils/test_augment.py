# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmcls.models.utils import Augments

augment_cfgs = [
    dict(type='BatchCutMix', alpha=1., prob=1.),
    dict(type='BatchMixup', alpha=1., prob=1.),
    dict(type='Identity', prob=1.),
    dict(type='BatchResizeMix', alpha=1., prob=1.)
]


def test_augments():
    imgs = torch.randn(4, 3, 32, 32)
    labels = torch.randint(0, 10, (4, ))

    # Test cutmix
    augments_cfg = dict(type='BatchCutMix', alpha=1., num_classes=10, prob=1.)
    augs = Augments(augments_cfg)
    mixed_imgs, mixed_labels = augs(imgs, labels)
    assert mixed_imgs.shape == torch.Size((4, 3, 32, 32))
    assert mixed_labels.shape == torch.Size((4, 10))

    # Test mixup
    augments_cfg = dict(type='BatchMixup', alpha=1., num_classes=10, prob=1.)
    augs = Augments(augments_cfg)
    mixed_imgs, mixed_labels = augs(imgs, labels)
    assert mixed_imgs.shape == torch.Size((4, 3, 32, 32))
    assert mixed_labels.shape == torch.Size((4, 10))

    # Test resizemix
    augments_cfg = dict(
        type='BatchResizeMix', alpha=1., num_classes=10, prob=1.)
    augs = Augments(augments_cfg)
    mixed_imgs, mixed_labels = augs(imgs, labels)
    assert mixed_imgs.shape == torch.Size((4, 3, 32, 32))
    assert mixed_labels.shape == torch.Size((4, 10))

    # Test cutmixup
    augments_cfg = [
        dict(type='BatchCutMix', alpha=1., num_classes=10, prob=0.5),
        dict(type='BatchMixup', alpha=1., num_classes=10, prob=0.3)
    ]
    augs = Augments(augments_cfg)
    mixed_imgs, mixed_labels = augs(imgs, labels)
    assert mixed_imgs.shape == torch.Size((4, 3, 32, 32))
    assert mixed_labels.shape == torch.Size((4, 10))

    augments_cfg = [
        dict(type='BatchCutMix', alpha=1., num_classes=10, prob=0.5),
        dict(type='BatchMixup', alpha=1., num_classes=10, prob=0.5)
    ]
    augs = Augments(augments_cfg)
    mixed_imgs, mixed_labels = augs(imgs, labels)
    assert mixed_imgs.shape == torch.Size((4, 3, 32, 32))
    assert mixed_labels.shape == torch.Size((4, 10))

    augments_cfg = [
        dict(type='BatchCutMix', alpha=1., num_classes=10, prob=0.5),
        dict(type='BatchMixup', alpha=1., num_classes=10, prob=0.3),
        dict(type='Identity', num_classes=10, prob=0.2)
    ]
    augs = Augments(augments_cfg)
    mixed_imgs, mixed_labels = augs(imgs, labels)
    assert mixed_imgs.shape == torch.Size((4, 3, 32, 32))
    assert mixed_labels.shape == torch.Size((4, 10))


@pytest.mark.parametrize('cfg', augment_cfgs)
def test_binary_augment(cfg):

    cfg_ = dict(num_classes=1, **cfg)
    augs = Augments(cfg_)

    imgs = torch.randn(4, 3, 32, 32)
    labels = torch.randint(0, 2, (4, 1)).float()

    mixed_imgs, mixed_labels = augs(imgs, labels)
    assert mixed_imgs.shape == torch.Size((4, 3, 32, 32))
    assert mixed_labels.shape == torch.Size((4, 1))


@pytest.mark.parametrize('cfg', augment_cfgs)
def test_multilabel_augment(cfg):

    cfg_ = dict(num_classes=10, **cfg)
    augs = Augments(cfg_)

    imgs = torch.randn(4, 3, 32, 32)
    labels = torch.randint(0, 2, (4, 10)).float()

    mixed_imgs, mixed_labels = augs(imgs, labels)
    assert mixed_imgs.shape == torch.Size((4, 3, 32, 32))
    assert mixed_labels.shape == torch.Size((4, 10))
