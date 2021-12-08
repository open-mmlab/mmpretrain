# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmcls.models.utils import Augments


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
