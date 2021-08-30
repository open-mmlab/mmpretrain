# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmcls.models import build_loss


def test_asymmetric_loss():
    # test asymmetric_loss
    cls_score = torch.Tensor([[5, -5, 0], [5, -5, 0]])
    label = torch.Tensor([[1, 0, 1], [0, 1, 0]])
    weight = torch.tensor([0.5, 0.5])

    loss_cfg = dict(
        type='AsymmetricLoss',
        gamma_pos=1.0,
        gamma_neg=4.0,
        clip=0.05,
        reduction='mean',
        loss_weight=1.0)
    loss = build_loss(loss_cfg)
    assert torch.allclose(loss(cls_score, label), torch.tensor(3.80845 / 3))

    # test asymmetric_loss with weight
    assert torch.allclose(
        loss(cls_score, label, weight=weight), torch.tensor(3.80845 / 6))

    # test asymmetric_loss without clip
    loss_cfg = dict(
        type='AsymmetricLoss',
        gamma_pos=1.0,
        gamma_neg=4.0,
        clip=None,
        reduction='mean',
        loss_weight=1.0)
    loss = build_loss(loss_cfg)
    assert torch.allclose(loss(cls_score, label), torch.tensor(5.1186 / 3))


def test_cross_entropy_loss():
    with pytest.raises(AssertionError):
        # use_sigmoid and use_soft could not be set simultaneously
        loss_cfg = dict(
            type='CrossEntropyLoss', use_sigmoid=True, use_soft=True)
        loss = build_loss(loss_cfg)

    # test ce_loss
    cls_score = torch.Tensor([[-1000, 1000], [100, -100]])
    label = torch.Tensor([0, 1]).long()
    class_weight = [0.3, 0.7]  # class 0 : 0.3, class 1 : 0.7
    weight = torch.tensor([0.6, 0.4])

    # test ce_loss without class weight
    loss_cfg = dict(type='CrossEntropyLoss', reduction='mean', loss_weight=1.0)
    loss = build_loss(loss_cfg)
    assert torch.allclose(loss(cls_score, label), torch.tensor(1100.))
    # test ce_loss with weight
    assert torch.allclose(
        loss(cls_score, label, weight=weight), torch.tensor(640.))

    # test ce_loss with class weight
    loss_cfg = dict(
        type='CrossEntropyLoss',
        reduction='mean',
        loss_weight=1.0,
        class_weight=class_weight)
    loss = build_loss(loss_cfg)
    assert torch.allclose(loss(cls_score, label), torch.tensor(370.))
    # test ce_loss with weight
    assert torch.allclose(
        loss(cls_score, label, weight=weight), torch.tensor(208.))

    # test bce_loss
    cls_score = torch.Tensor([[-200, 100], [500, -1000], [300, -300]])
    label = torch.Tensor([[1, 0], [0, 1], [1, 0]])
    weight = torch.Tensor([0.6, 0.4, 0.5])
    class_weight = [0.1, 0.9]  # class 0: 0.1, class 1: 0.9

    # test bce_loss without class weight
    loss_cfg = dict(
        type='CrossEntropyLoss',
        use_sigmoid=True,
        reduction='mean',
        loss_weight=1.0)
    loss = build_loss(loss_cfg)
    assert torch.allclose(loss(cls_score, label), torch.tensor(300.))
    # test ce_loss with weight
    assert torch.allclose(
        loss(cls_score, label, weight=weight), torch.tensor(130.))

    # test bce_loss with class weight
    loss_cfg = dict(
        type='CrossEntropyLoss',
        use_sigmoid=True,
        reduction='mean',
        loss_weight=1.0,
        class_weight=class_weight)
    loss = build_loss(loss_cfg)
    assert torch.allclose(loss(cls_score, label), torch.tensor(176.667))
    # test bce_loss with weight
    assert torch.allclose(
        loss(cls_score, label, weight=weight), torch.tensor(74.333))

    # test soft_ce_loss
    cls_score = torch.Tensor([[-1000, 1000], [100, -100]])
    label = torch.Tensor([[1.0, 0.0], [0.0, 1.0]])
    class_weight = [0.3, 0.7]  # class 0 : 0.3, class 1 : 0.7
    weight = torch.tensor([0.6, 0.4])

    # test soft_ce_loss without class weight
    loss_cfg = dict(
        type='CrossEntropyLoss',
        use_soft=True,
        reduction='mean',
        loss_weight=1.0)
    loss = build_loss(loss_cfg)
    assert torch.allclose(loss(cls_score, label), torch.tensor(1100.))
    # test soft_ce_loss with weight
    assert torch.allclose(
        loss(cls_score, label, weight=weight), torch.tensor(640.))

    # test soft_ce_loss with class weight
    loss_cfg = dict(
        type='CrossEntropyLoss',
        use_soft=True,
        reduction='mean',
        loss_weight=1.0,
        class_weight=class_weight)
    loss = build_loss(loss_cfg)
    assert torch.allclose(loss(cls_score, label), torch.tensor(370.))
    # test soft_ce_loss with weight
    assert torch.allclose(
        loss(cls_score, label, weight=weight), torch.tensor(208.))


def test_focal_loss():
    # test focal_loss
    cls_score = torch.Tensor([[5, -5, 0], [5, -5, 0]])
    label = torch.Tensor([[1, 0, 1], [0, 1, 0]])
    weight = torch.tensor([0.5, 0.5])

    loss_cfg = dict(
        type='FocalLoss',
        gamma=2.0,
        alpha=0.25,
        reduction='mean',
        loss_weight=1.0)
    loss = build_loss(loss_cfg)
    assert torch.allclose(loss(cls_score, label), torch.tensor(0.8522))
    # test focal_loss with weight
    assert torch.allclose(
        loss(cls_score, label, weight=weight), torch.tensor(0.8522 / 2))


def test_label_smooth_loss():
    # test label_smooth_val assertion
    with pytest.raises(AssertionError):
        loss_cfg = dict(type='LabelSmoothLoss', label_smooth_val=1.0)
        build_loss(loss_cfg)

    with pytest.raises(AssertionError):
        loss_cfg = dict(type='LabelSmoothLoss', label_smooth_val='str')
        build_loss(loss_cfg)

    # test reduction assertion
    with pytest.raises(AssertionError):
        loss_cfg = dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, reduction='unknown')
        build_loss(loss_cfg)

    # test mode assertion
    with pytest.raises(AssertionError):
        loss_cfg = dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='unknown')
        build_loss(loss_cfg)

    # test original mode label smooth loss
    cls_score = torch.tensor([[1., -1.]])
    label = torch.tensor([0])

    loss_cfg = dict(
        type='LabelSmoothLoss',
        label_smooth_val=0.1,
        mode='original',
        reduction='mean',
        loss_weight=1.0)
    loss = build_loss(loss_cfg)
    correct = 0.2269  # from timm
    assert loss(cls_score, label) - correct <= 0.0001

    # test classy_vision mode label smooth loss
    loss_cfg = dict(
        type='LabelSmoothLoss',
        label_smooth_val=0.1,
        mode='classy_vision',
        reduction='mean',
        loss_weight=1.0)
    loss = build_loss(loss_cfg)
    correct = 0.2178  # from ClassyVision
    assert loss(cls_score, label) - correct <= 0.0001

    # test multi_label mode label smooth loss
    cls_score = torch.tensor([[1., -1., 1]])
    label = torch.tensor([[1, 0, 1]])

    loss_cfg = dict(
        type='LabelSmoothLoss',
        label_smooth_val=0.1,
        mode='multi_label',
        reduction='mean',
        loss_weight=1.0)
    loss = build_loss(loss_cfg)
    smooth_label = torch.tensor([[0.9, 0.1, 0.9]])
    correct = torch.binary_cross_entropy_with_logits(cls_score,
                                                     smooth_label).mean()
    assert torch.allclose(loss(cls_score, label), correct)

    # test label linear combination smooth loss
    cls_score = torch.tensor([[1., -1., 0.]])
    label1 = torch.tensor([[1., 0., 0.]])
    label2 = torch.tensor([[0., 0., 1.]])
    label_mix = label1 * 0.6 + label2 * 0.4

    loss_cfg = dict(
        type='LabelSmoothLoss',
        label_smooth_val=0.1,
        mode='original',
        reduction='mean',
        num_classes=3,
        loss_weight=1.0)
    loss = build_loss(loss_cfg)
    smooth_label1 = loss.original_smooth_label(label1)
    smooth_label2 = loss.original_smooth_label(label2)
    label_smooth_mix = smooth_label1 * 0.6 + smooth_label2 * 0.4
    correct = (-torch.log_softmax(cls_score, -1) * label_smooth_mix).sum()

    assert loss(cls_score, label_mix) - correct <= 0.0001

    # test label smooth loss with weight
    cls_score = torch.tensor([[1., -1.], [1., -1.]])
    label = torch.tensor([0, 1])
    weight = torch.tensor([0.5, 0.5])

    loss_cfg = dict(
        type='LabelSmoothLoss',
        reduction='mean',
        label_smooth_val=0.1,
        loss_weight=1.0)
    loss = build_loss(loss_cfg)
    assert torch.allclose(
        loss(cls_score, label, weight=weight),
        loss(cls_score, label) / 2)
