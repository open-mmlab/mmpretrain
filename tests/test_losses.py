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
    cls_score = torch.Tensor([[100, -100]])
    label = torch.Tensor([1]).long()
    weight = torch.tensor(0.5)

    loss_cfg = dict(type='CrossEntropyLoss', reduction='mean', loss_weight=1.0)
    loss = build_loss(loss_cfg)
    assert torch.allclose(loss(cls_score, label), torch.tensor(200.))
    # test ce_loss with weight
    assert torch.allclose(
        loss(cls_score, label, weight=weight), torch.tensor(100.))

    # test bce_loss
    cls_score = torch.Tensor([[100, -100], [100, -100]])
    label = torch.Tensor([[1, 0], [0, 1]])
    weight = torch.Tensor([0.5, 0.5])

    loss_cfg = dict(
        type='CrossEntropyLoss',
        use_sigmoid=True,
        reduction='mean',
        loss_weight=1.0)
    loss = build_loss(loss_cfg)
    assert torch.allclose(loss(cls_score, label), torch.tensor(50.))
    # test ce_loss with weight
    assert torch.allclose(
        loss(cls_score, label, weight=weight), torch.tensor(25.))

    # test soft_ce_loss
    cls_score = torch.Tensor([[100, -100]])
    label = torch.Tensor([[1, 0], [0, 1]])
    weight = torch.tensor(0.5)

    loss_cfg = dict(
        type='CrossEntropyLoss',
        use_soft=True,
        reduction='mean',
        loss_weight=1.0)
    loss = build_loss(loss_cfg)
    assert torch.allclose(loss(cls_score, label), torch.tensor(100.))
    # test soft_ce_loss with weight
    assert torch.allclose(
        loss(cls_score, label, weight=weight), torch.tensor(50.))


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
    # test label smooth loss
    cls_score = torch.tensor([[1., -1.]])
    label = torch.tensor([0])

    loss_cfg = dict(
        type='LabelSmoothLoss',
        reduction='mean',
        label_smooth_val=0.1,
        loss_weight=1.0)
    loss = build_loss(loss_cfg)
    assert loss(cls_score, label) - 0.2179 <= 0.0001

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
