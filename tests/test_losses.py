import torch

from mmcls.models import build_loss


def test_asymmetric_loss():
    # test asymmetric_loss
    cls_score = torch.Tensor([[5, -5, 0], [5, -5, 0]])
    label = torch.Tensor([[1, 0, 1], [0, 1, 0]])
    weight = torch.tensor(0.5)

    loss_cfg = dict(
        type='AsymmetricLoss',
        gamma_pos=1.0,
        gamma_neg=4.0,
        clip=0.05,
        reduction='mean',
        loss_weight=1.0)
    loss = build_loss(loss_cfg)
    assert torch.allclose(loss(cls_score, label), torch.tensor(3.80845))

    # test asymmetric_loss with weight
    assert torch.allclose(
        loss(cls_score, label, weight=weight), torch.tensor(3.80845 / 2))

    # test asymmetric_loss without clip
    loss_cfg = dict(
        type='AsymmetricLoss',
        gamma_pos=1.0,
        gamma_neg=4.0,
        clip=None,
        reduction='mean',
        loss_weight=1.0)
    loss = build_loss(loss_cfg)
    assert torch.allclose(loss(cls_score, label), torch.tensor(5.1186))
