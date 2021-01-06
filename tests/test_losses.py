import torch

from mmcls.models import build_loss


def test_focal_loss():
    # test focal_loss
    cls_score = torch.Tensor([[5, -5, 0], [5, -5, 0]])
    label = torch.Tensor([[1, 0, 1], [0, 1, 0]])
    weight = torch.tensor(0.5)

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
