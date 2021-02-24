import torch

from mmcls.models.heads import MultiLabelClsHead, MultiLabelLinearClsHead


def test_multilabel_head():
    head = MultiLabelClsHead()
    fake_cls_score = torch.rand(4, 3)
    fake_gt_label = torch.randint(0, 2, (4, 3))

    losses = head.loss(fake_cls_score, fake_gt_label)
    assert losses['loss'].item() > 0


def test_multilabel_linear_head():
    head = MultiLabelLinearClsHead(3, 5)
    fake_cls_score = torch.rand(4, 3)
    fake_gt_label = torch.randint(0, 2, (4, 3))

    head.init_weights()
    losses = head.loss(fake_cls_score, fake_gt_label)
    assert losses['loss'].item() > 0
