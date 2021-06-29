from unittest.mock import patch

import pytest
import torch

from mmcls.models.heads import (ClsHead, LinearClsHead, MultiLabelClsHead,
                                MultiLabelLinearClsHead, StackedLinearClsHead)


def test_cls_head():

    # test ClsHead with cal_acc=False
    head = ClsHead()
    fake_cls_score = torch.rand(4, 3)
    fake_gt_label = torch.randint(0, 2, (4, ))

    losses = head.loss(fake_cls_score, fake_gt_label)
    assert losses['loss'].item() > 0

    # test ClsHead with cal_acc=True
    head = ClsHead(cal_acc=True)
    fake_cls_score = torch.rand(4, 3)
    fake_gt_label = torch.randint(0, 2, (4, ))

    losses = head.loss(fake_cls_score, fake_gt_label)
    assert losses['loss'].item() > 0

    # test LinearClsHead
    head = LinearClsHead(10, 100)
    fake_cls_score = torch.rand(4, 10)
    fake_gt_label = torch.randint(0, 10, (4, ))

    losses = head.loss(fake_cls_score, fake_gt_label)
    assert losses['loss'].item() > 0


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


def test_stacked_linear_cls_head():
    # test assertion
    with pytest.raises(AssertionError):
        StackedLinearClsHead(num_classes=3, in_channels=5, mid_channels=10)

    with pytest.raises(AssertionError):
        StackedLinearClsHead(num_classes=-1, in_channels=5, mid_channels=[10])

    fake_img = torch.rand(4, 5)  # B, channel
    fake_gt_label = torch.randint(0, 2, (4, ))  # B, num_classes

    # test forward with default setting
    head = StackedLinearClsHead(
        num_classes=3, in_channels=5, mid_channels=[10])
    head.init_weights()

    losses = head.forward_train(fake_img, fake_gt_label)
    assert losses['loss'].item() > 0

    # test simple test
    pred = head.simple_test(fake_img)
    assert len(pred) == 4

    # test simple test in tracing
    p = patch('torch.onnx.is_in_onnx_export', lambda: True)
    p.start()
    pred = head.simple_test(fake_img)
    assert pred.shape == torch.Size((4, 3))
    p.stop()

    # test forward with full function
    head = StackedLinearClsHead(
        num_classes=3,
        in_channels=5,
        mid_channels=[8, 10],
        dropout_rate=0.2,
        norm_cfg=dict(type='BN1d'),
        act_cfg=dict(type='HSwish'))
    head.init_weights()

    losses = head.forward_train(fake_img, fake_gt_label)
    assert losses['loss'].item() > 0
