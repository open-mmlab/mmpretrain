from unittest.mock import patch

import pytest
import torch

from mmcls.models.heads import (ClsHead, LinearClsHead, MultiLabelClsHead,
                                MultiLabelLinearClsHead, StackedLinearClsHead,
                                VisionTransformerClsHead)


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


def test_linear_head():

    fake_features = torch.rand(4, 100)
    fake_gt_label = torch.randint(0, 10, (4, ))

    # test LinearClsHead forward
    head = LinearClsHead(10, 100)
    losses = head.forward_train(fake_features, fake_gt_label)
    assert losses['loss'].item() > 0

    # test init weights
    head = LinearClsHead(10, 100)
    head.init_weights()
    assert abs(head.fc.weight).sum() > 0

    # test simple_test
    head = LinearClsHead(10, 100)
    pred = head.simple_test(fake_features)
    assert isinstance(pred, list) and len(pred) == 4

    with patch('torch.onnx.is_in_onnx_export', return_value=True):
        head = LinearClsHead(10, 100)
        pred = head.simple_test(fake_features)
        assert pred.shape == (4, 10)


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
    with patch('torch.onnx.is_in_onnx_export', return_value=True):
        pred = head.simple_test(fake_img)
        assert pred.shape == torch.Size((4, 3))

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


def test_vit_head():
    fake_features = torch.rand(4, 100)
    fake_gt_label = torch.randint(0, 10, (4, ))

    # test vit head forward
    head = VisionTransformerClsHead(10, 100)
    losses = head.forward_train(fake_features, fake_gt_label)
    assert not hasattr(head.layers, 'pre_logits')
    assert not hasattr(head.layers, 'act')
    assert losses['loss'].item() > 0

    # test vit head forward with hidden layer
    head = VisionTransformerClsHead(10, 100, hidden_dim=20)
    losses = head.forward_train(fake_features, fake_gt_label)
    assert hasattr(head.layers, 'pre_logits') and hasattr(head.layers, 'act')
    assert losses['loss'].item() > 0

    # test vit head init_weights
    head = VisionTransformerClsHead(10, 100, hidden_dim=20)
    head.init_weights()
    assert abs(head.layers.pre_logits.weight).sum() > 0

    # test simple_test
    head = VisionTransformerClsHead(10, 100, hidden_dim=20)
    pred = head.simple_test(fake_features)
    assert isinstance(pred, list) and len(pred) == 4

    with patch('torch.onnx.is_in_onnx_export', return_value=True):
        head = VisionTransformerClsHead(10, 100, hidden_dim=20)
        pred = head.simple_test(fake_features)
        assert pred.shape == (4, 10)

    # test assertion
    with pytest.raises(ValueError):
        VisionTransformerClsHead(-1, 100)
