# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import patch

import pytest
import torch

from mmcls.models.heads import (ClsHead, LinearClsHead, MultiLabelClsHead,
                                MultiLabelLinearClsHead, StackedLinearClsHead,
                                VisionTransformerClsHead)


@pytest.mark.parametrize('feat', [torch.rand(4, 3), (torch.rand(4, 3), )])
def test_cls_head(feat):

    # test ClsHead with cal_acc=False
    head = ClsHead()
    fake_gt_label = torch.randint(0, 2, (4, ))

    losses = head.forward_train(feat, fake_gt_label)
    assert losses['loss'].item() > 0

    # test ClsHead with cal_acc=True
    head = ClsHead(cal_acc=True)
    feat = torch.rand(4, 3)
    fake_gt_label = torch.randint(0, 2, (4, ))

    losses = head.forward_train(feat, fake_gt_label)
    assert losses['loss'].item() > 0


@pytest.mark.parametrize('feat', [torch.rand(4, 3), (torch.rand(4, 3), )])
def test_linear_head(feat):

    fake_gt_label = torch.randint(0, 10, (4, ))

    # test LinearClsHead forward
    head = LinearClsHead(10, 3)
    losses = head.forward_train(feat, fake_gt_label)
    assert losses['loss'].item() > 0

    # test init weights
    head = LinearClsHead(10, 3)
    head.init_weights()
    assert abs(head.fc.weight).sum() > 0

    # test simple_test
    head = LinearClsHead(10, 3)
    pred = head.simple_test(feat)
    assert isinstance(pred, list) and len(pred) == 4

    with patch('torch.onnx.is_in_onnx_export', return_value=True):
        head = LinearClsHead(10, 3)
        pred = head.simple_test(feat)
        assert pred.shape == (4, 10)


@pytest.mark.parametrize('feat', [torch.rand(4, 3), (torch.rand(4, 3), )])
def test_multilabel_head(feat):
    head = MultiLabelClsHead()
    fake_gt_label = torch.randint(0, 2, (4, 3))

    losses = head.forward_train(feat, fake_gt_label)
    assert losses['loss'].item() > 0


@pytest.mark.parametrize('feat', [torch.rand(4, 5), (torch.rand(4, 5), )])
def test_multilabel_linear_head(feat):
    head = MultiLabelLinearClsHead(3, 5)
    fake_gt_label = torch.randint(0, 2, (4, 3))

    head.init_weights()
    losses = head.forward_train(feat, fake_gt_label)
    assert losses['loss'].item() > 0


@pytest.mark.parametrize('feat', [torch.rand(4, 5), (torch.rand(4, 5), )])
def test_stacked_linear_cls_head(feat):
    # test assertion
    with pytest.raises(AssertionError):
        StackedLinearClsHead(num_classes=3, in_channels=5, mid_channels=10)

    with pytest.raises(AssertionError):
        StackedLinearClsHead(num_classes=-1, in_channels=5, mid_channels=[10])

    fake_gt_label = torch.randint(0, 2, (4, ))  # B, num_classes

    # test forward with default setting
    head = StackedLinearClsHead(
        num_classes=3, in_channels=5, mid_channels=[10])
    head.init_weights()

    losses = head.forward_train(feat, fake_gt_label)
    assert losses['loss'].item() > 0

    # test simple test
    pred = head.simple_test(feat)
    assert len(pred) == 4

    # test simple test in tracing
    with patch('torch.onnx.is_in_onnx_export', return_value=True):
        pred = head.simple_test(feat)
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

    losses = head.forward_train(feat, fake_gt_label)
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
