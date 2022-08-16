# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import patch

import pytest
import torch

from mmcls.models.heads import (ClsHead, ConformerHead, CSRAClsHead,
                                DeiTClsHead, EfficientFormerClsHead,
                                LinearClsHead, MultiLabelClsHead,
                                MultiLabelLinearClsHead, StackedLinearClsHead,
                                VisionTransformerClsHead)


@pytest.mark.parametrize('feat', [torch.rand(4, 10), (torch.rand(4, 10), )])
def test_cls_head(feat):
    fake_gt_label = torch.randint(0, 10, (4, ))

    # test forward_train with cal_acc=True
    head = ClsHead(cal_acc=True)
    losses = head.forward_train(feat, fake_gt_label)
    assert losses['loss'].item() > 0
    assert 'accuracy' in losses

    # test forward_train with cal_acc=False
    head = ClsHead()
    losses = head.forward_train(feat, fake_gt_label)
    assert losses['loss'].item() > 0

    # test forward_train with weight
    weight = torch.tensor([0.5, 0.5, 0.5, 0.5])
    losses_ = head.forward_train(feat, fake_gt_label)
    losses = head.forward_train(feat, fake_gt_label, weight=weight)
    assert losses['loss'].item() == losses_['loss'].item() * 0.5

    # test simple_test with post_process
    pred = head.simple_test(feat)
    assert isinstance(pred, list) and len(pred) == 4
    with patch('torch.onnx.is_in_onnx_export', return_value=True):
        pred = head.simple_test(feat)
        assert pred.shape == (4, 10)

    # test simple_test without post_process
    pred = head.simple_test(feat, post_process=False)
    assert isinstance(pred, torch.Tensor) and pred.shape == (4, 10)
    logits = head.simple_test(feat, softmax=False, post_process=False)
    torch.testing.assert_allclose(pred, torch.softmax(logits, dim=1))

    # test pre_logits
    features = head.pre_logits(feat)
    if isinstance(feat, tuple):
        torch.testing.assert_allclose(features, feat[0])
    else:
        torch.testing.assert_allclose(features, feat)


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

    # test simple_test with post_process
    pred = head.simple_test(feat)
    assert isinstance(pred, list) and len(pred) == 4
    with patch('torch.onnx.is_in_onnx_export', return_value=True):
        pred = head.simple_test(feat)
        assert pred.shape == (4, 10)

    # test simple_test without post_process
    pred = head.simple_test(feat, post_process=False)
    assert isinstance(pred, torch.Tensor) and pred.shape == (4, 10)
    logits = head.simple_test(feat, softmax=False, post_process=False)
    torch.testing.assert_allclose(pred, torch.softmax(logits, dim=1))

    # test pre_logits
    features = head.pre_logits(feat)
    if isinstance(feat, tuple):
        torch.testing.assert_allclose(features, feat[0])
    else:
        torch.testing.assert_allclose(features, feat)


@pytest.mark.parametrize('feat', [torch.rand(4, 10), (torch.rand(4, 10), )])
def test_multilabel_head(feat):
    head = MultiLabelClsHead()
    fake_gt_label = torch.randint(0, 2, (4, 10))

    losses = head.forward_train(feat, fake_gt_label)
    assert losses['loss'].item() > 0

    # test simple_test with post_process
    pred = head.simple_test(feat)
    assert isinstance(pred, list) and len(pred) == 4
    with patch('torch.onnx.is_in_onnx_export', return_value=True):
        pred = head.simple_test(feat)
        assert pred.shape == (4, 10)

    # test simple_test without post_process
    pred = head.simple_test(feat, post_process=False)
    assert isinstance(pred, torch.Tensor) and pred.shape == (4, 10)
    logits = head.simple_test(feat, sigmoid=False, post_process=False)
    torch.testing.assert_allclose(pred, torch.sigmoid(logits))

    # test pre_logits
    features = head.pre_logits(feat)
    if isinstance(feat, tuple):
        torch.testing.assert_allclose(features, feat[0])
    else:
        torch.testing.assert_allclose(features, feat)


@pytest.mark.parametrize('feat', [torch.rand(4, 5), (torch.rand(4, 5), )])
def test_multilabel_linear_head(feat):
    head = MultiLabelLinearClsHead(10, 5)
    fake_gt_label = torch.randint(0, 2, (4, 10))

    head.init_weights()
    losses = head.forward_train(feat, fake_gt_label)
    assert losses['loss'].item() > 0

    # test simple_test with post_process
    pred = head.simple_test(feat)
    assert isinstance(pred, list) and len(pred) == 4
    with patch('torch.onnx.is_in_onnx_export', return_value=True):
        pred = head.simple_test(feat)
        assert pred.shape == (4, 10)

    # test simple_test without post_process
    pred = head.simple_test(feat, post_process=False)
    assert isinstance(pred, torch.Tensor) and pred.shape == (4, 10)
    logits = head.simple_test(feat, sigmoid=False, post_process=False)
    torch.testing.assert_allclose(pred, torch.sigmoid(logits))

    # test pre_logits
    features = head.pre_logits(feat)
    if isinstance(feat, tuple):
        torch.testing.assert_allclose(features, feat[0])
    else:
        torch.testing.assert_allclose(features, feat)


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
        num_classes=10, in_channels=5, mid_channels=[20])
    head.init_weights()

    losses = head.forward_train(feat, fake_gt_label)
    assert losses['loss'].item() > 0

    # test simple_test with post_process
    pred = head.simple_test(feat)
    assert isinstance(pred, list) and len(pred) == 4
    with patch('torch.onnx.is_in_onnx_export', return_value=True):
        pred = head.simple_test(feat)
        assert pred.shape == (4, 10)

    # test simple_test without post_process
    pred = head.simple_test(feat, post_process=False)
    assert isinstance(pred, torch.Tensor) and pred.shape == (4, 10)
    logits = head.simple_test(feat, softmax=False, post_process=False)
    torch.testing.assert_allclose(pred, torch.softmax(logits, dim=1))

    # test pre_logits
    features = head.pre_logits(feat)
    assert features.shape == (4, 20)

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
    fake_features = ([torch.rand(4, 7, 7, 16), torch.rand(4, 100)], )
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

    head = VisionTransformerClsHead(10, 100, hidden_dim=20)
    # test simple_test with post_process
    pred = head.simple_test(fake_features)
    assert isinstance(pred, list) and len(pred) == 4
    with patch('torch.onnx.is_in_onnx_export', return_value=True):
        pred = head.simple_test(fake_features)
        assert pred.shape == (4, 10)

    # test simple_test without post_process
    pred = head.simple_test(fake_features, post_process=False)
    assert isinstance(pred, torch.Tensor) and pred.shape == (4, 10)
    logits = head.simple_test(fake_features, softmax=False, post_process=False)
    torch.testing.assert_allclose(pred, torch.softmax(logits, dim=1))

    # test pre_logits
    features = head.pre_logits(fake_features)
    assert features.shape == (4, 20)

    # test assertion
    with pytest.raises(ValueError):
        VisionTransformerClsHead(-1, 100)


def test_conformer_head():
    fake_features = ([torch.rand(4, 64), torch.rand(4, 96)], )
    fake_gt_label = torch.randint(0, 10, (4, ))

    # test conformer head forward
    head = ConformerHead(num_classes=10, in_channels=[64, 96])
    losses = head.forward_train(fake_features, fake_gt_label)
    assert losses['loss'].item() > 0

    # test simple_test with post_process
    pred = head.simple_test(fake_features)
    assert isinstance(pred, list) and len(pred) == 4
    with patch('torch.onnx.is_in_onnx_export', return_value=True):
        pred = head.simple_test(fake_features)
        assert pred.shape == (4, 10)

    # test simple_test without post_process
    pred = head.simple_test(fake_features, post_process=False)
    assert isinstance(pred, torch.Tensor) and pred.shape == (4, 10)
    logits = head.simple_test(fake_features, softmax=False, post_process=False)
    torch.testing.assert_allclose(pred, torch.softmax(sum(logits), dim=1))

    # test pre_logits
    features = head.pre_logits(fake_features)
    assert features is fake_features[0]


def test_deit_head():
    fake_features = ([
        torch.rand(4, 7, 7, 16),
        torch.rand(4, 100),
        torch.rand(4, 100)
    ], )
    fake_gt_label = torch.randint(0, 10, (4, ))

    # test deit head forward
    head = DeiTClsHead(num_classes=10, in_channels=100)
    losses = head.forward_train(fake_features, fake_gt_label)
    assert not hasattr(head.layers, 'pre_logits')
    assert not hasattr(head.layers, 'act')
    assert losses['loss'].item() > 0

    # test deit head forward with hidden layer
    head = DeiTClsHead(num_classes=10, in_channels=100, hidden_dim=20)
    losses = head.forward_train(fake_features, fake_gt_label)
    assert hasattr(head.layers, 'pre_logits') and hasattr(head.layers, 'act')
    assert losses['loss'].item() > 0

    # test deit head init_weights
    head = DeiTClsHead(10, 100, hidden_dim=20)
    head.init_weights()
    assert abs(head.layers.pre_logits.weight).sum() > 0

    head = DeiTClsHead(10, 100, hidden_dim=20)
    # test simple_test with post_process
    pred = head.simple_test(fake_features)
    assert isinstance(pred, list) and len(pred) == 4
    with patch('torch.onnx.is_in_onnx_export', return_value=True):
        pred = head.simple_test(fake_features)
        assert pred.shape == (4, 10)

    # test simple_test without post_process
    pred = head.simple_test(fake_features, post_process=False)
    assert isinstance(pred, torch.Tensor) and pred.shape == (4, 10)
    logits = head.simple_test(fake_features, softmax=False, post_process=False)
    torch.testing.assert_allclose(pred, torch.softmax(logits, dim=1))

    # test pre_logits
    cls_token, dist_token = head.pre_logits(fake_features)
    assert cls_token.shape == (4, 20)
    assert dist_token.shape == (4, 20)

    # test assertion
    with pytest.raises(ValueError):
        DeiTClsHead(-1, 100)


def test_efficientformer_head():
    fake_features = (torch.rand(4, 64), )
    fake_gt_label = torch.randint(0, 10, (4, ))

    # Test without distillation head
    head = EfficientFormerClsHead(
        num_classes=10, in_channels=64, distillation=False)

    # test EfficientFormer head forward
    losses = head.forward_train(fake_features, fake_gt_label)
    assert losses['loss'].item() > 0

    # test simple_test with post_process
    pred = head.simple_test(fake_features)
    assert isinstance(pred, list) and len(pred) == 4
    with patch('torch.onnx.is_in_onnx_export', return_value=True):
        pred = head.simple_test(fake_features)
        assert pred.shape == (4, 10)

    # test simple_test without post_process
    pred = head.simple_test(fake_features, post_process=False)
    assert isinstance(pred, torch.Tensor) and pred.shape == (4, 10)
    logits = head.simple_test(fake_features, softmax=False, post_process=False)
    torch.testing.assert_allclose(pred, torch.softmax(logits, dim=1))

    # test pre_logits
    features = head.pre_logits(fake_features)
    assert features is fake_features[0]

    # Test without distillation head
    head = EfficientFormerClsHead(num_classes=10, in_channels=64)
    assert hasattr(head, 'head')
    assert hasattr(head, 'dist_head')

    # Test loss
    with pytest.raises(NotImplementedError):
        losses = head.forward_train(fake_features, fake_gt_label)

    # test simple_test with post_process
    pred = head.simple_test(fake_features)
    assert isinstance(pred, list) and len(pred) == 4
    with patch('torch.onnx.is_in_onnx_export', return_value=True):
        pred = head.simple_test(fake_features)
        assert pred.shape == (4, 10)

    # test simple_test without post_process
    pred = head.simple_test(fake_features, post_process=False)
    assert isinstance(pred, torch.Tensor) and pred.shape == (4, 10)
    logits = head.simple_test(fake_features, softmax=False, post_process=False)
    torch.testing.assert_allclose(pred, torch.softmax(logits, dim=1))

    # test pre_logits
    features = head.pre_logits(fake_features)
    assert features is fake_features[0]


@pytest.mark.parametrize(
    'feat', [torch.rand(4, 20, 20, 30), (torch.rand(4, 20, 20, 30), )])
def test_csra_head(feat):
    head = CSRAClsHead(num_classes=10, in_channels=20, num_heads=1, lam=0.1)
    fake_gt_label = torch.randint(0, 2, (4, 10))

    losses = head.forward_train(feat, fake_gt_label)
    assert losses['loss'].item() > 0

    # test simple_test with post_process
    pred = head.simple_test(feat)
    assert isinstance(pred, list) and len(pred) == 4
    with patch('torch.onnx.is_in_onnx_export', return_value=True):
        pred = head.simple_test(feat)
        assert pred.shape == (4, 10)

    # test pre_logits
    features = head.pre_logits(feat)
    if isinstance(feat, tuple):
        torch.testing.assert_allclose(features, feat[0])
    else:
        torch.testing.assert_allclose(features, feat)
