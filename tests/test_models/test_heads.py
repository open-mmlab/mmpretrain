# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine import is_seq_of

from mmcls.core import ClsDataSample
from mmcls.registry import MODELS
from mmcls.utils import register_all_modules

register_all_modules()


class TestClsHead(TestCase):
    DEFAULT_ARGS = dict(type='ClsHead')

    def test_pre_logits(self):
        head = MODELS.build(self.DEFAULT_ARGS)

        # return the last item
        feats = (torch.rand(4, 10), torch.rand(4, 10))
        pre_logits = head.pre_logits(feats)
        self.assertIs(pre_logits, feats[-1])

    def test_forward(self):
        head = MODELS.build(self.DEFAULT_ARGS)

        # return the last item (same as pre_logits)
        feats = (torch.rand(4, 10), torch.rand(4, 10))
        outs = head(feats)
        self.assertIs(outs, feats[-1])

    def test_loss(self):
        feats = (torch.rand(4, 10), )
        data_samples = [ClsDataSample().set_gt_label(1) for _ in range(4)]

        # with cal_acc = False
        head = MODELS.build(self.DEFAULT_ARGS)

        losses = head.loss(feats, data_samples)
        self.assertEqual(losses.keys(), {'loss'})
        self.assertGreater(losses['loss'].item(), 0)

        # with cal_acc = True
        cfg = {**self.DEFAULT_ARGS, 'topk': (1, 2), 'cal_acc': True}
        head = MODELS.build(cfg)

        losses = head.loss(feats, data_samples)
        self.assertEqual(losses.keys(),
                         {'loss', 'accuracy_top-1', 'accuracy_top-2'})
        self.assertGreater(losses['loss'].item(), 0)

        # test assertion when cal_acc but data is batch agumented.
        data_samples = [
            sample.set_gt_score(torch.rand(10)) for sample in data_samples
        ]
        cfg = {
            **self.DEFAULT_ARGS, 'cal_acc': True,
            'loss': dict(type='CrossEntropyLoss', use_soft=True)
        }
        head = MODELS.build(cfg)
        with self.assertRaisesRegex(AssertionError, 'batch augmentation'):
            head.loss(feats, data_samples)

    def test_predict(self):
        feats = (torch.rand(4, 10), )
        data_samples = [ClsDataSample().set_gt_label(1) for _ in range(4)]
        head = MODELS.build(self.DEFAULT_ARGS)

        # with without data_samples
        predictions = head.predict(feats)
        self.assertTrue(is_seq_of(predictions, ClsDataSample))
        for pred in predictions:
            self.assertIn('label', pred.pred_label)
            self.assertIn('score', pred.pred_label)

        # with with data_samples
        predictions = head.predict(feats, data_samples)
        self.assertTrue(is_seq_of(predictions, ClsDataSample))
        for sample, pred in zip(data_samples, predictions):
            self.assertIs(sample, pred)
            self.assertIn('label', pred.pred_label)
            self.assertIn('score', pred.pred_label)


class TestLinearClsHead(TestCase):
    DEFAULT_ARGS = dict(type='LinearClsHead', in_channels=10, num_classes=5)

    def test_initialize(self):
        with self.assertRaisesRegex(ValueError, 'num_classes=-5 must be'):
            MODELS.build({**self.DEFAULT_ARGS, 'num_classes': -5})

    def test_pre_logits(self):
        head = MODELS.build(self.DEFAULT_ARGS)

        # return the last item
        feats = (torch.rand(4, 10), torch.rand(4, 10))
        pre_logits = head.pre_logits(feats)
        self.assertIs(pre_logits, feats[-1])

    def test_forward(self):
        head = MODELS.build(self.DEFAULT_ARGS)

        feats = (torch.rand(4, 10), torch.rand(4, 10))
        outs = head(feats)
        self.assertEqual(outs.shape, (4, 5))


"""Temporarily disabled.
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
"""
