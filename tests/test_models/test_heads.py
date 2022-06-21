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


class TestVisionTransformerClsHead(TestCase):
    DEFAULT_ARGS = dict(
        type='VisionTransformerClsHead', in_channels=10, num_classes=5)
    fake_feats = ([torch.rand(4, 7, 7, 16), torch.rand(4, 10)], )

    def test_initialize(self):
        with self.assertRaisesRegex(ValueError, 'num_classes=-5 must be'):
            MODELS.build({**self.DEFAULT_ARGS, 'num_classes': -5})

        # test vit head default
        head = MODELS.build(self.DEFAULT_ARGS)
        assert not hasattr(head.layers, 'pre_logits')
        assert not hasattr(head.layers, 'act')

        # test vit head hidden_dim
        head = MODELS.build({**self.DEFAULT_ARGS, 'hidden_dim': 30})
        assert hasattr(head.layers, 'pre_logits')
        assert hasattr(head.layers, 'act')

        # test vit head init_weights
        head = MODELS.build(self.DEFAULT_ARGS)
        head.init_weights()

        # test vit head init_weights with hidden_dim
        head = MODELS.build({**self.DEFAULT_ARGS, 'hidden_dim': 30})
        head.init_weights()
        assert abs(head.layers.pre_logits.weight).sum() > 0

    def test_pre_logits(self):
        # test default
        head = MODELS.build(self.DEFAULT_ARGS)
        pre_logits = head.pre_logits(self.fake_feats)
        self.assertIs(pre_logits, self.fake_feats[-1][1])

        # test hidden_dim
        head = MODELS.build({**self.DEFAULT_ARGS, 'hidden_dim': 30})
        pre_logits = head.pre_logits(self.fake_feats)
        self.assertEqual(pre_logits.shape, (4, 30))

    def test_forward(self):
        # test default
        head = MODELS.build(self.DEFAULT_ARGS)
        outs = head(self.fake_feats)
        self.assertEqual(outs.shape, (4, 5))

        # test hidden_dim
        head = MODELS.build({**self.DEFAULT_ARGS, 'hidden_dim': 30})
        outs = head(self.fake_feats)
        self.assertEqual(outs.shape, (4, 5))


class TestDeiTClsHead(TestVisionTransformerClsHead):
    DEFAULT_ARGS = dict(type='DeiTClsHead', in_channels=10, num_classes=5)
    fake_feats = ([
        torch.rand(4, 7, 7, 16),
        torch.rand(4, 10),
        torch.rand(4, 10)
    ], )

    def test_pre_logits(self):
        # test default
        head = MODELS.build(self.DEFAULT_ARGS)
        cls_token, dist_token = head.pre_logits(self.fake_feats)
        self.assertIs(cls_token, self.fake_feats[-1][1])
        self.assertIs(dist_token, self.fake_feats[-1][2])

        # test hidden_dim
        head = MODELS.build({**self.DEFAULT_ARGS, 'hidden_dim': 30})
        cls_token, dist_token = head.pre_logits(self.fake_feats)
        self.assertEqual(cls_token.shape, (4, 30))
        self.assertEqual(dist_token.shape, (4, 30))


class TestConformerHead(TestCase):
    DEFAULT_ARGS = dict(
        type='ConformerHead', in_channels=[64, 96], num_classes=5)
    fake_feats = ([torch.rand(4, 64), torch.rand(4, 96)], )

    def test_initialize(self):
        with self.assertRaisesRegex(ValueError, 'num_classes=-5 must be'):
            MODELS.build({**self.DEFAULT_ARGS, 'num_classes': -5})

        # test default
        head = MODELS.build(self.DEFAULT_ARGS)
        assert hasattr(head, 'conv_cls_head')
        assert hasattr(head, 'trans_cls_head')

        # test init_weights
        head = MODELS.build(self.DEFAULT_ARGS)
        head.init_weights()
        assert abs(head.conv_cls_head.weight).sum() > 0
        assert abs(head.trans_cls_head.weight).sum() > 0

    def test_pre_logits(self):
        # test default
        head = MODELS.build(self.DEFAULT_ARGS)
        pre_logits = head.pre_logits(self.fake_feats)
        self.assertIs(pre_logits, self.fake_feats[-1])

    def test_forward(self):
        head = MODELS.build(self.DEFAULT_ARGS)
        outs = head(self.fake_feats)
        self.assertEqual(outs[0].shape, (4, 5))
        self.assertEqual(outs[1].shape, (4, 5))

    def test_loss(self):
        data_samples = [ClsDataSample().set_gt_label(1) for _ in range(4)]

        # with cal_acc = False
        head = MODELS.build(self.DEFAULT_ARGS)

        losses = head.loss(self.fake_feats, data_samples)
        self.assertEqual(losses.keys(), {'loss'})
        self.assertGreater(losses['loss'].item(), 0)

        # with cal_acc = True
        cfg = {**self.DEFAULT_ARGS, 'topk': (1, 2), 'cal_acc': True}
        head = MODELS.build(cfg)

        losses = head.loss(self.fake_feats, data_samples)
        self.assertEqual(losses.keys(),
                         {'loss', 'accuracy_top-1', 'accuracy_top-2'})
        self.assertGreater(losses['loss'].item(), 0)

        # test assertion when cal_acc but data is batch agumented.
        data_samples = [
            sample.set_gt_score(torch.rand(5)) for sample in data_samples
        ]
        cfg = {
            **self.DEFAULT_ARGS, 'cal_acc': True,
            'loss': dict(type='CrossEntropyLoss', use_soft=True)
        }
        head = MODELS.build(cfg)
        with self.assertRaisesRegex(AssertionError, 'batch augmentation'):
            head.loss(self.fake_feats, data_samples)

    def test_predict(self):
        data_samples = [ClsDataSample().set_gt_label(1) for _ in range(4)]
        head = MODELS.build(self.DEFAULT_ARGS)

        # with without data_samples
        predictions = head.predict(self.fake_feats)
        self.assertTrue(is_seq_of(predictions, ClsDataSample))
        for pred in predictions:
            self.assertIn('label', pred.pred_label)
            self.assertIn('score', pred.pred_label)

        # with with data_samples
        predictions = head.predict(self.fake_feats, data_samples)
        self.assertTrue(is_seq_of(predictions, ClsDataSample))
        for sample, pred in zip(data_samples, predictions):
            self.assertIs(sample, pred)
            self.assertIn('label', pred.pred_label)
            self.assertIn('score', pred.pred_label)


class TestStackedLinearClsHead(TestCase):
    DEFAULT_ARGS = dict(
        type='StackedLinearClsHead', in_channels=10, num_classes=5)
    fake_feats = (torch.rand(4, 10), )

    def test_initialize(self):
        with self.assertRaisesRegex(ValueError, 'num_classes=-5 must be'):
            MODELS.build({
                **self.DEFAULT_ARGS, 'num_classes': -5,
                'mid_channels': 10
            })

        # test mid_channels
        with self.assertRaisesRegex(AssertionError, 'should be a sequence'):
            MODELS.build({**self.DEFAULT_ARGS, 'mid_channels': 10})

        # test default
        head = MODELS.build({**self.DEFAULT_ARGS, 'mid_channels': [20]})
        assert len(head.layers) == 2
        head.init_weights()

    def test_pre_logits(self):
        # test default
        head = MODELS.build({**self.DEFAULT_ARGS, 'mid_channels': [20, 30]})
        pre_logits = head.pre_logits(self.fake_feats)
        self.assertEqual(pre_logits.shape, (4, 30))

    def test_forward(self):
        # test default
        head = MODELS.build({**self.DEFAULT_ARGS, 'mid_channels': [20, 30]})
        outs = head(self.fake_feats)
        self.assertEqual(outs.shape, (4, 5))

        head = MODELS.build({
            **self.DEFAULT_ARGS, 'mid_channels': [8, 10],
            'dropout_rate': 0.2,
            'norm_cfg': dict(type='BN1d'),
            'act_cfg': dict(type='HSwish')
        })
        outs = head(self.fake_feats)
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

"""
