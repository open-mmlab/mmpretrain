# Copyright (c) OpenMMLab. All rights reserved.
import copy
import random
from unittest import TestCase

import numpy as np
import torch
from mmengine import is_seq_of

from mmcls.registry import MODELS
from mmcls.structures import ClsDataSample
from mmcls.utils import register_all_modules

register_all_modules()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class TestClsHead(TestCase):
    DEFAULT_ARGS = dict(type='ClsHead')
    FAKE_FEATS = (torch.rand(4, 10), )

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
        feats = self.FAKE_FEATS
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
    FAKE_FEATS = (torch.rand(4, 10), )

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


class TestMultiLabelClsHead(TestCase):
    DEFAULT_ARGS = dict(type='MultiLabelClsHead')

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
        data_samples = [ClsDataSample().set_gt_label([0, 3]) for _ in range(4)]

        # Test with thr and topk are all None
        head = MODELS.build(self.DEFAULT_ARGS)
        losses = head.loss(feats, data_samples)
        self.assertEqual(head.thr, 0.5)
        self.assertEqual(head.topk, None)
        self.assertEqual(losses.keys(), {'loss'})
        self.assertGreater(losses['loss'].item(), 0)

        # Test with topk
        cfg = copy.deepcopy(self.DEFAULT_ARGS)
        cfg['topk'] = 2
        head = MODELS.build(cfg)
        losses = head.loss(feats, data_samples)
        self.assertEqual(head.thr, None, cfg)
        self.assertEqual(head.topk, 2)
        self.assertEqual(losses.keys(), {'loss'})
        self.assertGreater(losses['loss'].item(), 0)

        # Test with thr
        setup_seed(0)
        cfg = copy.deepcopy(self.DEFAULT_ARGS)
        cfg['thr'] = 0.1
        head = MODELS.build(cfg)
        thr_losses = head.loss(feats, data_samples)
        self.assertEqual(head.thr, 0.1)
        self.assertEqual(head.topk, None)
        self.assertEqual(thr_losses.keys(), {'loss'})
        self.assertGreater(thr_losses['loss'].item(), 0)

        # Test with thr and topk are all not None
        setup_seed(0)
        cfg = copy.deepcopy(self.DEFAULT_ARGS)
        cfg['thr'] = 0.1
        cfg['topk'] = 2
        head = MODELS.build(cfg)
        thr_topk_losses = head.loss(feats, data_samples)
        self.assertEqual(head.thr, 0.1)
        self.assertEqual(head.topk, 2)
        self.assertEqual(thr_topk_losses.keys(), {'loss'})
        self.assertGreater(thr_topk_losses['loss'].item(), 0)

        # Test with gt_lable with score
        data_samples = [
            ClsDataSample().set_gt_score(torch.rand((10, ))) for _ in range(4)
        ]

        head = MODELS.build(self.DEFAULT_ARGS)
        losses = head.loss(feats, data_samples)
        self.assertEqual(head.thr, 0.5)
        self.assertEqual(head.topk, None)
        self.assertEqual(losses.keys(), {'loss'})
        self.assertGreater(losses['loss'].item(), 0)

    def test_predict(self):
        feats = (torch.rand(4, 10), )
        data_samples = [ClsDataSample().set_gt_label([1, 2]) for _ in range(4)]
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

        # Test with topk
        cfg = copy.deepcopy(self.DEFAULT_ARGS)
        cfg['topk'] = 2
        head = MODELS.build(cfg)
        predictions = head.predict(feats, data_samples)
        self.assertEqual(head.thr, None)
        self.assertTrue(is_seq_of(predictions, ClsDataSample))
        for sample, pred in zip(data_samples, predictions):
            self.assertIs(sample, pred)
            self.assertIn('label', pred.pred_label)
            self.assertIn('score', pred.pred_label)


class EfficientFormerClsHead(TestClsHead):
    DEFAULT_ARGS = dict(
        type='EfficientFormerClsHead',
        in_channels=10,
        num_classes=10,
        distillation=False)
    FAKE_FEATS = (torch.rand(4, 10), )

    def test_forward(self):
        # test with distillation head
        cfg = copy.deepcopy(self.DEFAULT_ARGS)
        cfg['distillation'] = True
        head = MODELS.build(cfg)
        self.assertTrue(hasattr(head, 'dist_head'))
        feats = (torch.rand(4, 10), torch.rand(4, 10))
        outs = head(feats)
        self.assertEqual(outs.shape, (4, 10))

        # test without distillation head
        cfg = copy.deepcopy(self.DEFAULT_ARGS)
        head = MODELS.build(cfg)
        self.assertFalse(hasattr(head, 'dist_head'))
        feats = (torch.rand(4, 10), torch.rand(4, 10))
        outs = head(feats)
        self.assertEqual(outs.shape, (4, 10))

    def test_loss(self):
        feats = (torch.rand(4, 10), )
        data_samples = [ClsDataSample().set_gt_label(1) for _ in range(4)]

        # test with distillation head
        cfg = copy.deepcopy(self.DEFAULT_ARGS)
        cfg['distillation'] = True
        head = MODELS.build(cfg)
        with self.assertRaisesRegex(NotImplementedError, 'MMClassification '):
            head.loss(feats, data_samples)

        # test without distillation head
        super().test_loss()


class TestMultiLabelLinearClsHead(TestMultiLabelClsHead):
    DEFAULT_ARGS = dict(
        type='MultiLabelLinearClsHead', num_classes=10, in_channels=10)

    def test_forward(self):
        head = MODELS.build(self.DEFAULT_ARGS)
        self.assertTrue(hasattr(head, 'fc'))
        self.assertTrue(isinstance(head.fc, torch.nn.Linear))

        # return the last item (same as pre_logits)
        feats = (torch.rand(4, 10), torch.rand(4, 10))
        head(feats)


class TestArcFaceClsHead(TestCase):
    DEFAULT_ARGS = dict(type='ArcFaceClsHead', in_channels=10, num_classes=5)

    def test_initialize(self):
        with self.assertRaisesRegex(ValueError, 'num_classes=-5 must be'):
            MODELS.build({**self.DEFAULT_ARGS, 'num_classes': -5})

        with self.assertRaisesRegex(AssertionError):
            MODELS.build({**self.DEFAULT_ARGS, 'num_subcenters': 0})

    def test_pre_logits(self):
        head = MODELS.build(self.DEFAULT_ARGS)

        # return the last item
        feats = (torch.rand(4, 10), torch.rand(4, 10))
        pre_logits = head.pre_logits(feats)
        self.assertIs(pre_logits, feats[-1])

        # Test with SubCenterArcFace
        head = MODELS.build({**self.DEFAULT_ARGS, 'num_subcenters': 3})
        feats = (torch.rand(4, 10), torch.rand(4, 10))
        pre_logits = head.pre_logits(feats)
        self.assertIs(pre_logits, feats[-1])

    def test_forward(self):
        head = MODELS.build(self.DEFAULT_ARGS)
        # target is not None
        feats = (torch.rand(4, 10), torch.rand(4, 10))
        target = torch.zeros(4)
        outs = head(feats, target)
        self.assertEqual(outs.shape, (4, 5))

        # target is None
        feats = (torch.rand(4, 10), torch.rand(4, 10))
        outs = head(feats)
        self.assertEqual(outs.shape, (4, 5))

        # Test with SubCenterArcFace
        head = MODELS.build({**self.DEFAULT_ARGS, 'num_subcenters': 3})
        # target is not None
        feats = (torch.rand(4, 10), torch.rand(4, 10))
        target = torch.zeros(4)
        outs = head(feats, target)
        self.assertEqual(outs.shape, (4, 5))

        # target is None
        feats = (torch.rand(4, 10), torch.rand(4, 10))
        outs = head(feats)
        self.assertEqual(outs.shape, (4, 5))

    def test_loss(self):
        feats = (torch.rand(4, 10), )
        data_samples = [ClsDataSample().set_gt_label(1) for _ in range(4)]

        # test loss with used='before'
        head = MODELS.build(self.DEFAULT_ARGS)
        losses = head.loss(feats, data_samples)
        self.assertEqual(losses.keys(), {'loss'})
        self.assertGreater(losses['loss'].item(), 0)

        # Test with SubCenterArcFace
        head = MODELS.build({**self.DEFAULT_ARGS, 'num_subcenters': 3})
        # test loss with used='before'
        losses = head.loss(feats, data_samples)
        self.assertEqual(losses.keys(), {'loss'})
        self.assertGreater(losses['loss'].item(), 0)
