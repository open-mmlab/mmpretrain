# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from unittest import TestCase

import torch
from mmengine import ConfigDict
from mmengine.registry import init_default_scope

from mmpretrain.models import AverageClsScoreTTA, ImageClassifier
from mmpretrain.registry import MODELS
from mmpretrain.structures import ClsDataSample

init_default_scope('mmpretrain')


class TestAverageClsScoreTTA(TestCase):
    DEFAULT_ARGS = dict(
        type='AverageClsScoreTTA',
        module=dict(
            type='ImageClassifier',
            backbone=dict(type='ResNet', depth=18),
            neck=dict(type='GlobalAveragePooling'),
            head=dict(
                type='LinearClsHead',
                num_classes=10,
                in_channels=512,
                loss=dict(type='CrossEntropyLoss'))))

    def test_initialize(self):
        model: AverageClsScoreTTA = MODELS.build(self.DEFAULT_ARGS)
        self.assertIsInstance(model.module, ImageClassifier)

    def test_forward(self):
        inputs = torch.rand(1, 3, 224, 224)
        model: AverageClsScoreTTA = MODELS.build(self.DEFAULT_ARGS)

        # The forward of TTA model should not be called.
        with self.assertRaisesRegex(NotImplementedError, 'will not be called'):
            model(inputs)

    def test_test_step(self):
        cfg = ConfigDict(deepcopy(self.DEFAULT_ARGS))
        cfg.module.data_preprocessor = dict(
            mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5])
        model: AverageClsScoreTTA = MODELS.build(cfg)

        img1 = torch.randint(0, 256, (1, 3, 224, 224))
        img2 = torch.randint(0, 256, (1, 3, 224, 224))
        data1 = {
            'inputs': img1,
            'data_samples': [ClsDataSample().set_gt_label(1)]
        }
        data2 = {
            'inputs': img2,
            'data_samples': [ClsDataSample().set_gt_label(1)]
        }
        data_tta = {
            'inputs': [img1, img2],
            'data_samples': [[ClsDataSample().set_gt_label(1)],
                             [ClsDataSample().set_gt_label(1)]]
        }

        score1 = model.module.test_step(data1)[0].pred_label.score
        score2 = model.module.test_step(data2)[0].pred_label.score
        score_tta = model.test_step(data_tta)[0].pred_label.score

        torch.testing.assert_allclose(score_tta, (score1 + score2) / 2)
