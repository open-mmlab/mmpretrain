# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
from copy import deepcopy

import numpy as np
import pytest
import torch
from mmcv import ConfigDict
from mmcv.runner.base_module import BaseModule

from mmcls.models import CLASSIFIERS
from mmcls.models.classifiers import ImageClassifier


def test_image_classifier():
    model_cfg = dict(
        type='ImageClassifier',
        backbone=dict(
            type='ResNet_CIFAR',
            depth=50,
            num_stages=4,
            out_indices=(3, ),
            style='pytorch'),
        neck=dict(type='GlobalAveragePooling'),
        head=dict(
            type='LinearClsHead',
            num_classes=10,
            in_channels=2048,
            loss=dict(type='CrossEntropyLoss')))

    imgs = torch.randn(16, 3, 32, 32)
    label = torch.randint(0, 10, (16, ))

    model_cfg_ = deepcopy(model_cfg)
    model = CLASSIFIERS.build(model_cfg_)

    # test property
    assert model.with_neck
    assert model.with_head

    # test train_step
    outputs = model.train_step({'img': imgs, 'gt_label': label}, None)
    assert outputs['loss'].item() > 0
    assert outputs['num_samples'] == 16

    # test val_step
    outputs = model.val_step({'img': imgs, 'gt_label': label}, None)
    assert outputs['loss'].item() > 0
    assert outputs['num_samples'] == 16

    # test forward
    losses = model(imgs, return_loss=True, gt_label=label)
    assert losses['loss'].item() > 0

    # test forward_test
    model_cfg_ = deepcopy(model_cfg)
    model = CLASSIFIERS.build(model_cfg_)
    pred = model(imgs, return_loss=False, img_metas=None)
    assert isinstance(pred, list) and len(pred) == 16

    single_img = torch.randn(1, 3, 32, 32)
    pred = model(single_img, return_loss=False, img_metas=None)
    assert isinstance(pred, list) and len(pred) == 1

    # test pretrained
    # TODO remove deprecated pretrained
    with pytest.warns(UserWarning):
        model_cfg_ = deepcopy(model_cfg)
        model_cfg_['pretrained'] = 'checkpoint'
        model = CLASSIFIERS.build(model_cfg_)
        assert model.init_cfg == dict(
            type='Pretrained', checkpoint='checkpoint')

    # test show_result
    img = np.random.random_integers(0, 255, (224, 224, 3)).astype(np.uint8)
    result = dict(pred_class='cat', pred_label=0, pred_score=0.9)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = osp.join(tmpdir, 'out.png')
        model.show_result(img, result, out_file=out_file)
        assert osp.exists(out_file)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = osp.join(tmpdir, 'out.png')
        model.show_result(img, result, out_file=out_file)
        assert osp.exists(out_file)


def test_image_classifier_with_mixup():
    # Test mixup in ImageClassifier
    model_cfg = dict(
        backbone=dict(
            type='ResNet_CIFAR',
            depth=50,
            num_stages=4,
            out_indices=(3, ),
            style='pytorch'),
        neck=dict(type='GlobalAveragePooling'),
        head=dict(
            type='MultiLabelLinearClsHead',
            num_classes=10,
            in_channels=2048,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0,
                      use_soft=True)),
        train_cfg=dict(
            augments=dict(
                type='BatchMixup', alpha=1., num_classes=10, prob=1.)))
    img_classifier = ImageClassifier(**model_cfg)
    img_classifier.init_weights()
    imgs = torch.randn(16, 3, 32, 32)
    label = torch.randint(0, 10, (16, ))

    losses = img_classifier.forward_train(imgs, label)
    assert losses['loss'].item() > 0

    # Considering BC-breaking
    # TODO remove deprecated mixup usage.
    model_cfg['train_cfg'] = dict(mixup=dict(alpha=1.0, num_classes=10))
    img_classifier = ImageClassifier(**model_cfg)
    img_classifier.init_weights()
    imgs = torch.randn(16, 3, 32, 32)
    label = torch.randint(0, 10, (16, ))

    losses = img_classifier.forward_train(imgs, label)
    assert losses['loss'].item() > 0


def test_image_classifier_with_cutmix():

    # Test cutmix in ImageClassifier
    model_cfg = dict(
        backbone=dict(
            type='ResNet_CIFAR',
            depth=50,
            num_stages=4,
            out_indices=(3, ),
            style='pytorch'),
        neck=dict(type='GlobalAveragePooling'),
        head=dict(
            type='MultiLabelLinearClsHead',
            num_classes=10,
            in_channels=2048,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0,
                      use_soft=True)),
        train_cfg=dict(
            augments=dict(
                type='BatchCutMix', alpha=1., num_classes=10, prob=1.)))
    img_classifier = ImageClassifier(**model_cfg)
    img_classifier.init_weights()
    imgs = torch.randn(16, 3, 32, 32)
    label = torch.randint(0, 10, (16, ))

    losses = img_classifier.forward_train(imgs, label)
    assert losses['loss'].item() > 0

    # Considering BC-breaking
    # TODO remove deprecated mixup usage.
    model_cfg['train_cfg'] = dict(
        cutmix=dict(alpha=1.0, num_classes=10, cutmix_prob=1.0))
    img_classifier = ImageClassifier(**model_cfg)
    img_classifier.init_weights()
    imgs = torch.randn(16, 3, 32, 32)
    label = torch.randint(0, 10, (16, ))

    losses = img_classifier.forward_train(imgs, label)
    assert losses['loss'].item() > 0


def test_image_classifier_with_augments():

    imgs = torch.randn(16, 3, 32, 32)
    label = torch.randint(0, 10, (16, ))

    # Test cutmix and mixup in ImageClassifier
    model_cfg = dict(
        backbone=dict(
            type='ResNet_CIFAR',
            depth=50,
            num_stages=4,
            out_indices=(3, ),
            style='pytorch'),
        neck=dict(type='GlobalAveragePooling'),
        head=dict(
            type='MultiLabelLinearClsHead',
            num_classes=10,
            in_channels=2048,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0,
                      use_soft=True)),
        train_cfg=dict(augments=[
            dict(type='BatchCutMix', alpha=1., num_classes=10, prob=0.5),
            dict(type='BatchMixup', alpha=1., num_classes=10, prob=0.3),
            dict(type='Identity', num_classes=10, prob=0.2)
        ]))
    img_classifier = ImageClassifier(**model_cfg)
    img_classifier.init_weights()

    losses = img_classifier.forward_train(imgs, label)
    assert losses['loss'].item() > 0

    # Test cutmix with cutmix_minmax in ImageClassifier
    model_cfg['train_cfg'] = dict(
        augments=dict(
            type='BatchCutMix',
            alpha=1.,
            num_classes=10,
            prob=1.,
            cutmix_minmax=[0.2, 0.8]))
    img_classifier = ImageClassifier(**model_cfg)
    img_classifier.init_weights()

    losses = img_classifier.forward_train(imgs, label)
    assert losses['loss'].item() > 0

    # Test not using train_cfg
    model_cfg = dict(
        backbone=dict(
            type='ResNet_CIFAR',
            depth=50,
            num_stages=4,
            out_indices=(3, ),
            style='pytorch'),
        neck=dict(type='GlobalAveragePooling'),
        head=dict(
            type='LinearClsHead',
            num_classes=10,
            in_channels=2048,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0)))
    img_classifier = ImageClassifier(**model_cfg)
    img_classifier.init_weights()
    imgs = torch.randn(16, 3, 32, 32)
    label = torch.randint(0, 10, (16, ))

    losses = img_classifier.forward_train(imgs, label)
    assert losses['loss'].item() > 0

    # Test not using cutmix and mixup in ImageClassifier
    model_cfg['train_cfg'] = dict(augments=None)
    img_classifier = ImageClassifier(**model_cfg)
    img_classifier.init_weights()

    losses = img_classifier.forward_train(imgs, label)
    assert losses['loss'].item() > 0


def test_image_classifier_return_tuple():
    model_cfg = ConfigDict(
        type='ImageClassifier',
        backbone=dict(
            type='ResNet_CIFAR',
            depth=50,
            num_stages=4,
            out_indices=(3, ),
            style='pytorch',
            return_tuple=False),
        head=dict(
            type='LinearClsHead',
            num_classes=10,
            in_channels=2048,
            loss=dict(type='CrossEntropyLoss')))

    imgs = torch.randn(16, 3, 32, 32)

    model_cfg_ = deepcopy(model_cfg)
    with pytest.warns(DeprecationWarning):
        model = CLASSIFIERS.build(model_cfg_)

    # test backbone return tensor
    feat = model.extract_feat(imgs)
    assert isinstance(feat, torch.Tensor)

    # test backbone return tuple
    model_cfg_ = deepcopy(model_cfg)
    model_cfg_.backbone.return_tuple = True
    model = CLASSIFIERS.build(model_cfg_)

    feat = model.extract_feat(imgs)
    assert isinstance(feat, tuple)

    # test warning if backbone return tensor
    class ToyBackbone(BaseModule):

        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 16, 3)

        def forward(self, x):
            return self.conv(x)

    model_cfg_ = deepcopy(model_cfg)
    model_cfg_.backbone.return_tuple = True
    model = CLASSIFIERS.build(model_cfg_)
    model.backbone = ToyBackbone()

    with pytest.warns(DeprecationWarning):
        model.extract_feat(imgs)
