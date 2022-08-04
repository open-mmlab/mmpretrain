# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
from copy import deepcopy

import numpy as np
import torch
from mmcv import ConfigDict

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

    # test train_step without optimizer
    outputs = model.train_step({'img': imgs, 'gt_label': label})
    assert outputs['loss'].item() > 0
    assert outputs['num_samples'] == 16

    # test val_step
    outputs = model.val_step({'img': imgs, 'gt_label': label}, None)
    assert outputs['loss'].item() > 0
    assert outputs['num_samples'] == 16

    # test val_step without optimizer
    outputs = model.val_step({'img': imgs, 'gt_label': label})
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

    pred = model.simple_test(imgs, softmax=False)
    assert isinstance(pred, list) and len(pred) == 16
    assert len(pred[0] == 10)

    pred = model.simple_test(imgs, softmax=False, post_process=False)
    assert isinstance(pred, torch.Tensor)
    assert pred.shape == (16, 10)

    soft_pred = model.simple_test(imgs, softmax=True, post_process=False)
    assert isinstance(soft_pred, torch.Tensor)
    assert soft_pred.shape == (16, 10)
    torch.testing.assert_allclose(soft_pred, torch.softmax(pred, dim=1))

    # test pretrained
    model_cfg_ = deepcopy(model_cfg)
    model_cfg_['pretrained'] = 'checkpoint'
    model = CLASSIFIERS.build(model_cfg_)
    assert model.init_cfg == dict(type='Pretrained', checkpoint='checkpoint')

    # test show_result
    img = np.random.randint(0, 256, (224, 224, 3)).astype(np.uint8)
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


def test_classifier_extract_feat():
    model_cfg = ConfigDict(
        type='ImageClassifier',
        backbone=dict(
            type='ResNet',
            depth=18,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            style='pytorch'),
        neck=dict(type='GlobalAveragePooling'),
        head=dict(
            type='LinearClsHead',
            num_classes=1000,
            in_channels=512,
            loss=dict(type='CrossEntropyLoss'),
            topk=(1, 5),
        ))

    model = CLASSIFIERS.build(model_cfg)

    # test backbone output
    outs = model.extract_feat(torch.rand(1, 3, 224, 224), stage='backbone')
    assert outs[0].shape == (1, 64, 56, 56)
    assert outs[1].shape == (1, 128, 28, 28)
    assert outs[2].shape == (1, 256, 14, 14)
    assert outs[3].shape == (1, 512, 7, 7)

    # test neck output
    outs = model.extract_feat(torch.rand(1, 3, 224, 224), stage='neck')
    assert outs[0].shape == (1, 64)
    assert outs[1].shape == (1, 128)
    assert outs[2].shape == (1, 256)
    assert outs[3].shape == (1, 512)

    # test pre_logits output
    out = model.extract_feat(torch.rand(1, 3, 224, 224), stage='pre_logits')
    assert out.shape == (1, 512)

    # test transformer style feature extraction
    model_cfg = dict(
        type='ImageClassifier',
        backbone=dict(
            type='VisionTransformer', arch='b', out_indices=[-3, -2, -1]),
        neck=None,
        head=dict(
            type='VisionTransformerClsHead',
            num_classes=1000,
            in_channels=768,
            hidden_dim=1024,
            loss=dict(type='CrossEntropyLoss'),
        ))
    model = CLASSIFIERS.build(model_cfg)

    # test backbone output
    outs = model.extract_feat(torch.rand(1, 3, 224, 224), stage='backbone')
    for out in outs:
        patch_token, cls_token = out
        assert patch_token.shape == (1, 768, 14, 14)
        assert cls_token.shape == (1, 768)

    # test neck output (the same with backbone)
    outs = model.extract_feat(torch.rand(1, 3, 224, 224), stage='neck')
    for out in outs:
        patch_token, cls_token = out
        assert patch_token.shape == (1, 768, 14, 14)
        assert cls_token.shape == (1, 768)

    # test pre_logits output
    out = model.extract_feat(torch.rand(1, 3, 224, 224), stage='pre_logits')
    assert out.shape == (1, 1024)

    # test extract_feats
    multi_imgs = [torch.rand(1, 3, 224, 224) for _ in range(3)]
    outs = model.extract_feats(multi_imgs)
    for outs_per_img in outs:
        for out in outs_per_img:
            patch_token, cls_token = out
            assert patch_token.shape == (1, 768, 14, 14)
            assert cls_token.shape == (1, 768)

    outs = model.extract_feats(multi_imgs, stage='pre_logits')
    for out_per_img in outs:
        assert out_per_img.shape == (1, 1024)

    out = model.forward_dummy(torch.rand(1, 3, 224, 224))
    assert out.shape == (1, 1024)
