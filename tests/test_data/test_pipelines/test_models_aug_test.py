# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import torch
from mmcv.parallel import collate
from mmcv.utils import build_from_cfg

from mmcls.datasets.builder import PIPELINES
from mmcls.datasets.pipelines import Compose
from mmcls.models import build_classifier


def model_aug_test_template(cfg_file):
    # get config
    cfg = mmcv.Config.fromfile(cfg_file)
    # init model
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_classifier(cfg.model)

    # init test pipeline and set aug test
    load_cfg = cfg.test_pipeline[:-1]
    multi_scale_cfg = cfg.test_pipeline[-1]
    multi_scale_cfg['flip'] = True
    multi_scale_cfg['flip_direction'] = ['horizontal', 'vertical', 'diagonal']

    load = Compose(load_cfg)
    transform = build_from_cfg(multi_scale_cfg, PIPELINES)

    results = dict(
        img_prefix=osp.join(osp.dirname(__file__), '../../data'),
        img_info=dict(filename='color.jpg'))
    results = transform(load(results))
    assert len(results['img']) == 4
    assert len(results['img_metas']) == 4

    results['img'] = [collate([x]) for x in results['img']]
    results['img_metas'] = [collate([x]).data[0] for x in results['img_metas']]
    # aug test the model
    model.eval()
    with torch.no_grad():
        aug_result = model(return_loss=False, **results)
    return aug_result


def test_aug_test_size():
    results = dict(
        img_prefix=osp.join(osp.dirname(__file__), '../../data'),
        img_info=dict(filename='color.jpg'))

    # Define simple pipeline
    load = dict(type='LoadImageFromFile')
    load = build_from_cfg(load, PIPELINES)

    # get config
    transform = dict(
        type='FlipAug',
        transforms=[],
        flip=True,
        flip_direction=['horizontal', 'vertical', 'diagonal'])
    multi_aug_test_module = build_from_cfg(transform, PIPELINES)

    results = load(results)
    results = multi_aug_test_module(load(results))
    # len(["original", "horizontal", "vertical", "diagonal"])
    assert len(results['img']) == 4


def test_resnet_aug_test():
    aug_result = model_aug_test_template(
        'configs/resnet/resnet18_8xb32_in1k_tta.py')
    assert len(aug_result[0]) == 1000
