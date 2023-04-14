# Copyright (c) OpenMMLab. All rights reserved.
import tempfile
from unittest import TestCase

import numpy as np
import torch
from mmengine.runner import Runner
from torch.utils.data import DataLoader, Dataset


class ExampleDataset(Dataset):

    def __init__(self):
        self.index = 0
        self.metainfo = None

    def __getitem__(self, idx):
        results = dict(imgs=torch.rand((224, 224, 3)).float(), )
        return results

    def get_gt_labels(self):
        gt_labels = np.array([0, 1, 2, 4, 0, 4, 1, 2, 2, 1])
        return gt_labels

    def __len__(self):
        return 10


class TestSetAdaptiveMarginsHook(TestCase):
    DEFAULT_HOOK_CFG = dict(type='SetAdaptiveMarginsHook')
    DEFAULT_MODEL = dict(
        type='ImageClassifier',
        backbone=dict(
            type='ResNet',
            depth=34,
            num_stages=4,
            out_indices=(3, ),
            style='pytorch'),
        neck=dict(type='GlobalAveragePooling'),
        head=dict(type='ArcFaceClsHead', in_channels=512, num_classes=5))

    def test_before_train(self):
        default_hooks = dict(
            timer=dict(type='IterTimerHook'),
            logger=None,
            param_scheduler=dict(type='ParamSchedulerHook'),
            checkpoint=dict(type='CheckpointHook', interval=1),
            sampler_seed=dict(type='DistSamplerSeedHook'),
            visualization=dict(type='VisualizationHook', enable=False),
        )
        tmpdir = tempfile.TemporaryDirectory()
        loader = DataLoader(ExampleDataset(), batch_size=2)
        self.runner = Runner(
            model=self.DEFAULT_MODEL,
            work_dir=tmpdir.name,
            train_dataloader=loader,
            train_cfg=dict(by_epoch=True, max_epochs=1),
            log_level='WARNING',
            optim_wrapper=dict(
                optimizer=dict(type='SGD', lr=0.1, momentum=0.9)),
            param_scheduler=dict(
                type='MultiStepLR', milestones=[1, 2], gamma=0.1),
            default_scope='mmpretrain',
            default_hooks=default_hooks,
            experiment_name='test_construct_with_arcface',
            custom_hooks=[self.DEFAULT_HOOK_CFG])

        default_margins = torch.tensor([0.5] * 5)
        torch.allclose(self.runner.model.head.margins.cpu(), default_margins)
        self.runner.call_hook('before_train')
        # counts = [2 ,3 , 3, 0, 2] -> [2 ,3 , 3, 1, 2] at least occur once
        # feqercy**-0.25 = [0.84089642, 0.75983569, 0.75983569, 1., 0.84089642]
        # normized = [0.33752196, 0.   , 0.   , 1.  , 0.33752196]
        # margins =  [0.20188488, 0.05, 0.05, 0.5, 0.20188488]
        expert_margins = torch.tensor(
            [0.20188488, 0.05, 0.05, 0.5, 0.20188488])
        torch.allclose(self.runner.model.head.margins.cpu(), expert_margins)

        model_cfg = {**self.DEFAULT_MODEL}
        model_cfg['head'] = dict(
            type='LinearClsHead',
            num_classes=1000,
            in_channels=512,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
            topk=(1, 5),
        )
        self.runner = Runner(
            model=model_cfg,
            work_dir=tmpdir.name,
            train_dataloader=loader,
            train_cfg=dict(by_epoch=True, max_epochs=1),
            log_level='WARNING',
            optim_wrapper=dict(
                optimizer=dict(type='SGD', lr=0.1, momentum=0.9)),
            param_scheduler=dict(
                type='MultiStepLR', milestones=[1, 2], gamma=0.1),
            default_scope='mmpretrain',
            default_hooks=default_hooks,
            experiment_name='test_construct_wo_arcface',
            custom_hooks=[self.DEFAULT_HOOK_CFG])
        with self.assertRaises(ValueError):
            self.runner.call_hook('before_train')
