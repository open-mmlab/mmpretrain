# Copyright (c) OpenMMLab. All rights reserved.
import logging
import tempfile
from unittest import TestCase

import torch
import torch.nn as nn
from mmengine.device import get_device
from mmengine.logging import MMLogger
from mmengine.model import BaseModule
from mmengine.optim import OptimWrapper
from mmengine.runner import Runner
from mmengine.structures import LabelData
from torch.utils.data import Dataset

from mmpretrain.engine import SwAVHook
from mmpretrain.models.heads import SwAVHead
from mmpretrain.models.selfsup import BaseSelfSupervisor
from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample
from mmpretrain.utils import get_ori_model


class DummyDataset(Dataset):
    METAINFO = dict()  # type: ignore
    data = torch.randn(12, 2)
    label = torch.ones(12)

    @property
    def metainfo(self):
        return self.METAINFO

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        data_sample = DataSample()
        gt_label = LabelData(value=self.label[index])
        setattr(data_sample, 'gt_label', gt_label)
        return dict(inputs=[self.data[index]], data_samples=data_sample)


@MODELS.register_module()
class SwAVDummyLayer(BaseModule):

    def __init__(self, init_cfg=None):
        super().__init__(init_cfg)
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)


class ToyModel(BaseSelfSupervisor):

    def __init__(self):
        super().__init__(backbone=dict(type='SwAVDummyLayer'))
        self.prototypes_test = nn.Linear(1, 1)
        self.head = SwAVHead(
            loss=dict(
                type='SwAVLoss',
                feat_dim=2,
                num_crops=[2, 6],
                num_prototypes=3))

    def loss(self, inputs, data_samples):
        labels = []
        for x in data_samples:
            labels.append(x.gt_label.value)
            labels = torch.stack(labels)
        outputs = self.backbone(inputs[0])
        loss = (labels - outputs).sum()
        outputs = dict(loss=loss)
        return outputs


class TestSwAVHook(TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        # `FileHandler` should be closed in Windows, otherwise we cannot
        # delete the temporary directory
        logging.shutdown()
        MMLogger._instance_dict.clear()
        self.temp_dir.cleanup()

    def test_swav_hook(self):
        device = get_device()
        dummy_dataset = DummyDataset()
        toy_model = ToyModel().to(device)
        swav_hook = SwAVHook(
            batch_size=1,
            epoch_queue_starts=15,
            crops_for_assign=[0, 1],
            feat_dim=128,
            queue_length=300,
            frozen_layers_cfg=dict(prototypes=2))

        # test SwAVHook
        runner = Runner(
            model=toy_model,
            work_dir=self.temp_dir.name,
            train_dataloader=dict(
                dataset=dummy_dataset,
                sampler=dict(type='DefaultSampler', shuffle=True),
                collate_fn=dict(type='default_collate'),
                batch_size=1,
                num_workers=0),
            optim_wrapper=OptimWrapper(
                torch.optim.Adam(toy_model.parameters())),
            param_scheduler=dict(type='MultiStepLR', milestones=[1]),
            train_cfg=dict(by_epoch=True, max_epochs=2),
            custom_hooks=[swav_hook],
            default_hooks=dict(logger=None),
            log_processor=dict(window_size=1),
            experiment_name='test_swav_hook',
            default_scope='mmpretrain')

        runner.train()

        for hook in runner.hooks:
            if isinstance(hook, SwAVHook):
                assert hook.queue_length == 300

        assert get_ori_model(runner.model).head.loss_module.use_queue is False
