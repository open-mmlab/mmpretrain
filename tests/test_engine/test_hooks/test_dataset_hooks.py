# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np
import torch
import torch.nn as nn
from mmengine.logging import MessageHub
from mmengine.model import BaseModel
from mmengine.optim import OptimWrapper
from mmengine.runner import Runner

from mmcls.datasets import BaseDataset
from mmcls.engine import PushDataInfoToMessageHubHook
from mmcls.utils import register_all_modules

register_all_modules()


class SimpleModel(BaseModel):

    def __init__(self):
        super().__init__()
        self.data_preprocessor = None
        self.fc = nn.Linear(3, 10)

    def forward(self, inputs, labels, mode='tensor'):
        pass


class ExampleDataset(BaseDataset):

    def __init__(self, ):
        super().__init__(ann_file='./tmp.txt')

    def load_data_list(self):
        return [{
            'img_path': f'base_folder/{i}.jpg',
            'gt_label': i,
        } for i in range(10)]


class TestPushDataInfoToMessageHubHook(TestCase):

    def test_init(self):
        with self.assertRaisesRegex(ValueError, '`keys` must be str'):
            PushDataInfoToMessageHubHook(keys=1)

        with self.assertRaisesRegex(AssertionError, 'key must be in'):
            PushDataInfoToMessageHubHook(keys=['gt_labels', 'metainfo'])

        # Test with seq of str type keys
        PushDataInfoToMessageHubHook(keys=['gt_labels'])

        # Test with str type keys
        PushDataInfoToMessageHubHook(keys='gt_labels')

    def test_before_run(self):
        push_hook = dict(
            type='PushDataInfoToMessageHubHook',
            keys=['gt_labels', 'img_paths'])

        model = SimpleModel()
        runner = Runner(
            model=model,
            train_dataloader=dict(
                dataset=ExampleDataset(),
                sampler=dict(type='DefaultSampler', shuffle=True),
                batch_size=5,
                num_workers=0,
                collate_fn=dict(type='default_collate'),
            ),
            optim_wrapper=OptimWrapper(
                optimizer=torch.optim.Adam(model.parameters(), lr=0.)),
            train_cfg=dict(by_epoch=True, max_epochs=2, val_interval=10),
            work_dir=MagicMock(),
            default_hooks=dict(logger=None),
            custom_hooks=[],
            default_scope='mmcls',
            experiment_name='test_dataset_hooks1')

        runner.register_custom_hooks([push_hook])
        runner.call_hook('before_run')

        gt_labels = MessageHub.get_current_instance().get_info('gt_labels')
        except_gt_labels = np.arange(10)
        self.assertEqual(gt_labels.shape, except_gt_labels.shape)
        assert np.equal(gt_labels, except_gt_labels).all()

        img_paths = MessageHub.get_current_instance().get_info('img_paths')
        except_img_paths = [f'base_folder/{i}.jpg' for i in range(10)]
        self.assertListEqual(img_paths, except_img_paths)
