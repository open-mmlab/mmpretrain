# Copyright (c) OpenMMLab. All rights reserved.
import logging
import tempfile
from unittest.mock import MagicMock

import mmcv.runner as mmcv_runner
import pytest
import torch
from mmcv.runner import obj_from_dict
from torch.utils.data import DataLoader, Dataset

from mmcls.core.hook import ClassNumCheckHook
from mmcls.models.heads.base_head import BaseHead


class ExampleDataset(Dataset):

    def __init__(self, CLASSES):
        self.CLASSES = CLASSES

    def __getitem__(self, idx):
        results = dict(img=torch.tensor([1]), img_metas=dict())
        return results

    def __len__(self):
        return 1


class ExampleHead(BaseHead):

    def __init__(self, init_cfg=None):
        super(BaseHead, self).__init__(init_cfg)
        self.num_classes = 4

    def forward_train(self, x, gt_label=None, **kwargs):
        pass


class ExampleModel(torch.nn.Module):

    def __init__(self):
        super(ExampleModel, self).__init__()
        self.test_cfg = None
        self.conv = torch.nn.Conv2d(3, 3, 3)
        self.head = ExampleHead()

    def forward(self, img, img_metas, test_mode=False, **kwargs):
        return img

    def train_step(self, data_batch, optimizer):
        loss = self.forward(**data_batch)
        return dict(loss=loss)


@pytest.mark.parametrize('runner_type',
                         ['EpochBasedRunner', 'IterBasedRunner'])
@pytest.mark.parametrize(
    'CLASSES', [None, ('A', 'B', 'C', 'D', 'E'), ('A', 'B', 'C', 'D')])
def test_num_class_hook(runner_type, CLASSES):
    test_dataset = ExampleDataset(CLASSES)
    loader = DataLoader(test_dataset, batch_size=1)
    model = ExampleModel()
    optim_cfg = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
    optimizer = obj_from_dict(optim_cfg, torch.optim,
                              dict(params=model.parameters()))

    with tempfile.TemporaryDirectory() as tmpdir:
        num_class_hook = ClassNumCheckHook()
        logger_mock = MagicMock(spec=logging.Logger)
        runner = getattr(mmcv_runner, runner_type)(
            model=model,
            optimizer=optimizer,
            work_dir=tmpdir,
            logger=logger_mock,
            max_epochs=1)
        runner.register_hook(num_class_hook)
        if CLASSES is None:
            runner.run([loader], [('train', 1)], 1)
            logger_mock.warning.assert_called()
        elif len(CLASSES) != 4:
            with pytest.raises(AssertionError):
                runner.run([loader], [('train', 1)], 1)
        else:
            runner.run([loader], [('train', 1)], 1)
