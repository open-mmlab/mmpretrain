import numpy as np
import pytest
import torch
import torch.nn as nn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import EpochBasedRunner, build_optimizer
from mmcv.utils import get_logger
from torch.utils.data import DataLoader, Dataset

from mmcls.core import PreciseBNHook


class ExampleDataset(Dataset):

    def __init__(self):
        self.index = 0

    def __getitem__(self, idx):
        results = dict(imgs=torch.tensor([1.0], dtype=torch.float32))
        return results

    def __len__(self):
        return 1


class BiggerDataset(ExampleDataset):

    def __init__(self, fixed_values=range(0, 12)):
        assert len(self) == len(fixed_values)
        self.fixed_values = fixed_values

    def __getitem__(self, idx):
        results = dict(
            imgs=torch.tensor([self.fixed_values[idx]], dtype=torch.float32))
        return results

    def __len__(self):
        # a bigger dataset
        return 12


class ExampleModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Linear(1, 1)
        self.bn = nn.BatchNorm1d(1)
        self.test_cfg = None

    def forward(self, imgs, return_loss=False):
        return self.bn(self.conv(imgs))

    def train_step(self, data_batch, optimizer, **kwargs):
        outputs = {
            'loss': 0.5,
            'log_vars': {
                'accuracy': 0.98
            },
            'num_samples': 1
        }
        return outputs


class SingleBNModel(ExampleModel):

    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(1)
        self.test_cfg = None

    def forward(self, imgs, return_loss=False):
        return self.bn(imgs)


class GNExampleModel(ExampleModel):

    def __init__(self):
        super().__init__()
        self.conv = nn.Linear(1, 1)
        self.bn = nn.GroupNorm(1, 1)
        self.test_cfg = None


class NoBNExampleModel(ExampleModel):

    def __init__(self):
        super().__init__()
        self.conv = nn.Linear(1, 1)
        self.test_cfg = None

    def forward(self, imgs, return_loss=False):
        return self.conv(imgs)


def test_precise_bn():
    with pytest.raises(TypeError):
        # dataloaders must be a List,but got str
        test_dataset = ExampleModel()
        loader = DataLoader(test_dataset, batch_size=2)
        loaders = [loader]
        PreciseBNHook('loaders')

    with pytest.raises(TypeError):
        # dataloaders must be a list of Pytorch Dataloaders ,
        # but got torch.Dataset
        test_dataset = ExampleModel()
        loaders = [test_dataset]
        PreciseBNHook(loaders)

    optimizer_cfg = dict(
        type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

    test_dataset = ExampleDataset()
    loader = DataLoader(test_dataset, batch_size=2)
    model = ExampleModel()
    optimizer = build_optimizer(model, optimizer_cfg)
    logger = get_logger('precise_bn')
    runner = EpochBasedRunner(
        model=model, batch_processor=None, optimizer=optimizer, logger=logger)

    with pytest.raises(AssertionError):
        # data_loaders must be larger than 0
        data_loaders = []
        precise_bn_hook = PreciseBNHook(data_loaders, num_items=5)
        runner.register_hook(precise_bn_hook)
        runner.run(data_loaders, [('train', 1)], 1)

    # test non-DDP model
    test_bigger_dataset = BiggerDataset()
    loader = DataLoader(test_bigger_dataset, batch_size=2)
    loaders = [loader]
    precise_bn_hook = PreciseBNHook(loaders, num_items=4)
    assert precise_bn_hook.num_items == 4
    assert precise_bn_hook.interval == 1
    runner = EpochBasedRunner(
        model=model, batch_processor=None, optimizer=optimizer, logger=logger)
    runner.register_hook(precise_bn_hook)
    runner.run(loaders, [('train', 1)], 1)

    # test DP model
    test_bigger_dataset = BiggerDataset()
    loader = DataLoader(test_bigger_dataset, batch_size=2)
    loaders = [loader]
    precise_bn_hook = PreciseBNHook(loaders, num_items=4)
    assert precise_bn_hook.num_items == 4
    assert precise_bn_hook.interval == 1
    model = MMDataParallel(model)
    runner = EpochBasedRunner(
        model=model, batch_processor=None, optimizer=optimizer, logger=logger)
    runner.register_hook(precise_bn_hook)
    runner.run(loaders, [('train', 1)], 1)

    # test model w/ gn layer
    loader = DataLoader(test_bigger_dataset, batch_size=2)
    loaders = [loader]
    precise_bn_hook = PreciseBNHook(loaders, num_items=4)
    assert precise_bn_hook.num_items == 4
    assert precise_bn_hook.interval == 1
    model = GNExampleModel()
    runner = EpochBasedRunner(
        model=model, batch_processor=None, optimizer=optimizer, logger=logger)
    runner.register_hook(precise_bn_hook)
    runner.run(loaders, [('train', 1)], 1)

    # test model without bn layer
    loader = DataLoader(test_bigger_dataset, batch_size=2)
    loaders = [loader]
    precise_bn_hook = PreciseBNHook(loaders, num_items=4)
    assert precise_bn_hook.num_items == 4
    assert precise_bn_hook.interval == 1
    model = NoBNExampleModel()
    runner = EpochBasedRunner(
        model=model, batch_processor=None, optimizer=optimizer, logger=logger)
    runner.register_hook(precise_bn_hook)
    runner.run(loaders, [('train', 1)], 1)

    # test how precise it is
    loader = DataLoader(test_bigger_dataset, batch_size=2)
    loaders = [loader]
    precise_bn_hook = PreciseBNHook(loaders, num_items=12)  # run all
    assert precise_bn_hook.num_items == 12
    assert precise_bn_hook.interval == 1
    model = SingleBNModel()
    runner = EpochBasedRunner(
        model=model, batch_processor=None, optimizer=optimizer, logger=logger)
    runner.register_hook(precise_bn_hook)
    runner.run(loaders, [('train', 1)], 1)
    imgs_list = list()
    for loader in loaders:
        for i, data in enumerate(loader):
            imgs_list.append(np.array(data['imgs']))
    mean = np.mean([np.mean(batch) for batch in imgs_list])
    # bassel correction used in Pytorch, therefore ddof=1
    var = np.mean([np.var(batch, ddof=1) for batch in imgs_list])
    assert np.equal(mean, np.array(
        model.bn.running_mean)), (mean, np.array(model.bn.running_mean))
    assert np.equal(var, np.array(
        model.bn.running_var)), (var, np.array(model.bn.running_var))

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason='requires CUDA support')
    def test_ddp_model_precise_bn():
        # test DDP model
        test_bigger_dataset = BiggerDataset()
        loader = DataLoader(test_bigger_dataset, batch_size=2)
        loaders = [loader]
        precise_bn_hook = PreciseBNHook(loaders, num_items=5)
        assert precise_bn_hook.num_items == 5
        assert precise_bn_hook.interval == 1
        model = ExampleModel()
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=True)
        runner = EpochBasedRunner(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            logger=logger)
        runner.register_hook(precise_bn_hook)
        runner.run(loaders, [('train', 1)], 1)
