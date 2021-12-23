from unittest.mock import patch

import torch
from mmcv.utils import digit_version

from mmcls.datasets import build_dataloader


class TestDataloaderBuilder():

    @classmethod
    def setup_class(cls):
        cls.data = list(range(20))
        cls.samples_per_gpu = 5
        cls.workers_per_gpu = 1

    @patch('mmcls.datasets.builder.get_dist_info', return_value=(0, 1))
    def test_single_gpu(self, _):
        common_cfg = dict(
            dataset=self.data,
            samples_per_gpu=self.samples_per_gpu,
            workers_per_gpu=self.workers_per_gpu,
            dist=False)

        # Test default config
        dataloader = build_dataloader(**common_cfg)

        if digit_version(torch.__version__) >= digit_version('1.8.0'):
            assert dataloader.persistent_workers
        elif hasattr(dataloader, 'persistent_workers'):
            assert not dataloader.persistent_workers

        assert dataloader.batch_size == self.samples_per_gpu
        assert dataloader.num_workers == self.workers_per_gpu
        assert not all(
            torch.cat(list(iter(dataloader))) == torch.tensor(self.data))

        # Test without shuffle
        dataloader = build_dataloader(**common_cfg, shuffle=False)
        assert all(
            torch.cat(list(iter(dataloader))) == torch.tensor(self.data))

        # Test with custom sampler_cfg
        dataloader = build_dataloader(
            **common_cfg,
            sampler_cfg=dict(type='RepeatAugSampler', selected_round=0),
            shuffle=False)
        expect = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6]
        assert all(torch.cat(list(iter(dataloader))) == torch.tensor(expect))

    @patch('mmcls.datasets.builder.get_dist_info', return_value=(0, 1))
    def test_multi_gpu(self, _):
        common_cfg = dict(
            dataset=self.data,
            samples_per_gpu=self.samples_per_gpu,
            workers_per_gpu=self.workers_per_gpu,
            num_gpus=2,
            dist=False)

        # Test default config
        dataloader = build_dataloader(**common_cfg)

        if digit_version(torch.__version__) >= digit_version('1.8.0'):
            assert dataloader.persistent_workers
        elif hasattr(dataloader, 'persistent_workers'):
            assert not dataloader.persistent_workers

        assert dataloader.batch_size == self.samples_per_gpu * 2
        assert dataloader.num_workers == self.workers_per_gpu * 2
        assert not all(
            torch.cat(list(iter(dataloader))) == torch.tensor(self.data))

        # Test without shuffle
        dataloader = build_dataloader(**common_cfg, shuffle=False)
        assert all(
            torch.cat(list(iter(dataloader))) == torch.tensor(self.data))

        # Test with custom sampler_cfg
        dataloader = build_dataloader(
            **common_cfg,
            sampler_cfg=dict(type='RepeatAugSampler', selected_round=0),
            shuffle=False)
        expect = torch.tensor(
            [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6])
        assert all(torch.cat(list(iter(dataloader))) == expect)

    @patch('mmcls.datasets.builder.get_dist_info', return_value=(1, 2))
    def test_distributed(self, _):
        common_cfg = dict(
            dataset=self.data,
            samples_per_gpu=self.samples_per_gpu,
            workers_per_gpu=self.workers_per_gpu,
            num_gpus=2,  # num_gpus will be ignored in distributed environment.
            dist=True)

        # Test default config
        dataloader = build_dataloader(**common_cfg)

        if digit_version(torch.__version__) >= digit_version('1.8.0'):
            assert dataloader.persistent_workers
        elif hasattr(dataloader, 'persistent_workers'):
            assert not dataloader.persistent_workers

        assert dataloader.batch_size == self.samples_per_gpu
        assert dataloader.num_workers == self.workers_per_gpu
        non_expect = torch.tensor(self.data[1::2])
        assert not all(torch.cat(list(iter(dataloader))) == non_expect)

        # Test without shuffle
        dataloader = build_dataloader(**common_cfg, shuffle=False)
        expect = torch.tensor(self.data[1::2])
        assert all(torch.cat(list(iter(dataloader))) == expect)

        # Test with custom sampler_cfg
        dataloader = build_dataloader(
            **common_cfg,
            sampler_cfg=dict(type='RepeatAugSampler', selected_round=0),
            shuffle=False)
        expect = torch.tensor(
            [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6][1::2])
        assert all(torch.cat(list(iter(dataloader))) == expect)
