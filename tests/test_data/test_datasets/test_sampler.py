# Copyright (c) OpenMMLab. All rights reserved.

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mmcls.apis.train import set_default_sampler_cfg
from mmcls.datasets import BaseDataset, RepeatAugSampler, build_sampler


@patch.multiple(BaseDataset, __abstractmethods__=set())
def construct_toy_single_label_dataset(length):
    BaseDataset.CLASSES = ('foo', 'bar')
    BaseDataset.__getitem__ = MagicMock(side_effect=lambda idx: idx)
    dataset = BaseDataset(data_prefix='', pipeline=[], test_mode=True)
    cat_ids_list = [[np.random.randint(0, 80)] for _ in range(length)]
    dataset.data_infos = MagicMock()
    dataset.data_infos.__len__.return_value = length
    dataset.get_cat_ids = MagicMock(side_effect=lambda idx: cat_ids_list[idx])
    return dataset, cat_ids_list


@patch('mmcls.datasets.samplers.repeat_aug.dist')
def test_sampler_builder(mock_torch_dist):
    assert build_sampler(None) is None
    mock_torch_dist.is_available.side_effect = lambda: True
    mock_torch_dist.get_world_size.side_effect = lambda: 1
    mock_torch_dist.get_rank.side_effect = lambda: 0
    dataset = construct_toy_single_label_dataset(1000)[0]
    build_sampler(dict(type='RepeatAugSampler', dataset=dataset))


def test_default_cfg_setter():
    runner = MagicMock()
    runner.type = 'EpochBasedRunner'
    cfg = dict(runner=runner)
    result = set_default_sampler_cfg(cfg, distributed=False)
    assert result is None
    result = set_default_sampler_cfg(cfg, distributed=True)
    assert result.get('type', None) == 'DistributedSampler'
    assert result.get('shuffle', None)
    assert result.get('round_up', None)

    runner = MagicMock()
    runner.type = 'CustomRunner'
    cfg = dict(runner=runner)
    with pytest.raises(ValueError):
        set_default_sampler_cfg(cfg, distributed=False)


@patch('mmcls.datasets.samplers.repeat_aug.dist')
def test_rep_aug_fail(mock_torch_dist):
    mock_torch_dist.is_available.side_effect = lambda: False
    dataset = construct_toy_single_label_dataset(1000)[0]
    with pytest.raises(Exception):
        build_sampler(dict(type='RepeatAugSampler', dataset=dataset))
    with pytest.raises(Exception):
        RepeatAugSampler(dataset, num_replicas=1)


@patch('mmcls.datasets.samplers.repeat_aug.dist')
def test_rep_aug(mock_torch_dist):
    dataset = construct_toy_single_label_dataset(1000)[0]
    mock_torch_dist.is_available.side_effect = lambda: True
    mock_torch_dist.get_world_size.side_effect = lambda: 1
    mock_torch_dist.get_rank.side_effect = lambda: 0
    ra = RepeatAugSampler(dataset, selected_round=0, shuffle=False)
    ra.set_epoch(0)
    assert len(ra) == 1000
    ra = RepeatAugSampler(dataset)
    assert len(ra) == 768
    val = None
    for idx, content in enumerate(ra):
        if idx % 3 == 0:
            val = content
        else:
            assert val is not None
            assert content == val
