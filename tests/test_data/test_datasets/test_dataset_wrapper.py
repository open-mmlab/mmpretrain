# Copyright (c) OpenMMLab. All rights reserved.
import bisect
import math
from collections import defaultdict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mmcls.datasets import (BaseDataset, ClassBalancedDataset, ConcatDataset,
                            KFoldDataset, RepeatDataset)


def mock_evaluate(results,
                  metric='accuracy',
                  metric_options=None,
                  indices=None,
                  logger=None):
    return dict(
        results=results,
        metric=metric,
        metric_options=metric_options,
        indices=indices,
        logger=logger)


@patch.multiple(BaseDataset, __abstractmethods__=set())
def construct_toy_multi_label_dataset(length):
    BaseDataset.CLASSES = ('foo', 'bar')
    BaseDataset.__getitem__ = MagicMock(side_effect=lambda idx: idx)
    dataset = BaseDataset(data_prefix='', pipeline=[], test_mode=True)
    cat_ids_list = [
        np.random.randint(0, 80, num).tolist()
        for num in np.random.randint(1, 20, length)
    ]
    dataset.data_infos = MagicMock()
    dataset.data_infos.__len__.return_value = length
    dataset.get_cat_ids = MagicMock(side_effect=lambda idx: cat_ids_list[idx])
    dataset.get_gt_labels = \
        MagicMock(side_effect=lambda: np.array(cat_ids_list))
    dataset.evaluate = MagicMock(side_effect=mock_evaluate)
    return dataset, cat_ids_list


@patch.multiple(BaseDataset, __abstractmethods__=set())
def construct_toy_single_label_dataset(length):
    BaseDataset.CLASSES = ('foo', 'bar')
    BaseDataset.__getitem__ = MagicMock(side_effect=lambda idx: idx)
    dataset = BaseDataset(data_prefix='', pipeline=[], test_mode=True)
    cat_ids_list = [[np.random.randint(0, 80)] for _ in range(length)]
    dataset.data_infos = MagicMock()
    dataset.data_infos.__len__.return_value = length
    dataset.get_cat_ids = MagicMock(side_effect=lambda idx: cat_ids_list[idx])
    dataset.get_gt_labels = \
        MagicMock(side_effect=lambda: np.array(cat_ids_list))
    dataset.evaluate = MagicMock(side_effect=mock_evaluate)
    return dataset, cat_ids_list


@pytest.mark.parametrize('construct_dataset', [
    'construct_toy_multi_label_dataset', 'construct_toy_single_label_dataset'
])
def test_concat_dataset(construct_dataset):
    construct_toy_dataset = eval(construct_dataset)
    dataset_a, cat_ids_list_a = construct_toy_dataset(10)
    dataset_b, cat_ids_list_b = construct_toy_dataset(20)

    concat_dataset = ConcatDataset([dataset_a, dataset_b])
    assert concat_dataset[5] == 5
    assert concat_dataset[25] == 15
    assert concat_dataset.get_cat_ids(5) == cat_ids_list_a[5]
    assert concat_dataset.get_cat_ids(25) == cat_ids_list_b[15]
    assert len(concat_dataset) == len(dataset_a) + len(dataset_b)
    assert concat_dataset.CLASSES == BaseDataset.CLASSES


@pytest.mark.parametrize('construct_dataset', [
    'construct_toy_multi_label_dataset', 'construct_toy_single_label_dataset'
])
def test_repeat_dataset(construct_dataset):
    construct_toy_dataset = eval(construct_dataset)
    dataset, cat_ids_list = construct_toy_dataset(10)
    repeat_dataset = RepeatDataset(dataset, 10)
    assert repeat_dataset[5] == 5
    assert repeat_dataset[15] == 5
    assert repeat_dataset[27] == 7
    assert repeat_dataset.get_cat_ids(5) == cat_ids_list[5]
    assert repeat_dataset.get_cat_ids(15) == cat_ids_list[5]
    assert repeat_dataset.get_cat_ids(27) == cat_ids_list[7]
    assert len(repeat_dataset) == 10 * len(dataset)
    assert repeat_dataset.CLASSES == BaseDataset.CLASSES


@pytest.mark.parametrize('construct_dataset', [
    'construct_toy_multi_label_dataset', 'construct_toy_single_label_dataset'
])
def test_class_balanced_dataset(construct_dataset):
    construct_toy_dataset = eval(construct_dataset)
    dataset, cat_ids_list = construct_toy_dataset(10)

    category_freq = defaultdict(int)
    for cat_ids in cat_ids_list:
        cat_ids = set(cat_ids)
        for cat_id in cat_ids:
            category_freq[cat_id] += 1
    for k, v in category_freq.items():
        category_freq[k] = v / len(cat_ids_list)

    mean_freq = np.mean(list(category_freq.values()))
    repeat_thr = mean_freq

    category_repeat = {
        cat_id: max(1.0, math.sqrt(repeat_thr / cat_freq))
        for cat_id, cat_freq in category_freq.items()
    }

    repeat_factors = []
    for cat_ids in cat_ids_list:
        cat_ids = set(cat_ids)
        repeat_factor = max({category_repeat[cat_id] for cat_id in cat_ids})
        repeat_factors.append(math.ceil(repeat_factor))
    repeat_factors_cumsum = np.cumsum(repeat_factors)
    repeat_factor_dataset = ClassBalancedDataset(dataset, repeat_thr)
    assert repeat_factor_dataset.CLASSES == BaseDataset.CLASSES
    assert len(repeat_factor_dataset) == repeat_factors_cumsum[-1]
    for idx in np.random.randint(0, len(repeat_factor_dataset), 3):
        assert repeat_factor_dataset[idx] == bisect.bisect_right(
            repeat_factors_cumsum, idx)


@pytest.mark.parametrize('construct_dataset', [
    'construct_toy_multi_label_dataset', 'construct_toy_single_label_dataset'
])
def test_kfold_dataset(construct_dataset):
    construct_toy_dataset = eval(construct_dataset)
    dataset, cat_ids_list = construct_toy_dataset(10)

    # test without random seed
    train_datasets = [
        KFoldDataset(dataset, fold=i, num_splits=3, test_mode=False)
        for i in range(5)
    ]
    test_datasets = [
        KFoldDataset(dataset, fold=i, num_splits=3, test_mode=True)
        for i in range(5)
    ]

    assert sum([i.indices for i in test_datasets], []) == list(range(10))
    for train_set, test_set in zip(train_datasets, test_datasets):
        train_samples = [train_set[i] for i in range(len(train_set))]
        test_samples = [test_set[i] for i in range(len(test_set))]
        assert set(train_samples + test_samples) == set(range(10))

    # test with random seed
    train_datasets = [
        KFoldDataset(dataset, fold=i, num_splits=3, test_mode=False, seed=1)
        for i in range(5)
    ]
    test_datasets = [
        KFoldDataset(dataset, fold=i, num_splits=3, test_mode=True, seed=1)
        for i in range(5)
    ]

    assert sum([i.indices for i in test_datasets], []) != list(range(10))
    assert set(sum([i.indices for i in test_datasets], [])) == set(range(10))
    for train_set, test_set in zip(train_datasets, test_datasets):
        train_samples = [train_set[i] for i in range(len(train_set))]
        test_samples = [test_set[i] for i in range(len(test_set))]
        assert set(train_samples + test_samples) == set(range(10))

    # test behavior of get_cat_ids method
    for train_set, test_set in zip(train_datasets, test_datasets):
        for i in range(len(train_set)):
            cat_ids = train_set.get_cat_ids(i)
            assert cat_ids == cat_ids_list[train_set.indices[i]]
        for i in range(len(test_set)):
            cat_ids = test_set.get_cat_ids(i)
            assert cat_ids == cat_ids_list[test_set.indices[i]]

    # test behavior of get_gt_labels method
    for train_set, test_set in zip(train_datasets, test_datasets):
        for i in range(len(train_set)):
            gt_label = train_set.get_gt_labels()[i]
            assert gt_label == cat_ids_list[train_set.indices[i]]
        for i in range(len(test_set)):
            gt_label = test_set.get_gt_labels()[i]
            assert gt_label == cat_ids_list[test_set.indices[i]]

    # test evaluate
    for test_set in test_datasets:
        eval_inputs = test_set.evaluate(None)
        assert eval_inputs['indices'] == test_set.indices
