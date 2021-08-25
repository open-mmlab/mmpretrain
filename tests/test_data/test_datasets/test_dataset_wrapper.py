# Copyright (c) OpenMMLab. All rights reserved.
import bisect
import math
from collections import defaultdict
from unittest.mock import MagicMock, patch

import numpy as np

from mmcls.datasets import (BaseDataset, ClassBalancedDataset, ConcatDataset,
                            RepeatDataset)


@patch.multiple(BaseDataset, __abstractmethods__=set())
def construct_toy_dataset(length):
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
    return dataset, cat_ids_list


def test_concat_dataset():
    dataset_a, cat_ids_list_a = construct_toy_dataset(10)
    dataset_b, cat_ids_list_b = construct_toy_dataset(20)

    concat_dataset = ConcatDataset([dataset_a, dataset_b])
    assert concat_dataset[5] == 5
    assert concat_dataset[25] == 15
    assert concat_dataset.get_cat_ids(5) == cat_ids_list_a[5]
    assert concat_dataset.get_cat_ids(25) == cat_ids_list_b[15]
    assert len(concat_dataset) == len(dataset_a) + len(dataset_b)
    assert concat_dataset.CLASSES == BaseDataset.CLASSES


def test_repeat_dataset():
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


def test_class_balanced_dataset():
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
