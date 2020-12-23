import bisect
import math
import random
import string
import tempfile
from collections import defaultdict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mmcls.datasets import (DATASETS, BaseDataset, ClassBalancedDataset,
                            ConcatDataset, RepeatDataset)
from mmcls.datasets.utils import check_integrity, rm_suffix


@pytest.mark.parametrize(
    'dataset_name',
    ['MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100', 'ImageNet'])
def test_datasets_override_default(dataset_name):
    dataset_class = DATASETS.get(dataset_name)
    dataset_class.load_annotations = MagicMock()

    original_classes = dataset_class.CLASSES

    # Test setting classes as a tuple
    dataset = dataset_class(
        data_prefix='', pipeline=[], classes=('bus', 'car'), test_mode=True)
    assert dataset.CLASSES != original_classes
    assert dataset.CLASSES == ('bus', 'car')

    # Test setting classes as a list
    dataset = dataset_class(
        data_prefix='', pipeline=[], classes=['bus', 'car'], test_mode=True)
    assert dataset.CLASSES != original_classes
    assert dataset.CLASSES == ['bus', 'car']

    # Test setting classes through a file
    tmp_file = tempfile.NamedTemporaryFile()
    with open(tmp_file.name, 'w') as f:
        f.write('bus\ncar\n')
    dataset = dataset_class(
        data_prefix='', pipeline=[], classes=tmp_file.name, test_mode=True)
    tmp_file.close()

    assert dataset.CLASSES != original_classes
    assert dataset.CLASSES == ['bus', 'car']

    # Test overriding not a subset
    dataset = dataset_class(
        data_prefix='', pipeline=[], classes=['foo'], test_mode=True)
    assert dataset.CLASSES != original_classes
    assert dataset.CLASSES == ['foo']

    # Test default behavior
    dataset = dataset_class(data_prefix='', pipeline=[])

    assert dataset.data_prefix == ''
    assert not dataset.test_mode
    assert dataset.ann_file is None
    assert dataset.CLASSES == original_classes


@patch.multiple(BaseDataset, __abstractmethods__=set())
def test_dataset_evaluation():
    dataset = BaseDataset(data_prefix='', pipeline=[], test_mode=True)
    dataset.data_infos = [
        dict(gt_label=0),
        dict(gt_label=0),
        dict(gt_label=1),
        dict(gt_label=2),
        dict(gt_label=1),
        dict(gt_label=0)
    ]
    fake_results = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                             [0, 0, 1], [0, 0, 1]])
    eval_results = dataset.evaluate(
        fake_results, metric=['precision', 'recall', 'f1_score'])
    assert eval_results['precision'] == pytest.approx(
        (1 + 1 + 1 / 3) / 3 * 100.0)
    assert eval_results['recall'] == pytest.approx(
        (2 / 3 + 1 / 2 + 1) / 3 * 100.0)
    assert eval_results['f1_score'] == pytest.approx(
        (4 / 5 + 2 / 3 + 1 / 2) / 3 * 100.0)


@patch.multiple(BaseDataset, __abstractmethods__=set())
def test_dataset_wrapper():
    BaseDataset.CLASSES = ('foo', 'bar')
    BaseDataset.__getitem__ = MagicMock(side_effect=lambda idx: idx)
    dataset_a = BaseDataset(data_prefix='', pipeline=[], test_mode=True)
    len_a = 10
    cat_ids_list_a = [
        np.random.randint(0, 80, num).tolist()
        for num in np.random.randint(1, 20, len_a)
    ]
    dataset_a.data_infos = MagicMock()
    dataset_a.data_infos.__len__.return_value = len_a
    dataset_a.get_cat_ids = MagicMock(
        side_effect=lambda idx: cat_ids_list_a[idx])
    dataset_b = BaseDataset(data_prefix='', pipeline=[], test_mode=True)
    len_b = 20
    cat_ids_list_b = [
        np.random.randint(0, 80, num).tolist()
        for num in np.random.randint(1, 20, len_b)
    ]
    dataset_b.data_infos = MagicMock()
    dataset_b.data_infos.__len__.return_value = len_b
    dataset_b.get_cat_ids = MagicMock(
        side_effect=lambda idx: cat_ids_list_b[idx])

    concat_dataset = ConcatDataset([dataset_a, dataset_b])
    assert concat_dataset[5] == 5
    assert concat_dataset[25] == 15
    assert concat_dataset.get_cat_ids(5) == cat_ids_list_a[5]
    assert concat_dataset.get_cat_ids(25) == cat_ids_list_b[15]
    assert len(concat_dataset) == len(dataset_a) + len(dataset_b)
    assert concat_dataset.CLASSES == BaseDataset.CLASSES

    repeat_dataset = RepeatDataset(dataset_a, 10)
    assert repeat_dataset[5] == 5
    assert repeat_dataset[15] == 5
    assert repeat_dataset[27] == 7
    assert repeat_dataset.get_cat_ids(5) == cat_ids_list_a[5]
    assert repeat_dataset.get_cat_ids(15) == cat_ids_list_a[5]
    assert repeat_dataset.get_cat_ids(27) == cat_ids_list_a[7]
    assert len(repeat_dataset) == 10 * len(dataset_a)
    assert repeat_dataset.CLASSES == BaseDataset.CLASSES

    category_freq = defaultdict(int)
    for cat_ids in cat_ids_list_a:
        cat_ids = set(cat_ids)
        for cat_id in cat_ids:
            category_freq[cat_id] += 1
    for k, v in category_freq.items():
        category_freq[k] = v / len(cat_ids_list_a)

    mean_freq = np.mean(list(category_freq.values()))
    repeat_thr = mean_freq

    category_repeat = {
        cat_id: max(1.0, math.sqrt(repeat_thr / cat_freq))
        for cat_id, cat_freq in category_freq.items()
    }

    repeat_factors = []
    for cat_ids in cat_ids_list_a:
        cat_ids = set(cat_ids)
        repeat_factor = max({category_repeat[cat_id] for cat_id in cat_ids})
        repeat_factors.append(math.ceil(repeat_factor))
    repeat_factors_cumsum = np.cumsum(repeat_factors)
    repeat_factor_dataset = ClassBalancedDataset(dataset_a, repeat_thr)
    assert repeat_factor_dataset.CLASSES == BaseDataset.CLASSES
    assert len(repeat_factor_dataset) == repeat_factors_cumsum[-1]
    for idx in np.random.randint(0, len(repeat_factor_dataset), 3):
        assert repeat_factor_dataset[idx] == bisect.bisect_right(
            repeat_factors_cumsum, idx)


def test_dataset_utils():
    # test rm_suffix
    assert rm_suffix('a.jpg') == 'a'
    assert rm_suffix('a.bak.jpg') == 'a.bak'
    assert rm_suffix('a.bak.jpg', suffix='.jpg') == 'a.bak'
    assert rm_suffix('a.bak.jpg', suffix='.bak.jpg') == 'a'

    # test check_integrity
    rand_file = ''.join(random.sample(string.ascii_letters, 10))
    assert not check_integrity(rand_file, md5=None)
    assert not check_integrity(rand_file, md5=2333)
    tmp_file = tempfile.NamedTemporaryFile()
    assert check_integrity(tmp_file.name, md5=None)
    assert not check_integrity(tmp_file.name, md5=2333)
