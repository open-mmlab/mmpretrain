# Copyright (c) OpenMMLab. All rights reserved.
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from mmcls.datasets import DATASETS, BaseDataset, MultiLabelDataset


@pytest.mark.parametrize(
    'dataset_name',
    ['MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100', 'ImageNet', 'VOC'])
def test_datasets_override_default(dataset_name):
    dataset_class = DATASETS.get(dataset_name)
    dataset_class.load_annotations = MagicMock()

    original_classes = dataset_class.CLASSES

    # Test VOC year
    if dataset_name == 'VOC':
        dataset = dataset_class(
            data_prefix='VOC2007',
            pipeline=[],
            classes=('bus', 'car'),
            test_mode=True)
        assert dataset.year == 2007
        with pytest.raises(ValueError):
            dataset = dataset_class(
                data_prefix='VOC',
                pipeline=[],
                classes=('bus', 'car'),
                test_mode=True)

    # Test setting classes as a tuple
    dataset = dataset_class(
        data_prefix='VOC2007' if dataset_name == 'VOC' else '',
        pipeline=[],
        classes=('bus', 'car'),
        test_mode=True)
    assert dataset.CLASSES == ('bus', 'car')

    # Test setting classes as a list
    dataset = dataset_class(
        data_prefix='VOC2007' if dataset_name == 'VOC' else '',
        pipeline=[],
        classes=['bus', 'car'],
        test_mode=True)
    assert dataset.CLASSES == ['bus', 'car']

    # Test setting classes through a file
    tmp_file = tempfile.NamedTemporaryFile()
    with open(tmp_file.name, 'w') as f:
        f.write('bus\ncar\n')
    dataset = dataset_class(
        data_prefix='VOC2007' if dataset_name == 'VOC' else '',
        pipeline=[],
        classes=tmp_file.name,
        test_mode=True)
    tmp_file.close()

    assert dataset.CLASSES == ['bus', 'car']

    # Test overriding not a subset
    dataset = dataset_class(
        data_prefix='VOC2007' if dataset_name == 'VOC' else '',
        pipeline=[],
        classes=['foo'],
        test_mode=True)
    assert dataset.CLASSES == ['foo']

    # Test default behavior
    dataset = dataset_class(
        data_prefix='VOC2007' if dataset_name == 'VOC' else '', pipeline=[])

    if dataset_name == 'VOC':
        assert dataset.data_prefix == 'VOC2007'
    else:
        assert dataset.data_prefix == ''
    assert not dataset.test_mode
    assert dataset.ann_file is None
    assert dataset.CLASSES == original_classes


@patch.multiple(MultiLabelDataset, __abstractmethods__=set())
@patch.multiple(BaseDataset, __abstractmethods__=set())
def test_dataset_evaluation():
    # test multi-class single-label evaluation
    dataset = BaseDataset(data_prefix='', pipeline=[], test_mode=True)
    dataset.data_infos = [
        dict(gt_label=0),
        dict(gt_label=0),
        dict(gt_label=1),
        dict(gt_label=2),
        dict(gt_label=1),
        dict(gt_label=0)
    ]
    fake_results = np.array([[0.7, 0, 0.3], [0.5, 0.2, 0.3], [0.4, 0.5, 0.1],
                             [0, 0, 1], [0, 0, 1], [0, 0, 1]])
    eval_results = dataset.evaluate(
        fake_results,
        metric=['precision', 'recall', 'f1_score', 'support', 'accuracy'],
        metric_options={'topk': 1})
    assert eval_results['precision'] == pytest.approx(
        (1 + 1 + 1 / 3) / 3 * 100.0)
    assert eval_results['recall'] == pytest.approx(
        (2 / 3 + 1 / 2 + 1) / 3 * 100.0)
    assert eval_results['f1_score'] == pytest.approx(
        (4 / 5 + 2 / 3 + 1 / 2) / 3 * 100.0)
    assert eval_results['support'] == 6
    assert eval_results['accuracy'] == pytest.approx(4 / 6 * 100)

    # test input as tensor
    fake_results_tensor = torch.from_numpy(fake_results)
    eval_results_ = dataset.evaluate(
        fake_results_tensor,
        metric=['precision', 'recall', 'f1_score', 'support', 'accuracy'],
        metric_options={'topk': 1})
    assert eval_results_ == eval_results

    # test thr
    eval_results = dataset.evaluate(
        fake_results,
        metric=['precision', 'recall', 'f1_score', 'accuracy'],
        metric_options={
            'thrs': 0.6,
            'topk': 1
        })
    assert eval_results['precision'] == pytest.approx(
        (1 + 0 + 1 / 3) / 3 * 100.0)
    assert eval_results['recall'] == pytest.approx((1 / 3 + 0 + 1) / 3 * 100.0)
    assert eval_results['f1_score'] == pytest.approx(
        (1 / 2 + 0 + 1 / 2) / 3 * 100.0)
    assert eval_results['accuracy'] == pytest.approx(2 / 6 * 100)
    # thrs must be a number or tuple
    with pytest.raises(TypeError):
        eval_results = dataset.evaluate(
            fake_results,
            metric=['precision', 'recall', 'f1_score', 'accuracy'],
            metric_options={
                'thrs': 'thr',
                'topk': 1
            })

    # test topk and thr as tuple
    eval_results = dataset.evaluate(
        fake_results,
        metric=['precision', 'recall', 'f1_score', 'accuracy'],
        metric_options={
            'thrs': (0.5, 0.6),
            'topk': (1, 2)
        })
    assert {
        'precision_thr_0.50', 'precision_thr_0.60', 'recall_thr_0.50',
        'recall_thr_0.60', 'f1_score_thr_0.50', 'f1_score_thr_0.60',
        'accuracy_top-1_thr_0.50', 'accuracy_top-1_thr_0.60',
        'accuracy_top-2_thr_0.50', 'accuracy_top-2_thr_0.60'
    } == eval_results.keys()
    assert type(eval_results['precision_thr_0.50']) == float
    assert type(eval_results['recall_thr_0.50']) == float
    assert type(eval_results['f1_score_thr_0.50']) == float
    assert type(eval_results['accuracy_top-1_thr_0.50']) == float

    eval_results = dataset.evaluate(
        fake_results,
        metric='accuracy',
        metric_options={
            'thrs': 0.5,
            'topk': (1, 2)
        })
    assert {'accuracy_top-1', 'accuracy_top-2'} == eval_results.keys()
    assert type(eval_results['accuracy_top-1']) == float

    eval_results = dataset.evaluate(
        fake_results,
        metric='accuracy',
        metric_options={
            'thrs': (0.5, 0.6),
            'topk': 1
        })
    assert {'accuracy_thr_0.50', 'accuracy_thr_0.60'} == eval_results.keys()
    assert type(eval_results['accuracy_thr_0.50']) == float

    # test evaluation results for classes
    eval_results = dataset.evaluate(
        fake_results,
        metric=['precision', 'recall', 'f1_score', 'support'],
        metric_options={'average_mode': 'none'})
    assert eval_results['precision'].shape == (3, )
    assert eval_results['recall'].shape == (3, )
    assert eval_results['f1_score'].shape == (3, )
    assert eval_results['support'].shape == (3, )

    # the average_mode method must be valid
    with pytest.raises(ValueError):
        eval_results = dataset.evaluate(
            fake_results,
            metric='precision',
            metric_options={'average_mode': 'micro'})
    with pytest.raises(ValueError):
        eval_results = dataset.evaluate(
            fake_results,
            metric='recall',
            metric_options={'average_mode': 'micro'})
    with pytest.raises(ValueError):
        eval_results = dataset.evaluate(
            fake_results,
            metric='f1_score',
            metric_options={'average_mode': 'micro'})
    with pytest.raises(ValueError):
        eval_results = dataset.evaluate(
            fake_results,
            metric='support',
            metric_options={'average_mode': 'micro'})

    # the metric must be valid for the dataset
    with pytest.raises(ValueError):
        eval_results = dataset.evaluate(fake_results, metric='map')

    # test multi-label evalutation
    dataset = MultiLabelDataset(data_prefix='', pipeline=[], test_mode=True)
    dataset.data_infos = [
        dict(gt_label=[1, 1, 0, -1]),
        dict(gt_label=[1, 1, 0, -1]),
        dict(gt_label=[0, -1, 1, -1]),
        dict(gt_label=[0, 1, 0, -1]),
        dict(gt_label=[0, 1, 0, -1]),
    ]
    fake_results = np.array([[0.9, 0.8, 0.3, 0.2], [0.1, 0.2, 0.2, 0.1],
                             [0.7, 0.5, 0.9, 0.3], [0.8, 0.1, 0.1, 0.2],
                             [0.8, 0.1, 0.1, 0.2]])

    # the metric must be valid
    with pytest.raises(ValueError):
        metric = 'coverage'
        dataset.evaluate(fake_results, metric=metric)
    # only one metric
    metric = 'mAP'
    eval_results = dataset.evaluate(fake_results, metric=metric)
    assert 'mAP' in eval_results.keys()
    assert 'CP' not in eval_results.keys()

    # multiple metrics
    metric = ['mAP', 'CR', 'OF1']
    eval_results = dataset.evaluate(fake_results, metric=metric)
    assert 'mAP' in eval_results.keys()
    assert 'CR' in eval_results.keys()
    assert 'OF1' in eval_results.keys()
    assert 'CF1' not in eval_results.keys()
