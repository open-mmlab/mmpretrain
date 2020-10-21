from unittest import mock

import pytest

from mmcls.datasets import DATASETS, BaseDataset


@pytest.mark.parametrize('dataset_name', [
    'MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100', 'ImageNet',
    'ImageFolderDataset'
])
def test_datasets_override_default(dataset_name):
    dataset_class = DATASETS.get(dataset_name)
    with mock.patch.object(
            dataset_class, 'load_annotations', new=mock.MagicMock):
        # Test default behavior
        dataset = dataset_class(data_prefix='', pipeline=[])

        assert dataset.data_prefix == ''
        assert not dataset.test_mode
        assert dataset.ann_file is None


@mock.patch.multiple(BaseDataset, __abstractmethods__=set())
def test_custom_classes_not_provided():
    dataset = BaseDataset(data_prefix='', pipeline=[])
    assert dataset.CLASSES is None

    with mock.patch.object(BaseDataset, 'CLASSES', new=('foo', 'bar')):
        dataset = BaseDataset(data_prefix='', pipeline=[])
        assert dataset.CLASSES == ('foo', 'bar')


@mock.patch.multiple(BaseDataset, __abstractmethods__=set())
def test_custom_classes_provided_as_tuple():
    dataset = BaseDataset(data_prefix='', pipeline=[], classes=('foo', 'bar'))
    assert dataset.CLASSES == ('foo', 'bar')


@mock.patch.multiple(BaseDataset, __abstractmethods__=set())
def test_custom_classes_provided_as_str(tmp_path):
    classes_path = tmp_path / 'classes.txt'
    classes_path.write_text('foo\nbar')
    dataset = BaseDataset(
        data_prefix='', pipeline=[], classes=str(classes_path))
    assert dataset.CLASSES == ['foo', 'bar']


@mock.patch.multiple(BaseDataset, __abstractmethods__=set())
def test_custom_classes_provided_with_wrong_type(tmp_path):
    with pytest.raises(TypeError, match='Unsupported type'):
        BaseDataset(data_prefix='', pipeline=[], classes={'foo': 0, 'bar': 1})


@pytest.mark.parametrize('dataset_name', [
    'MNIST',
    'FashionMNIST',
    'CIFAR10',
    'CIFAR100',
    'ImageNet',
])
def test_custom_classes_subset_of_predefined_datasets(dataset_name):
    dataset_class = DATASETS.get(dataset_name)
    class_0 = dataset_class.CLASSES[0]
    class_2 = dataset_class.CLASSES[2]
    with mock.patch.object(
            dataset_class,
            'load_annotations',
            return_value=[{
                'gt_label': 0
            }, {
                'gt_label': 1,
                'filtered': True
            }, {
                'gt_label': 2
            }]):
        dataset = dataset_class(
            data_prefix='', pipeline=[], classes=(class_0, class_2))
        assert dataset.CLASSES == (class_0, class_2)
        assert dataset.original_idx_to_subset_idx == {0: 0, 2: 1}
        # original {"gt_label": 1} is filtered
        # original {"gt_label": 2} is converted to {"gt_label": 1}
        assert dataset.data_infos == [{'gt_label': 0}, {'gt_label': 1}]
        assert dataset.class_to_idx == {class_0: 0, class_2: 1}


@pytest.mark.parametrize('dataset_name', [
    'MNIST',
    'FashionMNIST',
    'CIFAR10',
    'CIFAR100',
    'ImageNet',
])
def test_custom_classes_invalid_overwrite(dataset_name):
    dataset_class = DATASETS.get(dataset_name)
    with pytest.raises(
            ValueError, match='not found in original dataset.CLASSES'):
        dataset_class(
            data_prefix='', pipeline=[], classes=('invalid_0', 'invalid_1'))
