from unittest import mock

import pytest

from mmcls.datasets import DATASETS


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
