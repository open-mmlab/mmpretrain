import pytest

from mmcls.datasets import ImageFolderDataset


def test_imagefolder_requires_classes():
    with pytest.raises(ValueError, match='must be specified in config file.'):
        ImageFolderDataset(data_prefix='', pipeline=[])
