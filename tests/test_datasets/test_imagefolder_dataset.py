import pytest

from mmcls.datasets import ImageFolderDataset


def create_imagefolder_dataset(tmp_path, n_classes=3, samples_per_class=3):
    (tmp_path / 'file_to_skip.txt').touch()
    for n_class in range(n_classes):
        class_folder = tmp_path / f'class_{n_class}'
        class_folder.mkdir()
        for n_sample in range(samples_per_class):
            sample_file = class_folder / f'sample_{n_sample}.jpg'
            sample_file.touch()
            (class_folder / 'sample_to_skip.json').touch()

    return tmp_path


def test_imagefolder_requires_classes():
    with pytest.raises(ValueError, match='must be specified in config file.'):
        ImageFolderDataset(data_prefix='', pipeline=[])


def test_imagefolder_load_annotations_get_samples(tmp_path):
    data_prefix = create_imagefolder_dataset(tmp_path)
    dataset = ImageFolderDataset(
        data_prefix, pipeline=[], classes=('class_0', 'class_1', 'class_2'))
    assert len(dataset.data_infos) == 9


def test_imagefolder_load_annotations_subset(tmp_path):
    data_prefix = create_imagefolder_dataset(tmp_path)
    dataset = ImageFolderDataset(
        data_prefix, pipeline=[], classes=('class_0', 'class_2'))
    assert len(dataset.data_infos) == 6
