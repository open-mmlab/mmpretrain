# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

from mmengine import get_file_backend, list_from_file

from mmpretrain.registry import DATASETS
from .base_dataset import BaseDataset
from .categories import SUN397_CATEGORIES


@DATASETS.register_module()
class SUN397(BaseDataset):
    """The SUN397 Dataset.

    Support the `SUN397 Dataset <https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/>`_ Dataset.
    After downloading and decompression, the dataset directory structure is as follows.

    SUN397 dataset directory: ::

        SUN397 (data_root)
        ├── SUN397 (data_prefix)
        │   ├── a
        │   │   ├── abbey
        │   |   |   ├── sun_aaalbzqrimafwbiv.jpg
        │   |   |   └── ...
        │   │   ├── airplane_cabin
        │   |   |   ├── sun_aadqdkqaslqqoblu.jpg
        │   |   |   └── ...
        │   |   └── ...
        │   ├── b
        │   │   └── ...
        │   ├── c
        │   │   └── ...
        │   └── ...
        └── Partitions
            ├── ClassName.txt
            ├── Training_01.txt
            ├── Testing_01.txt
            └── ...

    Args:
        data_root (str): The root directory for Stanford Cars dataset.
        test_mode (bool): ``test_mode=True`` means in test phase. It determines
             to use the training set or test set. Defaults to False.
        ann_file (str): Annotation file path, path relative to
            ``data_root``. Defaults to 'Partitions/Training_01.txt'.
        data_prefix (str): Prefix for images, path relative to
            ``data_root``. Defaults to 'images'.

    Examples:
        >>> from mmpretrain.datasets import SUN397
        >>> train_cfg = dict(data_root='data/SUN397')
        >>> train = SUN397(**train_cfg)
        >>> train
        Dataset SUN397
            Number of samples:  19850
            Number of categories:       397
            Root of dataset:    data/SUN397
        >>> test_cfg = dict(data_root='data/SUN397', test_mode=True,
        ... ann_file='Partitions/Testing_01.txt')
        >>> test = SUN397(**test_cfg)
        >>> test
        Dataset SUN397
            Number of samples:  19850
            Number of categories:       397
            Root of dataset:    data/SUN397
    """  # noqa: E501

    METAINFO = {'classes': SUN397_CATEGORIES}

    def __init__(self,
                 data_root: str,
                 test_mode: bool = False,
                 data_prefix: str = 'SUN397',
                 ann_file: str = 'Partitions/Training_01.txt',
                 **kwargs):

        self.backend = get_file_backend(data_root, enable_singleton=True)

        super(SUN397, self).__init__(
            ann_file=ann_file,
            data_root=data_root,
            test_mode=test_mode,
            data_prefix=data_prefix,
            **kwargs)

    def load_data_list(self):
        pairs = list_from_file(self.ann_file)
        data_list = []
        for pair in pairs:
            items = pair.split('/')
            img_path = self.backend.join_path(self.img_prefix, pair[1:])
            class_name = '/'.join(items[:-1])
            gt_label = self.METAINFO['classes'].index(class_name)
            info = dict(img_path=img_path, gt_label=gt_label)
            data_list.append(info)

        return data_list

    def __getitem__(self, idx: int) -> dict:
        try:
            super(SUN397, self).__getitem__(idx)
        except FileNotFoundError:
            print('pass')

    def extra_repr(self) -> List[str]:
        """The extra repr information of the dataset."""
        body = [
            f'Root of dataset: \t{self.data_root}',
        ]
        return body
