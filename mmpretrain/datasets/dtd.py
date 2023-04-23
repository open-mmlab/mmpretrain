# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import mat4py
from mmengine import get_file_backend

from mmpretrain.registry import DATASETS
from .base_dataset import BaseDataset
from .categories import DTD_CATEGORIES


@DATASETS.register_module()
class DTD(BaseDataset):
    """The Describable Texture Dataset (DTD).

    Support the `Describable Texture Dataset <https://www.robots.ox.ac.uk/~vgg/data/dtd/>`_ Dataset.
    After downloading and decompression, the dataset directory structure is as follows.

    DTD dataset directory: ::

        dtd (data_root)/
        ├── images (data_prefix)
        │   ├── banded
        |   |   ├──banded_0002.jpg
        |   |   ├──banded_0004.jpg
        |   |   └── ...
        │   └── ...
        ├── imdb
        │   └── imdb.mat
        ├── labels
        |   |   ├──labels_joint_anno.txt
        |   |   ├──test1.txt
        |   |   ├──test2.txt
        |   |   └── ...
        │   └── ...
        └── ....

    Args:
        data_root (str): The root directory for Describable Texture dataset.
        mode (str): The part of the dataset. The value can be chosen in the 'train',
            'val', 'trainval', and 'test'. Defaults to 'trainval'.
        ann_file (str, optional): Annotation file path, path relative to
            ``data_root``. Defaults to 'imdb/imdb.mat'.
        data_prefix (str): Prefix for images, path relative to
            ``data_root``. Defaults to 'jpg'.

    Examples:
        >>> from mmpretrain.datasets import DTD
        >>> dtd_train_cfg = dict(data_root='data/dtd', mode='trainval')
        >>> dtd_train = DTD(**dtd_train_cfg)
        >>> dtd_train
        Dataset DTD
            Number of samples:  3760
            Number of categories:       47
            Root of dataset:    data/dtd
        >>> dtd_test_cfg = dict(data_root='data/dtd', mode='test')
        >>> dtd_test = DTD(**dtd_test_cfg)
        >>> dtd_test
        Dataset DTD
            Number of samples:  1880
            Number of categories:       47
            Root of dataset:    data/dtd
    """  # noqa: E501

    METAINFO = {'classes': DTD_CATEGORIES}

    def __init__(self,
                 data_root: str,
                 mode: str = 'trainval',
                 ann_file: str = 'imdb/imdb.mat',
                 data_prefix: str = 'images',
                 **kwargs):
        modes = ['train', 'val', 'trainval', 'test']
        assert mode in modes, f'Mode {mode} is not in default modes {modes}'
        self.backend = get_file_backend(data_root, enable_singleton=True)
        self.mode = mode
        test_mode = mode == 'test'
        super(DTD, self).__init__(
            ann_file=ann_file,
            data_root=data_root,
            data_prefix=data_prefix,
            test_mode=test_mode,
            **kwargs)

    def load_data_list(self):
        """Load images and ground truth labels."""

        data = mat4py.loadmat(self.ann_file)['images']
        names = data['name']
        labels = data['class']
        parts = data['set']
        num = len(names)
        assert num == len(labels) == len(parts), 'get error ann file'

        if self.mode == 'train':
            target_set = {1}
        elif self.mode == 'val':
            target_set = {2}
        elif self.mode == 'test':
            target_set = {3}
        else:
            target_set = {1, 2}

        data_list = []
        for i in range(num):
            if parts[i] in target_set:
                img_name = names[i]
                img_path = self.backend.join_path(self.img_prefix, img_name)
                gt_label = labels[i] - 1
                info = dict(img_path=img_path, gt_label=gt_label)
                data_list.append(info)

        return data_list

    def extra_repr(self) -> List[str]:
        """The extra repr information of the dataset."""
        body = [
            f'Root of dataset: \t{self.data_root}',
        ]
        return body
