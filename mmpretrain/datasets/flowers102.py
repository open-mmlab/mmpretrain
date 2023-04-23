# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import mat4py
from mmengine import get_file_backend

from mmpretrain.registry import DATASETS
from .base_dataset import BaseDataset


@DATASETS.register_module()
class Flowers102(BaseDataset):
    """The Oxford 102 Flower Dataset.

    Support the `Oxford 102 Flowers Dataset <https://www.robots.ox.ac.uk/~vgg/data/flowers/102/>`_ Dataset.
    After downloading and decompression, the dataset directory structure is as follows.

    Flowers102 dataset directory: ::

        Flowers102 (data_root)/
        ├── jpg (data_prefix)
        │   ├── image_00001.jpg
        │   ├── image_00002.jpg
        │   └── ...
        ├── imagelabels.mat (ann_file)
        ├── setid.mat (train_test_split_file)
        └── ...

    Args:
        data_root (str): The root directory for Oxford 102 Flowers dataset.
        mode (str): The part of the dataset. The value can be chosen in the 'train',
            'val', 'trainval', and 'test'. Defaults to 'trainval'.
        ann_file (str, optional): Annotation file path, path relative to
            ``data_root``. Defaults to 'imagelabels.mat'.
        data_prefix (str): Prefix for images, path relative to
            ``data_root``. Defaults to 'jpg'.
        train_test_split_file (str): The split file  to split train
            and test dataset, path relative to ``data_root``.
            Defaults to 'setid.mat'.

    Examples:
        >>> from mmpretrain.datasets import Flowers102
        >>> flower_train_cfg = dict(data_root='data/Flowers102', mode='trainval')
        >>> flower_train = Flowers102(**flower_train_cfg)
        >>> flower_train
        Dataset Flowers102
            Number of samples:  2040
            Root of dataset:    data/Flowers102
        >>> flower_test_cfg = dict(data_root='data/Flowers102', mode='test')
        >>> flower_test = Flowers102(**flower_test_cfg)
        >>> flower_test
        Dataset Flowers102
            Number of samples:  6149
            Root of dataset:    data/Flowers102
    """  # noqa: E501

    def __init__(self,
                 data_root: str,
                 mode: str = 'trainval',
                 ann_file: str = 'imagelabels.mat',
                 data_prefix: str = 'jpg',
                 train_test_split_file: str = 'setid.mat',
                 **kwargs):
        modes = ['train', 'val', 'trainval', 'test']
        assert mode in modes, f'Mode {mode} is not in default modes {modes}'
        self.backend = get_file_backend(data_root, enable_singleton=True)
        self.mode = mode
        test_mode = mode == 'test'
        self.train_test_split_file = self.backend.join_path(
            data_root, train_test_split_file)
        super(Flowers102, self).__init__(
            ann_file=ann_file,
            data_root=data_root,
            data_prefix=data_prefix,
            test_mode=test_mode,
            **kwargs)

    def load_data_list(self):
        """Load images and ground truth labels."""

        label_dict = mat4py.loadmat(self.ann_file)['labels']
        split_list = mat4py.loadmat(self.train_test_split_file)

        if self.mode == 'train':
            split_list = split_list['trnid']
        elif self.mode == 'val':
            split_list = split_list['valid']
        elif self.mode == 'test':
            split_list = split_list['tstid']
        else:
            train_ids = split_list['trnid']
            val_ids = split_list['valid']
            train_ids.extend(val_ids)
            split_list = train_ids

        data_list = []
        for sample_id in split_list:
            img_name = 'image_%05d.jpg' % (sample_id)
            img_path = self.backend.join_path(self.img_prefix, img_name)
            gt_label = int(label_dict[sample_id - 1]) - 1
            info = dict(img_path=img_path, gt_label=gt_label)
            data_list.append(info)

        return data_list

    def extra_repr(self) -> List[str]:
        """The extra repr information of the dataset."""
        body = [
            f'Root of dataset: \t{self.data_root}',
        ]
        return body
