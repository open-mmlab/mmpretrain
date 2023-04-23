# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import mat4py
from mmengine import get_file_backend

from mmpretrain.registry import DATASETS
from .base_dataset import BaseDataset
from .categories import STANFORD_CARS_CATEGORIES


@DATASETS.register_module()
class StanfordCars(BaseDataset):
    """The Stanford Cars Dataset.

    Support the `Stanford Cars Dataset <https://ai.stanford.edu/~jkrause/cars/car_dataset.html>`_ Dataset.
    The official website provides two ways to organize the dataset.
    Therefore, after downloading and decompression, the dataset directory structure is as follows.

    Stanford Cars dataset directory: ::

        Stanford Cars (data_root)
        ├── car_ims
        │   ├── 00001.jpg
        │   ├── 00002.jpg
        │   └── ...
        └── cars_annos.mat

    or

    Stanford Cars dataset directory: ::

        Stanford Cars (data_root)
        ├── cars_train (data_prefix)
        │   ├── 00001.jpg
        │   ├── 00002.jpg
        │   └── ...
        ├── cars_test (data_prefix)
        │   ├── 00001.jpg
        │   ├── 00002.jpg
        │   └── ...
        └── devkit
            ├── cars_meta.mat
            ├── cars_train_annos.mat
            ├── cars_test_annos.mat
            ├── cars_test_annoswithlabels.mat
            ├── eval_train.m
            └── train_perfect_preds.txt

    Args:
        data_root (str): The root directory for Stanford Cars dataset.
        test_mode (bool): ``test_mode=True`` means in test phase. It determines
             to use the training set or test set. Defaults to False.
        ann_file (str, optional): Annotation file path, path relative to
            ``data_root``. Defaults to 'cars_annos.mat'.
        data_prefix (str): Prefix for images, path relative to
            ``data_root``. Defaults to None.

    Examples:
        >>> # first way
        >>> from mmpretrain.datasets import StanfordCars
        >>> car_train_cfg = dict(data_root='data/Stanford_Cars')
        >>> car_train = StanfordCars(**car_train_cfg)
        >>> car_train
        Dataset StanfordCars
            Number of samples:  8144
            Number of categories:       196
            Root of dataset:    data/Stanford_Cars
        >>> car_test_cfg = dict(data_root='data/Stanford_Cars', test_mode=True)
        >>> car_test = StanfordCars(**car_test_cfg)
        >>> car_test
        Dataset StanfordCars
            Number of samples:  8041
            Number of categories:       196
            Root of dataset:    data/Stanford_Cars

        >>> # second way
        >>> from mmpretrain.datasets import StanfordCars
        >>> car_train_cfg = dict(data_root='data/Stanford_Cars',
        ... ann_file='devkit/cars_train_annos.mat', data_prefix='cars_train')
        >>> car_train = StanfordCars(**car_train_cfg)
        >>> car_train
        Dataset StanfordCars
            Number of samples:  8144
            Number of categories:       196
            Root of dataset:    data/Stanford_Cars
        >>> car_test_cfg = dict(data_root='data/Stanford_Cars', test_mode=True,
        ... ann_file='devkit/cars_test_annos_withlabels.mat', data_prefix='cars_test')
        >>> car_test = StanfordCars(**car_test_cfg)
        >>> car_test
        Dataset StanfordCars
            Number of samples:  8041
            Number of categories:       196
            Root of dataset:    data/Stanford_Cars
    """  # noqa: E501

    METAINFO = {'classes': STANFORD_CARS_CATEGORIES}

    def __init__(self,
                 data_root: str,
                 test_mode: bool = False,
                 ann_file: str = 'cars_annos.mat',
                 data_prefix: str = '',
                 **kwargs):

        self.backend = get_file_backend(data_root, enable_singleton=True)
        super(StanfordCars, self).__init__(
            ann_file=ann_file,
            data_root=data_root,
            data_prefix=data_prefix,
            test_mode=test_mode,
            **kwargs)

    def load_data_list(self):
        data = mat4py.loadmat(self.ann_file)['annotations']

        data_list = []
        if 'test' in data.keys():
            # first way
            img_paths, labels, test = data['relative_im_path'], data[
                'class'], data['test']
            num = len(img_paths)
            assert num == len(labels) == len(test), 'get error ann file'
            for i in range(num):
                if not self.test_mode and test[i] == 1:
                    continue
                if self.test_mode and test[i] == 0:
                    continue
                img_path = self.backend.join_path(self.img_prefix,
                                                  img_paths[i])
                gt_label = labels[i] - 1
                info = dict(img_path=img_path, gt_label=gt_label)
                data_list.append(info)
        else:
            # second way
            img_names, labels = data['fname'], data['class']
            num = len(img_names)
            assert num == len(labels), 'get error ann file'
            for i in range(num):
                img_path = self.backend.join_path(self.img_prefix,
                                                  img_names[i])
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
