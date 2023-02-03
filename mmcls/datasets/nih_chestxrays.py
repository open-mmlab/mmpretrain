# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import pandas as pd
from mmengine import get_file_backend, list_from_file

from mmcls.registry import DATASETS
from .categories import NIHChestXRays_CATEGORIES
from .multi_label import MultiLabelDataset


@DATASETS.register_module()
class NIHChestXRays(MultiLabelDataset):
    """The NIH-ChestXRays Dataset.

    Support the `NIH-ChestXRays <https://nihcc.app.box.com/v/ChestXray-NIHCC>`_ Dataset.

    NIHChestXRays dataset directory: ::

        nih-chestxrays (data_root)/
        ├── images (data_prefix)
        │   ├── xx1.jpg
        │   ├── xx2.jpg
        │   └── ...
        ├── Data_Entry_2017.csv (ann_file)
        ├── test_list.txt (train_test_list_file)
        ├── train_val_list.txt (train_test_list_file)
        └── ....

    Args:
        data_root (str): The root directory for NIH-ChestXRays dataset.
        ann_file (str, optional): Annotation file path, path relative to
            ``data_root``. Defaults to 'images.txt'.
        data_prefix (str): Prefix for iamges, path relative to
            ``data_root``. Defaults to 'images'.
        train_test_list_file (str, optional): The split file  to split train
            and val dataset, path relative to ``data_root``.
            Defaults to 'train_val_list.txt'.


    Examples:
        >>> from mmcls.datasets import CUB
        >>> cub_train_cfg = dict(data_root='data/CUB_200_2011', test_mode=True)
        >>> cub_train = CUB(**cub_train_cfg)
        >>> cub_train
        Dataset CUB
        Number of samples:  5994
        Number of categories:       200
        Root of dataset:    data/CUB_200_2011
        >>> cub_test_cfg = dict(data_root='data/CUB_200_2011', test_mode=True)
        >>> cub_test = CUB(**cub_test_cfg)
        >>> cub_test
        Dataset CUB
        Number of samples:  5794
        Number of categories:       200
        Root of dataset:    data/CUB_200_2011
    """  # noqa: E501

    METAINFO = {'classes': NIHChestXRays_CATEGORIES}

    def __init__(self,
                 data_root: str,
                 ann_file: str = 'Data_Entry_2017.csv',
                 data_prefix: str = 'images',
                 train_test_split_file: str = 'train_val_list.txt',
                 **kwargs):
        self.backend = get_file_backend(data_root, enable_singleton=True)
        self.train_test_split_file = self.backend.join_path(
            data_root, train_test_split_file)
        super(NIHChestXRays, self).__init__(
            ann_file=ann_file,
            data_root=data_root,
            data_prefix=data_prefix,
            **kwargs)

    def load_data_list(self):
        """Load images and ground truth labels."""
        anns = pd.read_csv(self.ann_file)

        filenames = list_from_file(self.train_test_split_file)

        data_list = []
        for img_path in filenames:
            labels = anns[anns['Image Index'] ==
                          img_path]['Finding Labels'].values
            assert len(labels) == 1
            # 'Cardiomegaly|Emphysema' -> ['Cardiomegaly', 'Emphysema']
            labels = labels[0].split('|')
            gt_labels = set()
            for label in labels:
                if label != 'No Finding':
                    gt_labels.add(self.CLASSES.index(label))

            img_path = self.backend.join_path(self.img_prefix, img_path)
            info = dict(img_path=img_path, gt_label=list(gt_labels))
            data_list.append(info)

        return data_list

    def extra_repr(self) -> List[str]:
        """The extra repr information of the dataset."""
        body = [
            f'Root of dataset: \t{self.data_root}',
        ]
        return body
