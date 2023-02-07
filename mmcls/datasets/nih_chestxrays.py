# Copyright (c) OpenMMLab. All rights reserved.
import csv
from io import StringIO
from typing import List

from mmengine import fileio, get_file_backend, list_from_file

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
        >>> from mmcls.datasets import NIHChestXRays
        >>> nih_train_cfg = dict(data_root='data/nih-chestxrays',
        >>>                      train_test_split_file='train_val_list.txt')
        >>> nih_train = NIHChestXRays(**nih_train_cfg)
        >>> nih_train
        Dataset NIHChestXRays
            Number of samples:  86524
            Number of categories:       14
            Root of dataset:    data/nih-chestxrays
        >>> nih_test_cfg = dict(data_root='data/nih-chestxrays',
        >>>                     train_test_split_file='test_list.txt')
        >>> nih_test = NIHChestXRays(**nih_test_cfg)
        >>> nih_test
        Dataset NIHChestXRays
            Number of samples:  25596
            Number of categories:       14
            Root of dataset:    data/nih-chestxrays
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

    def _load_csv_ann(self, csv_file):
        ann_dict = dict()
        data = fileio.get(csv_file)
        data = StringIO(data.decode('utf-8'), newline='')
        csv_reader = csv.reader(data)

        for row in csv_reader:
            # the format of the csv file:
            # Image Index, Finding Labels, .......
            ann_dict[row[0]] = row[1]
        return ann_dict

    def load_data_list(self):
        """Load images and ground truth labels."""
        anns = self._load_csv_ann(self.ann_file)
        filenames = list_from_file(self.train_test_split_file)

        data_list = []
        for img_path in filenames:
            # finding_labels are joined by '|' like fllowing:
            # 'Cardiomegaly|Emphysema' -> ['Cardiomegaly', 'Emphysema']
            finding_labels = anns[img_path].split('|')
            gt_labels = set()
            for label in finding_labels:
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
