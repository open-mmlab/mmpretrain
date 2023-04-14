# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

from mmengine import get_file_backend, list_from_file

from mmpretrain.registry import DATASETS
from .base_dataset import BaseDataset
from .categories import CUB_CATEGORIES


@DATASETS.register_module()
class CUB(BaseDataset):
    """The CUB-200-2011 Dataset.

    Support the `CUB-200-2011 <http://www.vision.caltech.edu/visipedia/CUB-200-2011.html>`_ Dataset.
    Comparing with the `CUB-200 <http://www.vision.caltech.edu/visipedia/CUB-200.html>`_ Dataset,
    there are much more pictures in `CUB-200-2011`. After downloading and decompression, the dataset
    directory structure is as follows.

    CUB dataset directory: ::

        CUB-200-2011 (data_root)/
        ├── images (data_prefix)
        │   ├── class_x
        │   │   ├── xx1.jpg
        │   │   ├── xx2.jpg
        │   │   └── ...
        │   ├── class_y
        │   │   ├── yy1.jpg
        │   │   ├── yy2.jpg
        │   │   └── ...
        │   └── ...
        ├── images.txt (ann_file)
        ├── image_class_labels.txt (image_class_labels_file)
        ├── train_test_split.txt (train_test_split_file)
        └── ....

    Args:
        data_root (str): The root directory for CUB-200-2011 dataset.
        test_mode (bool): ``test_mode=True`` means in test phase. It determines
             to use the training set or test set.
        ann_file (str, optional): Annotation file path, path relative to
            ``data_root``. Defaults to 'images.txt'.
        data_prefix (str): Prefix for iamges, path relative to
            ``data_root``. Defaults to 'images'.
        image_class_labels_file (str, optional): The label file, path
            relative to ``data_root``. Defaults to 'image_class_labels.txt'.
        train_test_split_file (str, optional): The split file  to split train
            and test dataset, path relative to ``data_root``.
            Defaults to 'train_test_split_file.txt'.


    Examples:
        >>> from mmpretrain.datasets import CUB
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

    METAINFO = {'classes': CUB_CATEGORIES}

    def __init__(self,
                 data_root: str,
                 test_mode: bool,
                 ann_file: str = 'images.txt',
                 data_prefix: str = 'images',
                 image_class_labels_file: str = 'image_class_labels.txt',
                 train_test_split_file: str = 'train_test_split.txt',
                 **kwargs):
        self.backend = get_file_backend(data_root, enable_singleton=True)
        self.image_class_labels_file = self.backend.join_path(
            data_root, image_class_labels_file)
        self.train_test_split_file = self.backend.join_path(
            data_root, train_test_split_file)
        super(CUB, self).__init__(
            ann_file=ann_file,
            data_root=data_root,
            data_prefix=data_prefix,
            test_mode=test_mode,
            **kwargs)

    def _load_data_from_txt(self, filepath):
        """load data from CUB txt file, the every line of the file is idx and a
        data item."""
        pairs = list_from_file(filepath)
        data_dict = dict()
        for pair in pairs:
            idx, data_item = pair.split()
            # all the index starts from 1 in CUB files,
            # here we need to '- 1' to let them start from 0.
            data_dict[int(idx) - 1] = data_item
        return data_dict

    def load_data_list(self):
        """Load images and ground truth labels."""
        sample_dict = self._load_data_from_txt(self.ann_file)

        label_dict = self._load_data_from_txt(self.image_class_labels_file)

        split_dict = self._load_data_from_txt(self.train_test_split_file)

        assert sample_dict.keys() == label_dict.keys() == split_dict.keys(),\
            f'sample_ids should be same in files {self.ann_file}, ' \
            f'{self.image_class_labels_file} and {self.train_test_split_file}'

        data_list = []
        for sample_id in sample_dict.keys():
            if split_dict[sample_id] == '1' and self.test_mode:
                # skip train samples when test_mode=True
                continue
            elif split_dict[sample_id] == '0' and not self.test_mode:
                # skip test samples when test_mode=False
                continue

            img_path = self.backend.join_path(self.img_prefix,
                                              sample_dict[sample_id])
            gt_label = int(label_dict[sample_id]) - 1
            info = dict(img_path=img_path, gt_label=gt_label)
            data_list.append(info)

        return data_list

    def extra_repr(self) -> List[str]:
        """The extra repr information of the dataset."""
        body = [
            f'Root of dataset: \t{self.data_root}',
        ]
        return body
