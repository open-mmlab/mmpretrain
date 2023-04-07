# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

from mmengine import get_file_backend, list_from_file

from mmpretrain.registry import DATASETS
from .base_dataset import BaseDataset
from .categories import OXFORD_IIIT_PETS_CATEGORIES


@DATASETS.register_module()
class Oxford_IIIT_Pets(BaseDataset):
    """The Oxford-IIIT Pets Dataset.

    Support the `Oxford-IIIT Pets Dataset <https://www.robots.ox.ac.uk/~vgg/data/pets/>`_ Dataset.
    After downloading and decompression, the dataset directory structure is as follows.

    Oxford_IIIT-Pets dataset directory: ::

        Oxford-IIIT_Pets (data_root)/
        ├── images (data_prefix)
        │   ├── Abyssinian_1.jpg
        │   ├── Abyssinian_2.jpg
        │   └── ...
        ├── annotations
        │   ├── trainval.txt
        │   ├── test.txt
        │   ├── list.txt
        │   └── ...
        └── ....

    Args:
        data_root (str): The root directory for Oxford-IIIT Pets dataset.
        test_mode (bool): ``test_mode=True`` means in test phase. It determines
             to use the training set or test set.
        ann_file (str, optional): Annotation file path, path relative to
            ``data_root``. Defaults to 'annotations/trainval.txt'.
        data_prefix (str): Prefix for images, path relative to
            ``data_root``. Defaults to 'images'.

    Examples:
        >>> from mmpretrain.datasets import Oxford_IIIT_Pets
        >>> pet_train_cfg = dict(data_root='data/Oxford-IIIT_Pets')
        >>> pet_train = Oxford_IIIT_Pets(**pet_train_cfg)
        >>> pet_train
        Dataset Oxford_IIIT_Pets
            Number of samples:  3680
            Number of categories:       37
            Root of dataset:    data/Oxford-IIIT_Pets
        >>> pet_test_cfg = dict(data_root='data/Oxford-IIIT_Pets',
        ... test_mode=True, ann_file='annotations/test.txt')
        >>> pet_test = Oxford_IIIT_Pets(**pet_test_cfg)
        >>> pet_test
        Dataset Oxford_IIIT_Pets
            Number of samples:  3669
            Number of categories:       37
            Root of dataset:    data/Oxford-IIIT_Pets
    """  # noqa: E501

    METAINFO = {'classes': OXFORD_IIIT_PETS_CATEGORIES}

    def __init__(self,
                 data_root: str,
                 test_mode: bool = False,
                 ann_file: str = 'annotations/trainval.txt',
                 data_prefix: str = 'images',
                 **kwargs):
        self.backend = get_file_backend(data_root, enable_singleton=True)

        super(Oxford_IIIT_Pets, self).__init__(
            ann_file=ann_file,
            data_root=data_root,
            data_prefix=data_prefix,
            test_mode=test_mode,
            **kwargs)

    def load_data_list(self):
        """Load images and ground truth labels."""

        pairs = list_from_file(self.ann_file)
        data_list = []
        for pair in pairs:
            img_name, class_id, _, _ = pair.split()
            img_name = f'{img_name}.jpg'
            img_path = self.backend.join_path(self.img_prefix, img_name)
            gt_label = int(class_id) - 1
            info = dict(img_path=img_path, gt_label=gt_label)
            data_list.append(info)
        return data_list

    def extra_repr(self) -> List[str]:
        """The extra repr information of the dataset."""
        body = [
            f'Root of dataset: \t{self.data_root}',
        ]
        return body
