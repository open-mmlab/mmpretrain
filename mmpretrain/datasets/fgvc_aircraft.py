# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

from mmengine import get_file_backend, list_from_file

from mmpretrain.registry import DATASETS
from .base_dataset import BaseDataset
from .categories import FGVC_AIRCRAFT_CATEGORIES


@DATASETS.register_module()
class FGVC_Aircraft(BaseDataset):
    """The FGVC_Aircraft Dataset.

    Support the `FGVC_Aircraft Dataset <https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/>`_ Dataset.
    After downloading and decompression, the dataset directory structure is as follows.

    FGVC Aircraft dataset directory: ::

        fgvc-aircraft-2013b/data (data_root)/
        ├── images (data_prefix)
        │   ├── 1.jpg
        │   ├── 2.jpg
        │   └── ...
        ├── images_variant_train.txt
        ├── images_variant_test.txt
        ├── images_variant_trainval.txt
        ├── images_variant_val.txt
        ├── variants.txt
        └── ....

    Args:
        data_root (str): The root directory for FGVC_Aircraft dataset.
        ann_file (str, optional): Annotation file path, path relative to
            ``data_root``. Defaults to 'images_variant_trainval.txt'.
        test_mode (bool): ``test_mode=True`` means in test phase. It determines
             to use the training set or test set. Defaults to False.
        data_prefix (str): Prefix for images, path relative to
            ``data_root``. Defaults to 'images'.

    Examples:
        >>> from mmpretrain.datasets import FGVC_Aircraft
        >>> aircraft_train_cfg = dict(data_root='data/fgvc-aircraft-2013b/data')
        >>> aircraft_train = FGVC_Aircraft(**aircraft_train_cfg)
        >>> aircraft_train
        Dataset FGVC_Aircraft
            Number of samples:  6667
            Number of categories:       100
            Root of dataset:    data/fgvc-aircraft-2013b/data
        >>> aircraft_test_cfg = dict(data_root='data/fgvc-aircraft-2013b/data', ann_file='images_variant_test.txt')
        >>> aircraft_test = FGVC_Aircraft(**aircraft_test_cfg)
        >>> aircraft_test
        Dataset FGVC_Aircraft
            Number of samples:  3333
            Number of categories:       100
            Root of dataset:    data/fgvc-aircraft-2013b/data
    """  # noqa: E501

    METAINFO = {'classes': FGVC_AIRCRAFT_CATEGORIES}

    def __init__(self,
                 data_root: str,
                 ann_file: str = 'images_variant_trainval.txt',
                 test_mode: bool = False,
                 data_prefix: str = 'images',
                 **kwargs):
        self.backend = get_file_backend(data_root, enable_singleton=True)

        super(FGVC_Aircraft, self).__init__(
            ann_file=ann_file,
            data_root=data_root,
            test_mode=test_mode,
            data_prefix=data_prefix,
            **kwargs)

    def load_data_list(self):
        """Load images and ground truth labels."""

        pairs = list_from_file(self.ann_file)
        data_list = []
        for pair in pairs:
            pair = pair.split()
            img_name = pair[0]
            class_name = ' '.join(pair[1:])
            img_name = f'{img_name}.jpg'
            img_path = self.backend.join_path(self.img_prefix, img_name)
            gt_label = self.METAINFO['classes'].index(class_name)
            info = dict(img_path=img_path, gt_label=gt_label)
            data_list.append(info)

        return data_list

    def extra_repr(self) -> List[str]:
        """The extra repr information of the dataset."""
        body = [
            f'Root of dataset: \t{self.data_root}',
        ]
        return body
