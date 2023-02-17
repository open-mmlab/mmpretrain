# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Optional, Sequence, Union

from mmengine.logging import MMLogger

from mmcls.registry import DATASETS
from .custom import CustomDataset, get_file_backend, get_samples


@DATASETS.register_module()
class CustomRetrievalDataset(CustomDataset):
    """Custom dataset for retrieval.

    The dataset supports following annotation format.

    1. The samples are arranged in the specific way: ::

           data_prefix/{split}
           ├── class_x
           │   ├── xxx.png
           │   ├── xxy.png
           │   └── ...
           │       └── xxz.png
           └── class_y
               ├── 123.png
               ├── nsdf3.png
               ├── ...
               └── asd932_.png

    Args:
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Defaults to None.
        data_root (str): The root directory for ``data_prefix`` and
            ``ann_file``. Defaults to ''.
        data_prefix (str | dict): Prefix for the data. Defaults to ''.
        extensions (Sequence[str]): A sequence of allowed extensions. Defaults
            to ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif').
        split (str): Choose from 'train', 'query' and 'gallery'.
            Defaults to 'train'.
        **kwargs: Other keyword arguments in :class:`BaseDataset`.
    """

    def __init__(self,
                 metainfo: Optional[dict] = None,
                 data_root: str = '',
                 data_prefix: Union[str, dict] = '',
                 extensions: Sequence[str] = ('.jpg', '.jpeg', '.png', '.ppm',
                                              '.bmp', '.pgm', '.tif'),
                 split: str = 'train',
                 **kwargs):

        assert split in ('train', 'query', 'gallery'), "'split' of `InShop`" \
            f" must be one of ['train', 'query', 'gallery'], bu get '{split}'"
        self.split = split
        self.split_for_load = self.split  # for img_prefix
        super().__init__(
            ann_file='',
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=data_prefix,
            extensions=extensions,
            **kwargs)

    @property
    def img_prefix(self):
        """The prefix of images."""
        return osp.join(self.data_prefix['img_path'], self.split_for_load)

    def load_data_list(self):
        if self.split in ['train', 'gallery']:
            data_list = super().load_data_list()
            return data_list
        else:
            # for query
            # load gallery
            self.split_for_load = 'gallery'
            gallery_data_list = super().load_data_list()
            gt_label_gallery = dict()
            gallery_num = 0
            for data in gallery_data_list:
                data['sample_idx'] = gallery_num
                item_id = data.pop('gt_label')
                if item_id not in gt_label_gallery:
                    gt_label_gallery[item_id] = []
                gt_label_gallery[item_id].append(gallery_num)
                gallery_num += 1

            # load query
            self.split_for_load = self.split
            samples, empty_classes = get_samples(
                self.img_prefix,
                self.folder_to_idx,
                is_valid_file=self.is_valid_file,
            )

            if len(samples) == 0:
                raise RuntimeError(
                    f'Found 0 files in subfolders of: {self.data_prefix}. '
                    f'Supported extensions are: {",".join(self.extensions)}')

            if empty_classes:
                logger = MMLogger.get_current_instance()
                logger.warning(
                    'Found no valid file in the folder '
                    f'{", ".join(empty_classes)}. '
                    f"Supported extensions are: {', '.join(self.extensions)}")

            # Pre-build file backend to prevent verbose file backend inference.
            backend = get_file_backend(self.img_prefix, enable_singleton=True)
            data_list = []
            for filename, gt_label in samples:
                img_path = backend.join_path(self.img_prefix, filename)
                info = {
                    'img_path': img_path,
                    'gt_label': gt_label_gallery[int(gt_label)]
                }
                data_list.append(info)
            return data_list
