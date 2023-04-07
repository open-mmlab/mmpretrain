# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

from mmengine.fileio import join_path, load

from mmpretrain.registry import DATASETS
from .base_dataset import BaseDataset


@DATASETS.register_module()
class COCOCaption(BaseDataset):
    """COCO Caption dataset.

    Args:
        ann_file (str): Annotation file path.
        data_root (str): The root directory for ``data_prefix`` and
            ``ann_file``. Defaults to ''.
        data_prefix (str | dict): Prefix for training data. Defaults to ''.
        pipeline (Sequence): Processing pipeline. Defaults to an empty tuple.
        **kwargs: Other keyword arguments in :class:`BaseDataset`.
    """

    def load_data_list(self) -> List[dict]:
        """Load data list."""
        raw_anno_info = load(self.ann_file)

        data_list = []
        for idx, anno in enumerate(raw_anno_info):

            anno['img_path'] = join_path(self.img_prefix, anno['image'])
            img_id = anno['image'].split('/')[-1].strip('.jpg').split('_')[-1]
            anno['image_id'] = img_id
            anno['instance_id'] = idx
            anno['text'] = anno.pop('caption')

            data_list.append(anno)

        return data_list
