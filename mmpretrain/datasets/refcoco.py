# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import mmengine
import numpy as np
from mmengine.dataset import BaseDataset
from mmengine.fileio import get_file_backend

from mmpretrain.registry import DATASETS


@DATASETS.register_module()
class RefCOCO(BaseDataset):
    """RefCOCO dataset.

    Args:
        ann_file (str): Annotation file path.
        data_root (str): The root directory for ``data_prefix`` and
            ``ann_file``. Defaults to ''.
        data_prefix (str | dict): Prefix for training data. Defaults to ''.
        pipeline (Sequence): Processing pipeline. Defaults to an empty tuple.
        **kwargs: Other keyword arguments in :class:`BaseDataset`.
    """

    def __init__(self, *args, ann_bbox_fmt='xyxy', **kwargs):
        if ann_bbox_fmt.lower() not in ['xyxy', 'xywh']:
            raise ValueError('Unknown bbox format, please '
                             'choose from ["xyxy", "xywh"]')
        self.ann_bbox_fmt = ann_bbox_fmt.lower()

        super().__init__(*args, **kwargs)

    def load_data_list(self) -> List[dict]:
        """Load data list."""
        img_prefix = self.data_prefix['img_path']
        annotations = mmengine.load(self.ann_file)
        file_backend = get_file_backend(img_prefix)

        data_list = []
        for ann in annotations:
            bbox = np.array(ann['bbox'], dtype=np.float32)
            if self.ann_bbox_fmt == 'xywh':
                # XYWH -> XYXY
                bbox[2:4] = bbox[0:2] + bbox[2:4]

            data_info = {
                'img_path': file_backend.join_path(img_prefix, ann['image']),
                'image_id': ann['image_id'],
                'text': ann['text'],
                'gt_bboxes': bbox[None, :],
            }

            data_list.append(data_info)

        return data_list
