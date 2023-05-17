# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Callable, List, Sequence

import mmengine
from mmengine.dataset import BaseDataset
from mmengine.fileio import get_file_backend

from mmpretrain.registry import DATASETS


@DATASETS.register_module()
class ScienceQA(BaseDataset):
    """ScienceQA dataset.

    This dataset is used to load the multimodal data of ScienceQA dataset.

    Args:
        data_root (str): The root directory for ``data_prefix`` and
            ``ann_file``.
        data_split (str): The split of dataset. Options: ``train``, ``val``,
            ``test``, ``trainval``, ``minival``, and ``minitest``.
        data_split_file (str): The split file of dataset, which contains the
            ids of data samples in the split.
        ann_file (str): Annotation file path.
        data_prefix (dict): Prefix for data field. Defaults to
            ``dict(img_path='')``.
        pipeline (Sequence): Processing pipeline. Defaults to an empty tuple.
        **kwargs: Other keyword arguments in :class:`BaseDataset`.
    """

    def __init__(self,
                 data_root: str,
                 data_split: str,
                 data_split_file: str,
                 ann_file: str,
                 data_prefix: dict = dict(img_path=''),
                 pipeline: Sequence[Callable] = (),
                 **kwargs):

        assert data_split in [
            'train', 'val', 'test', 'trainval', 'minival', 'minitest'
        ], f'Invalid split {data_split}'
        self.data_split = data_split
        self.data_split_file = os.path.join(data_root, data_split_file)

        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            data_prefix=data_prefix,
            pipeline=pipeline,
            **kwargs)

    def load_data_list(self) -> List[dict]:
        """Load data list."""
        img_prefix = self.data_prefix['img_path']
        annotations = mmengine.load(self.ann_file)
        current_data_split = mmengine.load(
            self.data_split_file)[self.data_split]  # noqa

        file_backend = get_file_backend(img_prefix)

        data_list = []
        for data_id in current_data_split:
            ann = annotations[data_id]
            if ann['image'] is not None:
                data_info = {
                    'image_id':
                    data_id,
                    'question':
                    ann['question'],
                    'choices':
                    ann['choices'],
                    'answer':
                    ann['answer'],
                    'hint':
                    ann['hint'],
                    'image_name':
                    ann['image'],
                    'task':
                    ann['task'],
                    'grade':
                    ann['grade'],
                    'subject':
                    ann['subject'],
                    'topic':
                    ann['topic'],
                    'category':
                    ann['category'],
                    'skill':
                    ann['skill'],
                    'lecture':
                    ann['lecture'],
                    'solution':
                    ann['solution'],
                    'split':
                    ann['split'],
                    'img_path':
                    file_backend.join_path(img_prefix, data_id,
                                           ann['image']),  # noqa
                }
                data_list.append(data_info)

        return data_list
