# Copyright (c) OpenMMLab. All rights reserved.
from collections import Counter
from typing import List

import mmengine
from mmengine.dataset import BaseDataset

from mmpretrain.registry import DATASETS
from mmengine.utils import is_abs
import os.path as osp


@DATASETS.register_module()
class ChartQA(BaseDataset):
    """ChartQA dataset.

    dataset:https://github.com/vis-nlp/ChartQA

    folder structure:
        data/chartqa
        ├── test
        │   ├── png
        │   ├── tables
        │   ├── test_human.json
        │   └── test_augmented.json
        ├── train
        │   ├── png
        │   ├── tables
        │   ├── train_human.json
        │   └── train_augmented.json
        └── val
            ├── png
            ├── tables
            ├── val_human.json
            └── val_augmented.json
    Args:
        data_root (str): The root directory for ``data_prefix``, ``ann_file``
            and ``question_file``.
        data_prefix (str): The directory of images.
        question_file (str): Question file path.
        ann_file (str, optional): Annotation file path for training and
            validation. Defaults to an empty string.
        **kwargs: Other keyword arguments in :class:`BaseDataset`.
    """

    def __init__(self,
                 data_root: str,
                 data_prefix: str,
                 ann_file: str = '',
                 **kwarg):
        super().__init__(
            data_root=data_root,
            data_prefix=dict(img_path=data_prefix),
            ann_file=ann_file,
            **kwarg,
        )    
        
    def _join_prefix(self):
        # Automatically join annotation file path with `self.root` if
        # `self.ann_file` is not an absolute path.
        if not any(is_abs(sub_ann_file) for sub_ann_file in self.ann_file) and self.ann_file:
            self.ann_file = [osp.join(self.data_root, sub_ann_file) for sub_ann_file in self.ann_file]
        # Automatically join data directory with `self.root` if path value in
        # `self.data_prefix` is not an absolute path.
        for data_key, prefix in self.data_prefix.items():
            if isinstance(prefix, str):
                if not is_abs(prefix):
                    self.data_prefix[data_key] = osp.join(
                        self.data_root, prefix)
                else:
                    self.data_prefix[data_key] = prefix
            else:
                raise TypeError('prefix should be a string, but got '
                                f'{type(prefix)}')
                
    def load_data_list(self) -> List[dict]:
        """Load data list."""
        data_list = []
        
        for sub_ann_file in self.ann_file:
            
            annotations = mmengine.load(sub_ann_file)

            

            for ann in annotations:

                # ann example
                # {
                #     'imgname': '41699051005347.png'
                #     'query': 'How many food item i...bar graph?',
                #       'label': '14'
                # }
                
                data_info = dict(question=ann['query'])
                data_info['image_id'] = ann['imgname']
                
                img_path = mmengine.join_path(self.data_prefix['img_path'],
                                            ann['imgname'])
                
                data_info['img_path'] = img_path
                data_info['gt_answer'] = ann['label']
                
                if 'human' in sub_ann_file:
                    data_info['sub_set'] = 'ChartQA-H'
                elif 'augmented' in sub_ann_file:
                    data_info['sub_set'] = 'ChartQA-M'
                else:
                    raise ValueError(f'Do not support to subset {sub_ann_file}.')

                data_list.append(data_info)

        return data_list


