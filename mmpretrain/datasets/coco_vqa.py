# Copyright (c) OpenMMLab. All rights reserved.
from collections import Counter
from typing import List

import mmengine
from mmengine.dataset import BaseDataset
from mmengine.fileio import get_file_backend

from mmpretrain.registry import DATASETS


@DATASETS.register_module()
class COCOVQA(BaseDataset):
    """ COCO VQA dataset """

    def load_data_list(self) -> List[dict]:
        """ Load data list. """

        img_prefix = self.data_prefix['img_path']
        annotations = mmengine.load(self.ann_file)
        file_backend = get_file_backend(img_prefix)

        data_list = []
        for ann in annotations:
            # A typical VQA data info element
            # {
            #   "question_id": 262148001,
            #   "question": "What are the people doing?",
            #   "answer": ["spectating", "watching", "watching"],
            #   "image": "val2014/COCO_val2014_000000262148.jpg",
            #   "dataset": "vqa"
            # }

            img_path = file_backend.join_path(img_prefix, ann.pop('image'))
            data_info = {'img_path': img_path}

            if 'answer' in ann:
                # add answer_weight & answer_count, delete duplicate answer
                answer = ann.pop('answer')
                if isinstance(answer, list):
                    count = Counter(answer)
                    data_info['gt_answer'] = list(count.keys())
                    data_info['gt_answer_weight'] = ann.get(
                        'answer_weight',
                        [i / len(answer) for i in count.values()],
                    )
                elif isinstance(answer, str):
                    data_info['gt_answer'] = [answer]
                    data_info['gt_answer_weight'] = [1.0]

            # Update other keys to the data info.
            data_info.update(ann)
            data_list.append(data_info)

        return data_list
