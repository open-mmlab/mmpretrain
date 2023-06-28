# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import mmengine
from mmengine.dataset import BaseDataset

from mmpretrain.registry import DATASETS


@DATASETS.register_module()
class VisDial(BaseDataset):
    """VisDial dataset.

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

    def load_data_list(self) -> List[dict]:
        """Load data list."""
        annotations = mmengine.load(self.ann_file)['data']

        dialogs = annotations['dialogs']
        answers = annotations['answers']
        questions = annotations['questions']

        data_list = []

        for dialog in dialogs:
            image_id = dialog['image_id']
            caption = dialog['caption']

            historys = ['Caption:' + caption + '.']

            for i in range(1, len(dialog['dialog'])):
                historys.append('')

                previous_idx = i - 1
                # for j in range(i):
                question_id = dialog['dialog'][previous_idx]['question']
                answer_id = dialog['dialog'][previous_idx]['answer']

                history = ' Question:{question}? Answer:{answer}.' \
                    .format(question=questions[question_id],
                            answer=answers[answer_id])

                historys[i] = historys[previous_idx] + history

            # get question and answer options for each dialog round
            for dialog_id, dialog_round in enumerate(dialog['dialog']):
                question_id = dialog_round['question']
                answer_id = dialog_round['answer']
                answer_options = [
                    answers[answer_id]
                    for answer_id in dialog_round['answer_options']
                ]

                data_info = dict(image_id=image_id)
                data_info['dialog_history'] = historys[dialog_id]

                data_info['question'] = questions[question_id]
                data_info['answer'] = answers[answer_id]
                data_info['answer_options'] = answer_options

                data_list.append(data_info)

        return data_list
