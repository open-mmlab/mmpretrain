# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import mmengine
from mmengine.dataset import BaseDataset
from mmengine.fileio import get_file_backend

from mmpretrain.registry import DATASETS


@DATASETS.register_module()
class MiniGPT4Dataset(BaseDataset):
    """Dataset for training MiniGPT4.

    MiniGPT4 dataset directory:

        minigpt4_dataset
            ├── image
            │   ├── id0.jpg
            │   │── id1.jpg
            │   │── id2.jpg
            │   └── ...
            └── conversation_data.json

    The structure of conversation_data.json:

        [
            // English data
            {
                "id": str(id0),
                "conversation": "###Ask: <Img><ImageHere></Img> [Ask content]
                                ###Answer: [Answer content]"
            },

            // Chinese data
            {
                "id": str(id1),
                "conversation": "###问：<Img><ImageHere></Img> [Ask content]
                                ###答：[Answer content]"
            },

            ...
        ]

    Args:
        data_root (str): The root directory for ``ann_file`` and ``image``.
        ann_file (str): Conversation file path.
        **kwargs: Other keyword arguments in :class:`BaseDataset`.
    """

    def load_data_list(self) -> List[dict]:
        file_backend = get_file_backend(self.data_root)
        conversation_path = file_backend.join_path(self.data_root,
                                                   self.ann_file)
        conversation = mmengine.load(conversation_path)
        img_ids = {}
        n = 0
        for conv in conversation:
            img_id = conv['id']
            if img_id not in img_ids.keys():
                img_ids[img_id] = n
                n += 1

        img_root = file_backend.join_path(self.data_root, 'image')
        data_list = []
        for conv in conversation:
            img_file = '{}.jpg'.format(conv['id'])
            chat_content = conv['conversation']
            lang = 'en' if chat_content.startswith('###Ask: ') else 'zh'
            data_info = {
                'image_id': img_ids[conv['id']],
                'img_path': file_backend.join_path(img_root, img_file),
                'chat_content': chat_content,
                'lang': lang,
            }

            data_list.append(data_info)

        return data_list
