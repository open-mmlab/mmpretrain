# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp
from collections import OrderedDict
from os import PathLike
from typing import List, Sequence, Union

from mmengine import get_file_backend

from mmpretrain.registry import DATASETS, TRANSFORMS
from .base_dataset import BaseDataset


def expanduser(data_prefix):
    if isinstance(data_prefix, (str, PathLike)):
        return osp.expanduser(data_prefix)
    else:
        return data_prefix


@DATASETS.register_module()
class COCORetrieval(BaseDataset):
    """COCO Retrieval dataset.

    COCO (Common Objects in Context): The COCO dataset contains more than
    330K images,each of which has approximately 5 descriptive annotations.
    This dataset was releasedin collaboration between Microsoft and Carnegie
    Mellon University

    COCO_2014 dataset directory: ::

        COCO_2014
        ├── val2014
        ├── train2014
        ├── annotations
                 ├── instances_train2014.json
                 ├── instances_val2014.json
                 ├── person_keypoints_train2014.json
                 ├── person_keypoints_val2014.json
                 ├── captions_train2014.json
                 ├── captions_val2014.json

    Args:
        ann_file (str): Annotation file path.
        test_mode (bool): Whether dataset is used for evaluation. This will
            decide the annotation format in data list annotations.
            Defaults to False.
        data_root (str): The root directory for ``data_prefix`` and
            ``ann_file``. Defaults to ''.
        data_prefix (str | dict): Prefix for training data. Defaults to ''.
        pipeline (Sequence): Processing pipeline. Defaults to an empty tuple.
        **kwargs: Other keyword arguments in :class:`BaseDataset`.

    Examples:
        >>> from mmpretrain.datasets import COCORetrieval
        >>> train_dataset=COCORetrieval(data_root='coco2014/')
        >>> train_dataset
        Dataset COCORetrieval
            Number of samples: 	414113
            Annotation file:  /coco2014/annotations/captions_train2014.json
            Prefix of images:  /coco2014/
        >>> from mmpretrain.datasets import COCORetrieval
        >>> val_dataset = COCORetrieval(data_root='coco2014/')
        >>> val_dataset
         Dataset COCORetrieval
             Number of samples: 	202654
             Annotation file: 	/coco2014/annotations/captions_val2014.json
             Prefix of images: 	/coco2014/
    """

    def __init__(self,
                 ann_file: str,
                 test_mode: bool = False,
                 data_prefix: Union[str, dict] = '',
                 data_root: str = '',
                 pipeline: Sequence = (),
                 **kwargs):

        if isinstance(data_prefix, str):
            data_prefix = dict(img_path=expanduser(data_prefix))

        ann_file = expanduser(ann_file)
        transforms = []
        for transform in pipeline:
            if isinstance(transform, dict):
                transforms.append(TRANSFORMS.build(transform))
            else:
                transforms.append(transform)

        super().__init__(
            data_root=data_root,
            data_prefix=data_prefix,
            test_mode=test_mode,
            pipeline=transforms,
            ann_file=ann_file,
            **kwargs,
        )

    def load_data_list(self) -> List[dict]:
        """Load data list."""
        # get file backend
        img_prefix = self.data_prefix['img_path']
        file_backend = get_file_backend(img_prefix)

        anno_info = json.load(open(self.ann_file, 'r'))
        # mapping img_id to img filename
        img_dict = OrderedDict()
        for idx, img in enumerate(anno_info['images']):
            if img['id'] not in img_dict:
                img_rel_path = img['coco_url'].rsplit('/', 2)[-2:]
                img_path = file_backend.join_path(img_prefix, *img_rel_path)

                # create new idx for image
                img_dict[img['id']] = dict(
                    ori_id=img['id'],
                    image_id=idx,  # will be used for evaluation
                    img_path=img_path,
                    text=[],
                    gt_text_id=[],
                    gt_image_id=[],
                )

        train_list = []
        for idx, anno in enumerate(anno_info['annotations']):
            anno['text'] = anno.pop('caption')
            anno['ori_id'] = anno.pop('id')
            anno['text_id'] = idx  # will be used for evaluation
            # 1. prepare train data list item
            train_data = anno.copy()
            train_image = img_dict[train_data['image_id']]
            train_data['img_path'] = train_image['img_path']
            train_data['image_ori_id'] = train_image['ori_id']
            train_data['image_id'] = train_image['image_id']
            train_data['is_matched'] = True
            train_list.append(train_data)
            # 2. prepare eval data list item based on img dict
            img_dict[anno['image_id']]['gt_text_id'].append(anno['text_id'])
            img_dict[anno['image_id']]['text'].append(anno['text'])
            img_dict[anno['image_id']]['gt_image_id'].append(
                train_image['image_id'])

        self.img_size = len(img_dict)
        self.text_size = len(anno_info['annotations'])

        # return needed format data list
        if self.test_mode:
            return list(img_dict.values())
        return train_list
