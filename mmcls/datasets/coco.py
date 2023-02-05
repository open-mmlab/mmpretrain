# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from pycocotools.coco import COCO as _COCO

from .builder import DATASETS
from .multi_label import MultiLabelDataset


@DATASETS.register_module()
class COCO(MultiLabelDataset):
    """`COCO2017 <https://cocodataset.org/#download>`_ Dataset.

    Args:
        data_prefix (str): the prefix of data path
        pipeline (list): a list of dict, where each element represents
            a operation defined in `mmcls.datasets.pipelines`
        ann_file (str | None): the annotation file. When ann_file is str,
            the subclass is expected to read from the ann_file. When ann_file
            is None, the subclass is expected to read according to data_prefix
        iscrowd_as_postive (Optional[bool]): Whether to map the crowd
            labels as positive. If it set to True, map crowd examples to
            positive ones(1), If it set to False, map crowd examples to
            negative ones(0). Defaults to None, the crowd labels will be
            set to '-1'.
        extra_data_fields (list): a list of data fields.
    """

    CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

    def __init__(self,
                 iscrowd_as_postive=None,
                 extra_data_fields=['width', 'height', 'id'],
                 **kwargs):
        self.iscrowd_as_postive = iscrowd_as_postive
        self.extra_data_fields = extra_data_fields
        super(COCO, self).__init__(**kwargs)
        if '2017' in self.data_prefix:
            self.year = 2017
        else:
            raise ValueError('Cannot infer dataset year from img_prefix.')

    def load_annotations(self):
        """Load annotation from COCO style annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        self.coco = _COCO(self.ann_file)
        # The order of returned `cat_ids` will not
        # change with the order of the CLASSES
        self.cat_ids = self.coco.getCatIds(catNms=self.CLASSES)

        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.getImgIds()
        data_infos = []
        total_ann_ids = []
        for i in self.img_ids:

            info = self.coco.loadImgs([i])[0]
            ann_ids = self.coco.getAnnIds(imgIds=[i])
            ann_info = self.coco.loadAnns(ann_ids)
            labels = [self.cat2label[ann['category_id']] for ann in ann_info]
            labels_iscrowd = [
                self.cat2label[ann['category_id']] for ann in ann_info
                if ann['iscrowd']
            ]
            total_ann_ids.extend(ann_ids)

            gt_label = np.zeros(len(self.CLASSES))
            if self.iscrowd_as_postive is None:
                gt_label[labels_iscrowd] = -1
            elif self.iscrowd_as_postive:
                gt_label[labels_iscrowd] = 1
            else:
                gt_label[labels_iscrowd] = 0

            gt_label[labels] = 1

            img_info = dict(filename=info['file_name'])
            extra_info = {
                field: info[field]
                for field in self.extra_data_fields
            }
            img_info.update(extra_info)

            info = dict(
                img_prefix=self.data_prefix,
                img_info=img_info,
                gt_label=gt_label.astype(np.int8))
            data_infos.append(info)

        assert len(set(total_ann_ids)) == len(
            total_ann_ids
        ), f"Annotation ids in '{self.ann_file}' are not unique!"
        return data_infos
