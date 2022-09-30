# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import xml.etree.ElementTree as ET

import mmcv
import numpy as np

from .builder import DATASETS
from .multi_label import MultiLabelDataset


@DATASETS.register_module()
class VOC(MultiLabelDataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Dataset.

    Args:
        data_prefix (str): the prefix of data path
        pipeline (list): a list of dict, where each element represents
            a operation defined in `mmcls.datasets.pipelines`
        ann_file (str | None): the annotation file. When ann_file is str,
            the subclass is expected to read from the ann_file. When ann_file
            is None, the subclass is expected to read according to data_prefix
        difficult_as_postive (Optional[bool]): Whether to map the difficult
            labels as positive. If it set to True, map difficult examples to
            positive ones(1), If it set to False, map difficult examples to
            negative ones(0). Defaults to None, the difficult labels will be
            set to '-1'.
    """

    CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor')

    def __init__(self, difficult_as_postive=None, **kwargs):
        self.difficult_as_postive = difficult_as_postive
        super(VOC, self).__init__(**kwargs)
        if 'VOC2007' in self.data_prefix:
            self.year = 2007
        else:
            raise ValueError('Cannot infer dataset year from img_prefix.')

    def load_annotations(self):
        """Load annotations.

        Returns:
            list[dict]: Annotation info from XML file.
        """
        data_infos = []
        img_ids = mmcv.list_from_file(self.ann_file)
        for img_id in img_ids:
            filename = f'JPEGImages/{img_id}.jpg'
            xml_path = osp.join(self.data_prefix, 'Annotations',
                                f'{img_id}.xml')
            tree = ET.parse(xml_path)
            root = tree.getroot()
            labels = []
            labels_difficult = []
            for obj in root.findall('object'):
                label_name = obj.find('name').text
                # in case customized dataset has wrong labels
                # or CLASSES has been override.
                if label_name not in self.CLASSES:
                    continue
                label = self.class_to_idx[label_name]
                difficult = int(obj.find('difficult').text)
                if difficult:
                    labels_difficult.append(label)
                else:
                    labels.append(label)

            gt_label = np.zeros(len(self.CLASSES))
            # set difficult example first, then set postivate examples.
            # The order cannot be swapped for the case where multiple objects
            # of the same kind exist and some are difficult.
            if self.difficult_as_postive is None:
                # map difficult examples to -1,
                # it may be used in evaluation to ignore difficult targets.
                gt_label[labels_difficult] = -1
            elif self.difficult_as_postive:
                # map difficult examples to positive ones(1).
                gt_label[labels_difficult] = 1
            else:
                # map difficult examples to negative ones(0).
                gt_label[labels_difficult] = 0
            gt_label[labels] = 1

            info = dict(
                img_prefix=self.data_prefix,
                img_info=dict(filename=filename),
                gt_label=gt_label.astype(np.int8))
            data_infos.append(info)

        return data_infos
