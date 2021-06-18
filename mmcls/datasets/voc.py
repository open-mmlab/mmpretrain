import os.path as osp
import xml.etree.ElementTree as ET

import mmcv
import torch

from .builder import DATASETS
from .multi_label import MultiLabelDataset


@DATASETS.register_module()
class VOC(MultiLabelDataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Dataset."""

    CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor')

    def __init__(self, **kwargs):
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
        samples = []
        all_gt_labels = []
        sample_cnt = 0
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

            gt_label = torch.zeros(len(self.CLASSES), dtype=torch.int8)
            # The order cannot be swapped for the case where multiple objects
            # of the same kind exist and some are difficult.
            gt_label[labels_difficult] = -1
            gt_label[labels] = 1
            all_gt_labels.append(gt_label)

            sample = dict(
                img_prefix=self.data_prefix,
                img_info=dict(filename=filename),
                gt_label_index=sample_cnt)
            samples.append(sample)
            sample_cnt += 1

        # Stack tensors to accelerate copying to shared memory,
        # also avoid being limited by max number of file descriptors.
        # If there is a way to know total sample number from the beginning,
        # we'd better pre-allocate memory of the whole tensor instead of
        # stacking a list of tensors.
        all_gt_labels = torch.stack(all_gt_labels)

        data_infos = dict(
            all_gt_labels=all_gt_labels,
            samples=samples
        )
        return data_infos
