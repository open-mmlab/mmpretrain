import copy
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

    def get_gt_labels(self):
        """Get all ground-truth labels (categories).

        Returns:
            torch.Tensor: ground truth labels for all images.
        """

        return self.data_infos['all_gt_labels'].numpy()

    def get_cat_ids(self, idx):
        """Get category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            torch.Tensor: Image categories of specified index.
        """
        gt_label_index = self.data_infos['samples'][idx]['gt_label_index']
        gt_label = self.data_infos['all_gt_labels'][gt_label_index]
        cat_ids = torch.where(gt_label == 1)[0]
        return cat_ids

    def __len__(self):
        return len(self.data_infos['samples'])

    def prepare_data(self, idx):
        results = copy.deepcopy(self.data_infos['samples'][idx])
        gt_label_index = results.pop('gt_label_index')
        results['gt_label'] = self.data_infos['all_gt_labels'][gt_label_index]
        return self.pipeline(results)

    def load_annotations(self):
        """Load annotations.

        Returns:
            list[dict]: Annotation info from XML file.
        """
        samples = []
        img_ids = mmcv.list_from_file(self.ann_file)
        all_gt_labels = torch.zeros(len(img_ids), len(self.CLASSES),
                                    dtype=torch.int8)
        for index, img_id in enumerate(img_ids):
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

            # The order cannot be swapped for the case where multiple objects
            # of the same kind exist and some are difficult.
            all_gt_labels[index, labels_difficult] = -1
            all_gt_labels[index, labels] = 1

            sample = dict(
                img_prefix=self.data_prefix,
                img_info=dict(filename=filename),
                gt_label_index=index)
            samples.append(sample)

        data_infos = dict(
            all_gt_labels=all_gt_labels,
            samples=samples
        )
        return data_infos
