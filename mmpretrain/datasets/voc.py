# Copyright (c) OpenMMLab. All rights reserved.
import xml.etree.ElementTree as ET
from typing import List, Optional, Union

from mmengine import get_file_backend, list_from_file

from mmpretrain.registry import DATASETS
from .base_dataset import expanduser
from .categories import VOC2007_CATEGORIES
from .multi_label import MultiLabelDataset


@DATASETS.register_module()
class VOC(MultiLabelDataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Dataset.

    After decompression, the dataset directory structure is as follows:

    VOC dataset directory: ::

        VOC2007 (data_root)/
        ├── JPEGImages (data_prefix['img_path'])
        │   ├── xxx.jpg
        │   ├── xxy.jpg
        │   └── ...
        ├── Annotations (data_prefix['ann_path'])
        │   ├── xxx.xml
        │   ├── xxy.xml
        │   └── ...
        └── ImageSets (directory contains various imageset file)

    Extra difficult label is in VOC annotations, we will use
    `gt_label_difficult` to record the difficult labels in each sample
    and corresponding evaluation should take care of this field
    to calculate metrics. Usually, difficult labels are reckoned as
    negative in defaults.

    Args:
        data_root (str): The root directory for VOC dataset.
        image_set_path (str): The path of image set, The file which
            lists image ids of the sub dataset, and this path is relative
            to ``data_root``.
        data_prefix (dict): Prefix for data and annotation, keyword
            'img_path' and 'ann_path' can be set. Defaults to be
            ``dict(img_path='JPEGImages', ann_path='Annotations')``.
        test_mode (bool): ``test_mode=True`` means in test phase.
            It determines to use the training set or test set.
        metainfo (dict, optional): Meta information for dataset, such as
            categories information. Defaults to None.
        **kwargs: Other keyword arguments in :class:`BaseDataset`.
    """  # noqa: E501

    METAINFO = {'classes': VOC2007_CATEGORIES}

    def __init__(self,
                 data_root: str,
                 image_set_path: str,
                 data_prefix: Union[str, dict] = dict(
                     img_path='JPEGImages', ann_path='Annotations'),
                 test_mode: bool = False,
                 metainfo: Optional[dict] = None,
                 **kwargs):
        if isinstance(data_prefix, str):
            data_prefix = dict(img_path=expanduser(data_prefix))
        assert isinstance(data_prefix, dict) and 'img_path' in data_prefix, \
            '`data_prefix` must be a dict with key img_path'

        if test_mode is False:
            assert 'ann_path' in data_prefix and data_prefix[
                'ann_path'] is not None, \
                '"ann_path" must be set in `data_prefix` if `test_mode` is' \
                ' False.'

        self.data_root = data_root
        self.backend = get_file_backend(data_root, enable_singleton=True)
        self.image_set_path = self.backend.join_path(data_root, image_set_path)

        super().__init__(
            ann_file='',
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=data_prefix,
            test_mode=test_mode,
            **kwargs)

    @property
    def ann_prefix(self):
        """The prefix of images."""
        if 'ann_path' in self.data_prefix:
            return self.data_prefix['ann_path']
        else:
            return None

    def _get_labels_from_xml(self, img_id):
        """Get gt_labels and labels_difficult from xml file."""
        xml_path = self.backend.join_path(self.ann_prefix, f'{img_id}.xml')
        content = self.backend.get(xml_path)
        root = ET.fromstring(content)

        labels, labels_difficult = set(), set()
        for obj in root.findall('object'):
            label_name = obj.find('name').text
            # in case customized dataset has wrong labels
            # or CLASSES has been override.
            if label_name not in self.CLASSES:
                continue
            label = self.class_to_idx[label_name]
            difficult = int(obj.find('difficult').text)
            if difficult:
                labels_difficult.add(label)
            else:
                labels.add(label)

        return list(labels), list(labels_difficult)

    def load_data_list(self):
        """Load images and ground truth labels."""
        data_list = []
        img_ids = list_from_file(self.image_set_path)

        for img_id in img_ids:
            img_path = self.backend.join_path(self.img_prefix, f'{img_id}.jpg')

            labels, labels_difficult = None, None
            if self.ann_prefix is not None:
                labels, labels_difficult = self._get_labels_from_xml(img_id)

            info = dict(
                img_path=img_path,
                gt_label=labels,
                gt_label_difficult=labels_difficult)
            data_list.append(info)

        return data_list

    def extra_repr(self) -> List[str]:
        """The extra repr information of the dataset."""
        body = [
            f'Prefix of dataset: \t{self.data_root}',
            f'Path of image set: \t{self.image_set_path}',
            f'Prefix of images: \t{self.img_prefix}',
            f'Prefix of annotations: \t{self.ann_prefix}'
        ]

        return body
