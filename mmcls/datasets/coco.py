# Copyright (c) OpenMMLab. All rights reserved.
import time
from typing import List

from mmengine.fileio import get_file_backend, load

from mmcls.registry import DATASETS
from .categories import COCO_CATEGORITES
from .multi_label import MultiLabelDataset


@DATASETS.register_module()
class CocoDataset(MultiLabelDataset):
    """`COCO2017 <https://cocodataset.org/#download>`_ Dataset.

    After decompression, the dataset directory structure is as follows:

    COCO dataset directory: ::

        COCO (data_root)/
        ├── train2017 (data_prefix['img_path'])
        │   ├── xxx.jpg
        │   ├── xxy.jpg
        │   └── ...
        ├── val2017 (data_prefix['img_path'])
        │   ├── xxx.jpg
        │   ├── xxy.jpg
        │   └── ...
        ├──annotations (directory contains COCO annotation JSON file)
        └── ...

    Extra iscrowd label is in COCO annotations, we will use
    `gt_label_crowd` to record the crowd labels in each sample
    and corresponding evaluation should take care of this field
    to calculate metrics. Usually, crowd labels are reckoned as
    negative in defaults.

    Args:
        data_root (str): The root directory for ``data_prefix`` and
            ``ann_file``. Defaults to ''.
        data_prefix (str): the prefix of data path for COCO dataset.
        ann_file (str): coco annotation file path.
        test_mode (bool): ``test_mode=True`` means in test phase.
            It determines to use the training set or test set.
        metainfo (dict, optional): Meta information for dataset, such as
            categories information. Defaults to None.
        **kwargs: Other keyword arguments in :class:`BaseDataset`.

    Examples:
        >>> from mmcls.datasets import CocoDataset
        >>> coco_train_cfg = dict(data_root='./data/coco/',
        >>>                       ann_file='annotations/instances_train2017.json',
        >>>                       data_prefix='train2017/')
        >>> coco_train = CocoDataset(**coco_train_cfg)
        >>> coco_train
        Dataset CocoDataset
            Number of samples:  118287
            Number of categories:       80
            Prefix of images:   ./data/coco/train2017/
            Path of annotation file:    ./data/coco/annotations/instances_train2017.json
        >>> coco_test_cfg = dict(data_root='./data/coco/',
        >>>                      ann_file='annotations/instances_val2017.json',
        >>>                      data_prefix='val2017/',
        >>>                      test_mode=True)
        >>> coco_test = CocoDataset(**coco_test_cfg)
        >>> coco_test
        Dataset CocoDataset
            Number of samples:  5000
            Number of categories:       80
            Prefix of images:   ./data/coco/val2017/
            Path of annotation file:    ./data/coco/annotations/instances_val2017.json
    """  # noqa: E501

    METAINFO = {'classes': COCO_CATEGORITES}

    def _get_labels_from_coco(self, img_id):
        """Get gt_labels and labels_crowd from COCO object."""

        info = self.coco.loadImgs([img_id])[0]
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        labels = [self.cat2label[ann['category_id']] for ann in ann_info]
        labels_crowd = [
            self.cat2label[ann['category_id']] for ann in ann_info
            if ann['iscrowd']
        ]

        labels, labels_crowd = set(labels), set(labels_crowd)
        img_path = info['file_name']

        return list(labels), list(labels_crowd), img_path

    def load_data_list(self):
        """Load images and ground truth labels."""
        self.backend = get_file_backend(self.ann_file, enable_singleton=True)

        try:
            from pycocotools.coco import COCO as _COCO
        except ImportError:
            raise ModuleNotFoundError(
                'please run `pip install pycocotools` to install 3rd package.')

        class MMClsCOCO(_COCO):
            """the difference with pycocotools.COCO is loading ann_file."""

            def __init__(self, annotation_file=None):
                super().__init__()
                if annotation_file is not None:
                    print('loading annotations into memory...')
                    tic = time.time()
                    # use `mmengint.fileio.load`` here to handle different
                    # file backend, such as ceph, local....
                    dataset = load(annotation_file)
                    assert type(dataset) == dict, (
                        f'annotation file format {type(dataset)} not supported'
                    )
                    print('Done (t={:0.2f}s)'.format(time.time() - tic))
                    self.dataset = dataset
                    self.createIndex()

        self.coco = MMClsCOCO(self.ann_file)
        # The order of returned `cat_ids` will not
        # change with the order of the CLASSES
        self.cat_ids = self.coco.getCatIds(catNms=self.METAINFO['classes'])
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.getImgIds()

        data_list = []
        for img_id in self.img_ids:

            labels, labels_crowd, img_path = self._get_labels_from_coco(img_id)
            img_path = self.backend.join_path(self.data_prefix['img_path'],
                                              img_path)

            info = dict(
                img_path=img_path,
                gt_label=labels,
                gt_label_crowd=labels_crowd)
            data_list.append(info)

        return data_list

    def extra_repr(self) -> List[str]:
        """The extra repr information of the dataset."""
        body = [
            f'Prefix of images: \t{self.data_prefix["img_path"]}',
            f'Path of annotation file: \t{self.ann_file}',
        ]

        return body
