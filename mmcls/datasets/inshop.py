# Copyright (c) OpenMMLab. All rights reserved.
from mmengine import get_file_backend, list_from_file

from mmcls.registry import DATASETS
from .base_dataset import BaseDataset


@DATASETS.register_module()
class InShop(BaseDataset):
    """InShop Dataset for Image Retrieval.

    Please download the images from the homepage
    'https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html'
    (In-shop Clothes Retrieval Benchmark -> Img -> img.zip,
    Eval/list_eval_partition.txt), and organize them as follows way: ::

        In-shop Clothes Retrieval Benchmark (data_root)/
           ├── Eval /
           │    └── list_eval_partition.txt (ann_file)
           ├── Img
           │    └── img/ (img_prefix)
           ├── README.txt
           └── .....

    Args:
        data_root (str): The root directory for dataset.
        split (str): Choose from 'train', 'query' and 'gallery'.
            Defaults to 'train'.
        data_prefix (str | dict): Prefix for training data. 
            Defaults to 'Img/img'.
        ann_file (str): Annotation file path, path relative to
            ``data_root``. Defaults to 'Eval/list_eval_partition.txt'.
        **kwargs: Other keyword arguments in :class:`BaseDataset`.

    Examples:
        >>> from mmcls.datasets import InShop
        >>> inshop_train_cfg = dict(data_root='data/inshop', \
        >>> ... split='train')
        >>> inshop_train = InShop(**inshop_train_cfg)
        >>> inshop_train
        Dataset InShop
            Number of samples:  25882
            The `CLASSES` meta info is not set.
            Root of dataset:    data/inshop
        >>> from mmcls.datasets import InShop
        >>> inshop_query_cfg = dict(data_root='data/inshop', \
        >>> ... split='query')
        >>> inshop_query = InShop(**inshop_query_cfg)
        >>> inshop_query
        Dataset InShop
            Number of samples:  14218
            The `CLASSES` meta info is not set.
            Root of dataset:    data/inshop
        >>> from mmcls.datasets import InShop
        >>> inshop_gallery_cfg = dict(data_root='data/inshop',\
        >>> ... split='gallery')
        >>> inshop_gallery = InShop(**inshop_gallery_cfg)
        >>> inshop_gallery
        Dataset InShop
            Number of samples:  12612
            The `CLASSES` meta info is not set.
            Root of dataset:    data/inshop
    """

    def __init__(self,
                 data_root: str,
                 split: str = 'train',
                 data_prefix: str = 'Img/img',
                 ann_file: str = 'Eval/list_eval_partition.txt',
                 **kwargs):

        assert split in ('train', 'query', 'gallery'), "'split' of `InShop`" \
            f" must be one of ['train', 'query', 'gallery'], bu get '{split}'"
        self.backend = get_file_backend(data_root, enable_singleton=True)
        super().__init__(
            data_root=data_root,
            data_prefix=data_prefix,
            ann_file=ann_file,
            **kwargs)

        self.split = split

    def _process_annotations(self):
        ann_path = self.backend.join_path(self.data_root, self.ann_file)
        anno_train = {'metainfo': {}, 'data_list': []}
        anno_query = {'metainfo': {}, 'data_list': []}
        anno_gallery = {'metainfo': {}, 'data_list': []}
        lines = list_from_file(ann_path)

        # item_id to label, each item corresponds to one class label
        class_num = 0
        gt_label_train = {}

        # item_id to label, each label corresponds to several items
        gallery_num = 0
        gt_label_gallery = {}

        for line in lines[2:]:
            # The first line (lines[0]) is the image number
            # and the second line (lines[1]) is the field name,
            # so process the sample from the third line (lines[2]).
            # Each line is formatted as follows,
            # ``image_name, item_id, evaluation_status``,
            # such as ``02_1_front.jpg id_001 train``.
            img_name, item_id, status = line.split()
            if status == 'train':
                if item_id not in gt_label_train:
                    gt_label_train[item_id] = class_num
                    class_num += 1
                # item_id to class_id (for the training set)
                anno_train['data_list'].append({
                    'img_path':
                    f'{self.data_root}/Img/{img_name}',
                    'gt_label':
                    gt_label_train[item_id],
                })
            elif status == 'gallery':
                if item_id not in gt_label_gallery:
                    gt_label_gallery[item_id] = []
                # Since there are multiple images for each item,
                # record the corresponding item for each image.
                gt_label_gallery[item_id].append(gallery_num)
                anno_gallery['data_list'].append({
                    'img_path': f'{self.data_root}/Img/{img_name}',
                    'sample_idx': gallery_num
                })
                gallery_num += 1
            else:
                continue

        # Generate the label for the query set
        query_num = 0
        for line in lines[2:]:
            img_name, item_id, status = line.split()
            if status == 'query':
                anno_query['data_list'].append({
                    'img_path':
                    f'{self.data_root}/Img/{img_name}',
                    'gt_label':
                    gt_label_gallery[item_id],
                })
                query_num += 1
            else:
                continue

        if self.split == 'train':
            anno_train['metainfo']['class_number'] = class_num
            anno_train['metainfo']['sample_number'] = \
                len(anno_train['data_list'])
            return anno_train
        elif self.split == 'query':
            anno_query['metainfo']['sample_number'] = query_num
            return anno_query
        else:
            anno_gallery['metainfo']['sample_number'] = gallery_num
            return anno_gallery

    def load_data_list(self):
        """For the train set, return image and ground truth label.

        For the query set, return image and ids of images in gallery. For the
        gallery set, return image and its id.
        """
        data_info = self._process_annotations()
        data_list = data_info['data_list']
        for data in data_list:
            data['img_path'] = self.backend.join_path(self.data_root,
                                                      data['img_path'])
        return data_list

    def extra_repr(self):
        """The extra repr information of the dataset."""
        body = [f'Root of dataset: \t{self.data_root}']
        return body
