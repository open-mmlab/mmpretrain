# Copyright (c) OpenMMLab. All rights reserved.
import copy

from mmengine import FileClient

from mmcls.registry import DATASETS
from .base_dataset import BaseDataset


@DATASETS.register_module()
class GLDv2Retrieval(BaseDataset):
    """Google Landmark Dataset v2 for Image Retrieval.

    Please download the images and meta data according to
    'https://github.com/cvdfoundation/google-landmark'
    and organize them as follows:

        google-landmark_v2 (data_root)/
           ├── train
           │   ├── 0
           │   ├── 1
           │   └── ...
           ├── index
           │   ├── 0
           │   ├── 1
           │   └── ...
           ├── test
           │   ├── 0
           │   ├── 1
           │   └── ...
           └── metadata (anno_prefix)
               ├── index.csv
               ├── index_image_to_landmark.csv
               ├── index_label_to_category.csv
               ├── recognition_solution_v2.1.csv
               ├── retrieval_solution_v2.1.csv
               ├── test.csv
               ├── train.csv
               ├── train_attribution.csv
               ├── train_clean.csv
               └── train_label_to_category.csv

    Args:
        data_root (str): The root directory for dataset
        mode (str): The value is in 'train', 'index' and 'test'.
            Defaults to 'train'.
        train_set (str): When mode is 'train',
            'train_set'='clean' indicates the clean version, and
            'train_set'='full' indicates the full version.
            Defaults to 'clean'.
        test_set (str): When mode is 'test',\
            'test_set'='full' indicates the full test set,
            'test_set'='public' indicates the public set, and
            'test_set'='private' indicates the private set.
            Defaults to 'full'.
        anno_prefix (str): The sub-directory for meta-data.
            Defaults to 'metadata'.
        **kwargs: Other keyword arguments in :class:`BaseDataset`.

    Examples:
        >>> from mmcls.datasets import GLDv2Retrieval as GLDv2
        >>> gld_train_cfg = dict(data_root='data/google-landmark', \
        >>> ... mode='train')
        >>> gld_train = GLDv2(**gld_train_cfg)
        >>> gld_train
        Dataset GLDv2
            Number of samples:  1580470
            The `CLASSES` meta info is not set.
            Root of dataset:    data/google-landmark-v2
        >>> from mmcls.datasets import GLDv2Retrieval as GLDv2
        >>> gld_train_cfg = dict(data_root='data/google-landmark', \
        >>> ... mode='train', train_set='full')
        >>> gld_train = GLDv2(**gld_train_cfg)
        >>> gld_train
        Dataset GLDv2
            Number of samples:  4132914
            The `CLASSES` meta info is not set.
            Root of dataset:    data/google-landmark-v2
        >>> from mmcls.datasets import GLDv2Retrieval as GLDv2
        >>> gld_index_cfg = dict(data_root='data/google-landmark',\
        >>> ... mode='index')
        >>> gld_index = GLDv2(**gld_index_cfg)
        >>> gld_index
        Dataset GLDv2
            Number of samples:  761757
            The `CLASSES` meta info is not set.
            Root of dataset:    data/google-landmark-v2
        >>> from mmcls.datasets import GLDv2Retrieval as GLDv2
        >>> gld_test_cfg = dict(data_root='data/google-landmark', \
        >>> ... mode='test')
        >>> gld_test = GLDv2(**gld_test_cfg)
        >>> gld_test
        Dataset GLDv2
            Number of samples:  117577
            The `CLASSES` meta info is not set.
            Root of dataset:    data/google-landmark-v2
        >>> from mmcls.datasets import GLDv2Retrieval as GLDv2
        >>> gld_test_cfg = dict(data_root='data/google-landmark', \
        >>> ... mode='test', test_set='public')
        >>> gld_test = GLDv2(**gld_test_cfg)
        >>> gld_test
        Dataset GLDv2
            Number of samples:  379
            The `CLASSES` meta info is not set.
            Root of dataset:    data/google-landmark-v2
        >>> from mmcls.datasets import GLDv2Retrieval as GLDv2
        >>> gld_test_cfg = dict(data_root='data/google-landmark', \
        >>> ... mode='test', test_set='private')
        >>> gld_test = GLDv2(**gld_test_cfg)
        >>> gld_test
        Dataset GLDv2
            Number of samples:  750
            The `CLASSES` meta info is not set.
            Root of dataset:    data/google-landmark-v2
    """

    def __init__(self,
                 data_root: str = 'data/Google_Lanmark_v2',
                 mode: str = 'train',
                 train_set: str = 'clean',
                 test_set: str = 'full',
                 anno_prefix: str = 'data/Google_Lanmark_v2/metadata',
                 **kwargs):
        test_mode = False if mode == 'train' else True

        assert mode in ['train', 'index', 'test'], \
            '``{}`` is an illegal mode'.format(mode)

        # Because the dataset is very large, images and
        # annotations stored on different storage is supported.
        self.file_client = FileClient.infer_client(uri=data_root)
        self.anno_client = FileClient.infer_client(uri=anno_prefix)

        self.anno_prefix = anno_prefix
        self.mode = mode
        self.train_set = train_set
        self.test_set = test_set
        self.data_info = self._process_annotations()

        super().__init__(
            data_root=data_root, test_mode=test_mode, ann_file='', **kwargs)
        self.data_root = data_root

    def _process_train(self):
        if self.train_set == 'full':
            return self._process_train_full()
        elif self.train_set == 'clean':
            return self._process_train_clean()
        else:
            raise ValueError(f'Invalid train_set={self.train_set}')

    def _process_train_clean(self):
        path = self.anno_client.join_path(self.anno_prefix, 'train_clean.csv')
        anno = {'metainfo': {}, 'data_list': []}
        lines = self.anno_client.get_text(path).strip()
        lines = lines.split('\n')
        gt_label = 0
        sample_number = 0
        for line in lines[1:]:
            # The first line (lines[0]) is the field name,
            # so process the sample from the second line (lines[1]).
            # Each line is a landmark in the format:
            # ``landmark_id, images(list)``,
            # such as ``1, 17660ef415d37059 92b6290d571448f6``.
            line = line.split(',')
            images = line[1].split()
            for image in images:
                image_path = 'train/{}/{}/{}/{}.jpg'. \
                    format(image[0], image[1], image[2], image)
                anno['data_list'].append({
                    'img_path': image_path,
                    'gt_label': gt_label,
                })
                sample_number += 1
            gt_label += 1
        anno['metainfo']['class_number'] = gt_label
        anno['metainfo']['sample_number'] = sample_number
        return anno

    def _process_train_full(self):
        path = self.anno_client.join_path(self.anno_prefix, 'train.csv')
        anno = {'metainfo': {}, 'data_list': []}
        lines = self.anno_client.get_text(path).strip()
        lines = lines.split('\n')
        gt_label = set()
        sample_number = 0
        for line in lines[1:]:
            # The first line (lines[0]) is the field name,
            # so process the sample from the second line (lines[1]).
            # Each line is a landmark in the format:
            # ``landmark_id, url, landmark_id``.
            line = line.split(',')
            image = line[0]
            landmark_id = int(line[-1])
            image_path = 'train/{}/{}/{}/{}.jpg'. \
                format(image[0], image[1], image[2], image)
            anno['data_list'].append({
                'img_path': image_path,
                'gt_label': landmark_id,
            })
            sample_number += 1
            gt_label.add(landmark_id)
        anno['metainfo']['class_number'] = len(gt_label)
        anno['metainfo']['sample_number'] = sample_number
        return anno

    def _process_index(self):
        path = self.anno_client.join_path(self.anno_prefix, 'index.csv')
        anno = {'metainfo': {}, 'data_list': []}
        lines = self.anno_client.get_text(path).strip()
        lines = lines.split('\n')
        sample_number = 0
        for line in lines[1:]:
            # The first line (lines[0]) is the field name,
            # so process the sample from the second line (lines[1]).
            # Each line is a image name.
            image = line
            image_path = 'index/{}/{}/{}/{}.jpg'. \
                format(image[0], image[1], image[2], image)
            anno['data_list'].append({
                'img_path': image_path,
                'sample_idx': sample_number
            })
            sample_number += 1
        anno['metainfo']['sample_number'] = sample_number
        return anno

    def _process_test_full(self):
        path = self.anno_client.join_path(self.anno_prefix, 'test.csv')
        anno = {'metainfo': {}, 'data_list': []}
        lines = self.anno_client.get_text(path).strip()
        lines = lines.split('\n')
        sample_number = 0
        for line in lines[1:]:
            # The first line (lines[0]) is the field name,
            # so process the sample from the second line (lines[1]).
            # Each line is a image name.
            image = line
            image_path = 'test/{}/{}/{}/{}.jpg'. \
                format(image[0], image[1], image[2], image)
            anno['data_list'].append({
                'img_path': image_path,
            })
            sample_number += 1
        anno['metainfo']['sample_number'] = sample_number
        return anno

    def _process_test_sub(self):
        path = self.anno_client.join_path(self.anno_prefix,
                                          'retrieval_solution_v2.1.csv')

        index = {}
        anno_public = {'metainfo': {}, 'data_list': []}
        anno_private = {'metainfo': {}, 'data_list': []}

        # Process the index set to get the path
        # and serial number of each image in order.
        # eg., index = {index_image_pat: index_image_index}
        index_data = self._process_index()
        for i in range(index_data['metainfo']['sample_number']):
            index[index_data['data_list'][i]['img_path']] = i

        lines = self.anno_client.get_text(path).strip()
        lines = lines.split('\n')
        for line in lines[1:]:
            # The first line (lines[0]) is the field name,
            # so we process the sample from the second line (lines[1]).
            # Each line is a landmark in the format: ``id, images, Usage``,
            line = line.split(',')
            image_id, image_list, usage = line
            if usage == 'Ignored':
                # if images=None and Usage=Ignored,
                # just ignore this sample
                continue
            else:
                if image_list[0] == '"':
                    # for some samples, some images are retrieved,
                    # and the images should look like "0001 0002 003"
                    # that is, the first and last characters
                    # are double quotations.
                    image_list = image_list[1:-1].split()
                else:
                    # for some samples, only one image is retrieved
                    image_list = image_list.split()
                query_path = 'test/{}/{}/{}/{}.jpg'. \
                    format(image_id[0], image_id[1], image_id[2],
                           image_id)

                index_list = []
                for image in image_list:
                    image_path = 'index/{}/{}/{}/{}.jpg'. \
                        format(image[0], image[1], image[2], image)
                    index_list.append(index[image_path])

                info = {'img_path': query_path, 'gt_label': index_list}

                if usage == 'Public':
                    anno_public['data_list'].append(info)
                else:
                    anno_private['data_list'].append(info)

        if self.test_set == 'public':
            return anno_public
        elif self.test_set == 'private':
            return anno_private
        else:
            raise ValueError(f'Invalid test_set={self.test_set}')

    def _process_annotations(self):
        if self.mode == 'train':
            return self._process_train()
        elif self.mode == 'index':
            return self._process_index()
        else:
            if self.test_set == 'full':
                return self._process_test_full()
            else:
                return self._process_test_sub()

    def load_data_list(self):
        """Load image paths and gt_labels.

        ``train`` mode returns image path and gt_labels. ``index`` mode returns
        image path. ``test_public/private`` mode returns image path,  index
        image path and index_ids.
        """
        data_list = self.data_info['data_list']
        data_list_new = []
        for data in data_list:
            img_path = data['img_path']
            full_path = self.file_client.join_path(self.data_root, img_path)
            if self.mode == 'index':
                # For index set, return image_path and sample_idx
                data_list_new.append({
                    'img_path': full_path,
                    'sample_idx': data['sample_idx']
                })
            elif self.mode == 'test' and self.test_set == 'full':
                # For test set (full), only the image path is returned.
                data_list_new.append({
                    'img_path': full_path,
                })
            else:
                # For train Set and test set (public/private),
                # return image path and ground-truth.
                gt_label = data['gt_label']
                data_list_new.append({
                    'img_path': full_path,
                    'gt_label': gt_label
                })
        return copy.deepcopy(data_list_new)

    def extra_repr(self):
        """The extra repr information of the dataset."""
        body = [f'Root of dataset: \t{self.data_root}']
        return body
