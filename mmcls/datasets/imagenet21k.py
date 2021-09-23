# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os

import numpy as np

from .base_dataset import BaseDataset
from .builder import DATASETS
from .imagenet import find_folders, has_file_allowed_extension


@DATASETS.register_module()
class ImageNet21k(BaseDataset):
    """ImageNet21k Dataset.

    Args:
    data_prefix (str): the prefix of data path
    pipeline (list): a list of dict, where each element represents
        a operation defined in `mmcls.datasets.pipelines`
    ann_file (str | None): the annotation file. When ann_file is str,
        the subclass is expected to read from the ann_file. When ann_file
        is None, the subclass is expected to read according to data_prefix
    test_mode (bool): in train mode or test mode
    multi_lable (bool): use multi lable or not.
    recursion_subdir(bool): whether to use sub-directory pictures, which
        are meet the conditions in the folder under category directory.
    """

    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif')
    CLASSES = None

    def __init__(self,
                 data_prefix,
                 pipeline,
                 classes=None,
                 ann_file=None,
                 multi_lable=False,
                 recursion_subdir=False,
                 test_mode=False):
        self.multi_lable = multi_lable
        self.recursion_subdir = recursion_subdir
        super(ImageNet21k, self).__init__(data_prefix, pipeline, classes,
                                          ann_file, test_mode)

    def prepare_data(self, idx):
        results = copy.deepcopy(self.data_infos[idx])
        results['img_prefix'] = self.data_prefix
        return self.pipeline(results)

    def load_annotations(self):
        """load dataset annotations."""
        if self.ann_file is None:
            data_infos = self._load_annotations_from_dir()
        elif isinstance(self.ann_file, str):
            data_infos = self._load_annotations_from_file()
        else:
            raise TypeError('ann_file must be a str or None')

        if len(data_infos) == 0:
            msg = 'Found no valid file in '
            msg += f'{self.ann_file}. ' if self.ann_file \
                else f'{self.data_prefix}. '
            msg += 'Supported extensions are: ' + \
                {', '.join(self.IMG_EXTENSIONS)}
            raise RuntimeError(msg)

        return data_infos

    def _find_allowed_files(self, root, folder_name, class_id):
        """find all the allowed files in a folder, including sub folder if
        recursion_subdir is true."""
        _dir = os.path.join(root, folder_name)
        infos_pre_class = []
        for item in os.scandir(_dir):
            if item.is_file() and has_file_allowed_extension(
                    item.name, self.IMG_EXTENSIONS):
                path = os.path.join(folder_name, item.name)
                info = {
                    'img_info': dict(filename=path),
                    'gt_label': np.array(class_id, dtype=np.int64)
                }
                infos_pre_class.append(info)
            elif self.recursion_subdir and item.is_dir(
            ) and not item.name.startswith('.'):
                sub_folder_name = os.path.join(folder_name, item.name)
                infos_pre_class.extend(
                    self._find_allowed_files(root, sub_folder_name, class_id))

        return infos_pre_class

    def _load_annotations_from_dir(self):
        """load annotations from self.data_prefix directory."""
        data_infos = []
        empty_classes = []
        folder_to_idx = find_folders(self.data_prefix)
        self.folder_to_idx = folder_to_idx
        root = os.path.expanduser(self.data_prefix)
        for folder_name, idx in folder_to_idx.items():
            infos_pre_class = self._find_allowed_files(root, folder_name, idx)
            if len(infos_pre_class) == 0:
                empty_classes.append(folder_name)
            data_infos.extend(infos_pre_class)

        if len(empty_classes) != 0:
            msg = 'Found no valid file for the classes ' + \
                f"{', '.join(sorted(empty_classes))}"
            msg += 'Supported extensions are: ' + \
                f"{', '.join(self.IMG_EXTENSIONS)}."
            raise FileNotFoundError(msg)

        return data_infos

    def _load_annotations_from_file(self):
        """load annotations from self.ann_file."""
        data_infos = []
        with open(self.ann_file) as f:
            for line in f.readlines():
                if line == '':
                    continue
                filename, gt_label = line.strip().rsplit(' ', 1)
                info = {
                    'img_info': dict(filename=filename),
                    'gt_label': np.array(gt_label, dtype=np.int64)
                }
                data_infos.append(info)

        return data_infos
