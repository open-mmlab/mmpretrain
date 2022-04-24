# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from collections import defaultdict
from os import PathLike
from typing import Dict, List, Optional, Sequence

import mmcv
import numpy as np
from mmcv.fileio import FileClient
from mmcv.utils.misc import is_list_of

from .base_dataset import BaseDataset
from .builder import DATASETS
from .multi_label import MultiLabelDataset
from .pipelines import Compose


def expanduser(path):
    if isinstance(path, (str, PathLike)):
        return osp.expanduser(path)
    else:
        return path


def isabs(uri):
    return osp.isabs(uri) or ('://' in uri)


@DATASETS.register_module()
class MultiTaskDataset:
    """Custom dataset for multi-task dataset.

    To use the dataset, please generate and provide an annotation file in the
    below format:

    .. code-block:: json

        {
          "metainfo": {
            "tasks":
              [
                {"name": "gender",
                 "type": "single-label",
                 "categories": ["male", "female"]},
                {"name": "wear",
                 "type": "multi-label",
                 "categories": ["shirt", "coat", "jeans", "pants"]}
              ]
          },
          "data_list": [
            {
              "img_path": "a.jpg",
              "gender_img_label": 0,
              "wear_img_label": [1, 0, 1, 0]
            },
            {
              "img_path": "b.jpg",
              "gender_img_label": 1,
              "wear_img_label": [0, 1, 0, 1]
            }
          ]
        }

    Assume we put our dataset in the ``data/mydataset`` folder in the
    repository and organize it as the below format: ::

        mmclassification/
        └── data
            └── mydataset
                ├── annotation
                │   ├── train.json
                │   ├── test.json
                │   └── val.json
                ├── train
                │   ├── a.jpg
                │   └── ...
                ├── test
                │   ├── b.jpg
                │   └── ...
                └── val
                    ├── c.jpg
                    └── ...

    We can use the below config to build datasets:

    .. code:: python

        >>> from mmcls.datasets import build_dataset
        >>> train_cfg = dict(
        ...     type="MultiTaskDataset",
        ...     ann_file="annotation/train.json",
        ...     data_root="data/mydataset",
        ...     # The `img_path` field in the train annotation file is relative
        ...     # to the `train` folder.
        ...     data_prefix='train',
        ... )
        >>> train_dataset = build_dataset(train_cfg)

    Or we can put all files in the same folder: ::

        mmclassification/
        └── data
            └── mydataset
                 ├── train.json
                 ├── test.json
                 ├── val.json
                 ├── a.jpg
                 ├── b.jpg
                 ├── c.jpg
                 └── ...

    And we can use the below config to build datasets:

    .. code:: python

        >>> from mmcls.datasets import build_dataset
        >>> train_cfg = dict(
        ...     type="MultiTaskDataset",
        ...     ann_file="train.json",
        ...     data_root="data/mydataset",
        ...     # the `data_prefix` is not required since all paths are
        ...     # relative to the `data_root`.
        ... )
        >>> train_dataset = build_dataset(train_cfg)


    Args:
        ann_file (str): The annotation file path. It can be either absolute
            path or relative path to the ``data_root``.
        metainfo (dict, optional): The extra meta information. It should be
            a dict with the same format as the ``"metainfo"`` field in the
            annotation file. Defaults to None.
        data_root (str, optional): The root path of the data directory. It's
            the prefix of the ``data_prefix`` and the ``ann_file``. And it can
            be a remote path like "s3://openmmlab/xxx/". Defaults to None.
        data_prefix (str, optional): The base folder relative to the
            ``data_root`` for the ``"img_path"`` field in the annotation file.
            Defaults to None.
        pipeline (Sequence[dict]): A list of dict, where each element
            represents a operation defined in :mod:`mmcls.datasets.pipelines`.
            Defaults to an empty tuple.
        test_mode (bool): in train mode or test mode. Defaults to False.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmcv.fileio.FileClient` for details.
            If None, automatically inference from the ``data_root``.
            Defaults to None.
    """
    METAINFO = dict()

    def __init__(self,
                 ann_file: str,
                 metainfo: Optional[dict] = None,
                 data_root: Optional[str] = None,
                 data_prefix: Optional[str] = None,
                 pipeline: Sequence = (),
                 test_mode: bool = False,
                 file_client_args: Optional[dict] = None):

        self.data_root = expanduser(data_root)

        # Inference the file client
        if self.data_root is not None:
            file_client = FileClient.infer_client(
                file_client_args, uri=self.data_root)
        else:
            file_client = FileClient(file_client_args)
        self.file_client: FileClient = file_client

        self.ann_file = self._join_root(expanduser(ann_file))
        self.data_prefix = self._join_root(data_prefix)

        self.test_mode = test_mode
        self.pipeline = Compose(pipeline)
        self.data_list = self.load_data_list(self.ann_file, metainfo)

    def _join_root(self, path):
        """Join ``self.data_root`` with the specified path.

        If the path is an absolute path, just return the path. And if the
        path is None, return ``self.data_root``.

        Examples:
            >>> self.data_root = 'a/b/c'
            >>> self._join_root('d/e/')
            'a/b/c/d/e'
            >>> self._join_root('https://openmmlab.com')
            'https://openmmlab.com'
            >>> self._join_root(None)
            'a/b/c'
        """
        if path is None:
            return self.data_root
        if isabs(path):
            return path

        joined_path = self.file_client.join_path(self.data_root, path)
        return joined_path

    @classmethod
    def _get_meta_info(cls, in_metainfo: dict = None) -> dict:
        """Collect meta information from the dictionary of meta.

        Args:
            in_metainfo (dict): Meta information dict.

        Returns:
            dict: Parsed meta information.
        """
        # `cls.METAINFO` will be overwritten by in_meta
        metainfo = copy.deepcopy(cls.METAINFO)
        if in_metainfo is None:
            return metainfo

        metainfo.update(in_metainfo)

        # Format check
        assert 'tasks' in metainfo, \
            'Please specify the `tasks` in the `metainfo` argument or ' \
            'the `metainfo` field in the annotation file.'
        tasks = metainfo['tasks']
        assert is_list_of(tasks, dict), \
            'Every task of `tasks` in the `metainfo` should be a dict.'
        for task in tasks:
            for field in ['name', 'categories', 'type']:
                assert field in task, \
                    f'Missing "{field}" in some tasks meta information.'

        return metainfo

    def load_data_list(self, ann_file, metainfo_override=None):
        """Load annotations from an annotation file.

        Args:
            ann_file (str): Absolute annotation file path if ``self.root=None``
                or relative path if ``self.root=/path/to/data/``.

        Returns:
            list[dict]: A list of annotation.
        """
        annotations = mmcv.load(ann_file)
        if not isinstance(annotations, dict):
            raise TypeError(f'The annotations loaded from annotation file '
                            f'should be a dict, but got {type(annotations)}!')
        if 'data_list' not in annotations:
            raise ValueError('The annotation file must have the `data_list` '
                             'field.')
        metainfo = annotations.get('metainfo', {})
        raw_data_list = annotations['data_list']

        # Set meta information.
        assert isinstance(metainfo, dict), 'The `metainfo` field in the '\
            f'annotation file should be a dict, but got {type(metainfo)}'
        if metainfo_override is not None:
            assert isinstance(metainfo_override, dict), 'The `metainfo` ' \
                f'argument should be a dict, but got {type(metainfo_override)}'
            metainfo.update(metainfo_override)
        self._metainfo = self._get_meta_info(metainfo)

        data_list = []
        for i, raw_data in enumerate(raw_data_list):
            try:
                data_list.append(self.parse_data_info(raw_data))
            except AssertionError as e:
                raise RuntimeError(
                    f'The format check fails during parse the item {i} of '
                    f'the annotation file with error: {e}')
        return data_list

    def parse_data_info(self, raw_data):
        """Parse raw annotation to target format.

        This method will return a dict which contains the data information of a
        sample.

        Args:
            raw_data (dict): Raw data information load from ``ann_file``

        Returns:
            dict: Parsed annotation.
        """
        assert isinstance(raw_data, dict), \
            f'The item should be a dict, but got {type(raw_data)}'
        assert 'img_path' in raw_data, \
            "The item doesn't have `img_path` field."
        data = dict(
            img_prefix=self.data_root,
            img_info=dict(filename=raw_data['img_path']),
        )

        for task in self._metainfo['tasks']:
            label_key = task['name'] + '_img_label'
            task_type = task['type']
            assert label_key in raw_data, \
                f"The item doesn't have `{label_key}` field."
            if task_type == 'single-label':
                data[label_key] = np.array(raw_data[label_key], dtype=np.int64)
                assert data[label_key].ndim == 0, \
                    'The label of single-label task should be a single number.'
            elif task_type == 'multi-label':
                data[label_key] = np.array(raw_data[label_key], dtype=np.int8)
                assert (data[label_key] <= 1).all(), \
                    'The label of multi-label task should be one-hot format.'

        return data

    @property
    def class_to_idx(self):
        """Map mapping class name to class index.

        Returns:
            Dict[str, dict]: The mapping from class name to class index of
            each tasks.
        """

        mapping_dict = {}
        for task in self.metainfo['tasks']:
            name = task['name']
            categories = task['categories']
            mapping_dict[name] = {
                category: i
                for i, category in enumerate(categories)
            }
        return mapping_dict

    @property
    def metainfo(self) -> dict:
        """Get meta information of dataset.

        Returns:
            dict: meta information collected from ``cls.METAINFO``,
            annotation file and metainfo argument during instantiation.
        """
        return copy.deepcopy(self._metainfo)

    @property
    def CLASSES(self) -> dict:
        """Get the classes information of dataset.

        Returns:
            Dict[str, list]: The categories list for each task.
        """
        return {
            task['name']: task['categories']
            for task in self._metainfo['tasks']
        }

    def get_gt_labels(self):
        """Get all ground-truth labels (categories).

        Returns:
            Dict[str, np.ndarray]: categories of all images for each task.
        """

        gt_labels_dict = defaultdict(list)
        for data in self.data_list:
            for task in self.metainfo['tasks']:
                name = task['name']
                gt_labels_dict[name].append(data[f'{name}_img_label'])
        for k, v in gt_labels_dict.items():
            gt_labels_dict[k] = np.array(v)
        return dict(gt_labels_dict)

    def get_cat_ids(self, idx: int) -> Dict[str, List[int]]:
        """Get the category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            Dict[str, List[int]]: Image category ids of specified index for
            each task.
        """
        data = self.data_list[idx]
        cat_ids_dict = {}
        for task in self.metainfo['tasks']:
            name = task['name']
            task_type = task['type']
            gt_label = data[f'{name}_img_label']
            if task_type == 'single-label':
                cat_ids_dict[name] = [int(gt_label)]
            elif task_type == 'multi-label':
                cat_ids_dict[name] = np.where(gt_label == 1)[0].tolist()

        return cat_ids_dict

    def prepare_data(self, idx):
        """Get data processed by ``self.pipeline``.

        Args:
            idx (int): The index of ``data_info``.

        Returns:
            Any: Depends on ``self.pipeline``.
        """
        results = copy.deepcopy(self.data_list[idx])
        return self.pipeline(results)

    def __len__(self):
        """Get the length of the whole dataset.

        Returns:
            int: The length of filtered dataset.
        """
        return len(self.data_list)

    def __getitem__(self, idx):
        """Get the idx-th image and data information of dataset after
        ``self.pipeline``.

        Args:
            idx (int): The index of of the data.

        Returns:
            dict: The idx-th image and data information after
            ``self.pipeline``.
        """
        return self.prepare_data(idx)

    def evaluate(self,
                 results,
                 metric=None,
                 metric_options=None,
                 indices=None,
                 logger=None):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (Dict[str, str | list[str]], optional): Metrics for each
                task to be evaluated. Defaults to None, which means to use the
                default metrics for every kinds of tasks.
            metric_options (Dict[str, dict], optional): Options for calculating
                metrics. Allowed keys for single-label tasks can be found in
                the :class:`BaseDataset` and for multi-label tasks can be found
                in the :class:`MultiLabelDataset`. Defaults to None.
            indices (list, optional): The indices of samples corresponding to
                the results. Defaults to None.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Defaults to None.

        Returns:
            dict: evaluation results
        """
        eval_results = {}
        gt_labels_dict = self.get_gt_labels()
        results_dict = defaultdict(list)
        for result in results:
            for task in self.metainfo['tasks']:
                name = task['name']
                results_dict[name].append(result[name])

        for task in self.metainfo['tasks']:
            name = task['name']
            task_type = task['type']
            gt_labels = gt_labels_dict[name]
            if indices is not None:
                gt_labels = gt_labels[indices]
            if task_type == 'single-label':
                eval_func = BaseDataset.evaluate_single_label
            elif task_type == 'multi-label':
                eval_func = MultiLabelDataset.evaluate_multi_label

            # To enable default values of `metric` and `metric_options`.
            eval_args = dict(
                results=np.vstack(results_dict[name]),
                gt_labels=gt_labels,
                logger=logger)
            if metric is not None and name in metric:
                eval_args['metric'] = metric[name]
            if metric_options is not None and name in metric_options:
                eval_args['metric_options'] = metric_options[name]

            eval_result = eval_func(**eval_args)
            for k, v in eval_result.items():
                eval_results[f'{name}_{k}'] = v

        return eval_results

    def __repr__(self):
        """Print the basic information of the dataset.

        Returns:
            str: Formatted string.
        """
        head = 'Dataset ' + self.__class__.__name__
        body = [f'Number of samples: \t{self.__len__()}']
        if self.data_root is not None:
            body.append(f'Root location: \t{self.data_root}')
        body.append(f'Annotation file: \t{self.ann_file}')
        if self.data_prefix is not None:
            body.append(f'Prefix of images: \t{self.data_prefix}')
        # -------------------- extra repr --------------------
        tasks = self.metainfo['tasks']
        body.append(f'For {len(tasks)} tasks')
        for task in tasks:
            body.append(f'    {task["name"]} ({len(task["categories"])} '
                        f'categories, {task["type"]})')
        # ----------------------------------------------------

        if len(self.pipeline.transforms) > 0:
            body.append('With transforms:')
            for t in self.pipeline.transforms:
                body.append(f'    {t}')

        lines = [head] + [' ' * 4 + line for line in body]
        return '\n'.join(lines)
