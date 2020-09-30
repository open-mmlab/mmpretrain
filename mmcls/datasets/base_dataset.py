import copy
from abc import ABCMeta, abstractmethod

import numpy as np
from torch.utils.data import Dataset

from mmcls.models.losses import accuracy
from .pipelines import Compose


class BaseDataset(Dataset, metaclass=ABCMeta):
    """Base dataset.

    Args:
        data_prefix (str): the prefix of data path
        pipeline (list): a list of dict, where each element represents
            a operation defined in `mmcls.datasets.pipelines`
        ann_file (str | None): the annotation file. When ann_file is str,
            the subclass is expected to read from the ann_file. When ann_file
            is None, the subclass is expected to read according to data_prefix
        test_mode (bool): in train mode or test mode
    """

    CLASSES = None

    def __init__(self, data_prefix, pipeline, ann_file=None, test_mode=False, classes=None):
        super(BaseDataset, self).__init__()

        self.ann_file = ann_file
        self.data_prefix = data_prefix
        self.test_mode = test_mode
        self.pipeline = Compose(pipeline)
        self.original_idx_to_subset_idx = None
        self.CLASSES = self.get_classes(classes)
        self.data_infos = self.load_annotations()

        if self.cusoriginal_idx_to_subset_idx is not None:
            self.data_infos = self.get_subset_by_classes()
        
    @abstractmethod
    def load_annotations(self):
        pass

    def get_classes(self, classes):
        if classes is None:
            return self.CLASSES

        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')
        
        if self.CLASSES is not None:
            # Uses subset of CLASSES
            self.original_idx_to_subset_idx = {
                self.CLASSES.index(x): n for n, x in enumerate(classes)
            }

        return class_names

    def get_subset_by_classes(self):
        new_data_infos = []
        for data in self.data_infos:
            if data['gt_label'] not in self.original_idx_to_subset_idx:
                continue
            else:
                data['gt_label'] = self.original_idx_to_subset_idx[data['gt_label']]
                new_data_infos.append(data)
        return new_data_infos

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.CLASSES)}

    def get_gt_labels(self):
        gt_labels = np.array([data['gt_label'] for data in self.data_infos])
        return gt_labels

    def prepare_data(self, idx):
        results = copy.deepcopy(self.data_infos[idx])
        return self.pipeline(results)

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        return self.prepare_data(idx)

    def evaluate(self,
                 results,
                 metric='accuracy',
                 metric_options={'topk': (1, 5)},
                 logger=None):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
                Default value is `accuracy`.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
        Returns:
            dict: evaluation results
        """
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['accuracy']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')

        eval_results = {}
        if metric == 'accuracy':
            topk = metric_options.get('topk')
            results = np.vstack(results)
            gt_labels = self.get_gt_labels()
            num_imgs = len(results)
            assert len(gt_labels) == num_imgs
            acc = accuracy(results, gt_labels, topk)
            eval_results = {f'top-{k}': a.item() for k, a in zip(topk, acc)}
        return eval_results
