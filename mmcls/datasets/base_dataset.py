import copy
from abc import ABCMeta, abstractmethod

import numpy as np
from torch.utils.data import Dataset

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

    def __init__(self, data_prefix, pipeline, ann_file=None, test_mode=False):
        super(BaseDataset, self).__init__()

        self.ann_file = ann_file
        self.data_prefix = data_prefix
        self.test_mode = test_mode
        self.pipeline = Compose(pipeline)
        self.data_infos = self.load_annotations()

    @abstractmethod
    def load_annotations(self):
        pass

    def prepare_data(self, idx):
        results = copy.deepcopy(self.data_infos[idx])
        return self.pipeline(results)

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        return self.prepare_data(idx)

    def evaluate(self, results, metric='accuracy', logger=None):
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
            nums = []
            for result in results:
                nums.append(result['num_samples'].item())
                for topk, v in result['accuracy'].items():
                    if topk not in eval_results:
                        eval_results[topk] = []
                    eval_results[topk].append(v.item())
            assert sum(nums) == len(self.data_infos)
            for topk, accs in eval_results.items():
                eval_results[topk] = np.average(accs, weights=nums)
        return eval_results
