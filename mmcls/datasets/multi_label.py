import copy
import warnings

import numpy as np
import torch

from mmcls.core import average_performance, mAP
from .base_dataset import BaseDataset


class MultiLabelDataset(BaseDataset):
    """Multi-label Dataset."""

    def get_gt_labels(self):
        """Get all ground-truth labels (categories).

        Returns:
            torch.Tensor: ground truth labels for all images.
        """

        return self.data_infos['all_gt_labels']

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

    def prepare_data(self, idx):
        results = copy.deepcopy(self.data_infos['samples'][idx])
        gt_label_index = results.pop('gt_label_index')
        results['gt_label'] = self.data_infos['all_gt_labels'][gt_label_index]
        return self.pipeline(results)

    def __len__(self):
        return len(self.data_infos['samples'])

    def evaluate(self,
                 results,
                 metric='mAP',
                 metric_options=None,
                 logger=None,
                 **deprecated_kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
                Default value is 'mAP'. Options are 'mAP', 'CP', 'CR', 'CF1',
                'OP', 'OR' and 'OF1'.
            metric_options (dict, optional): Options for calculating metrics.
                Allowed keys are 'k' and 'thr'. Defaults to None
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Defaults to None.
            deprecated_kwargs (dict): Used for containing deprecated arguments.

        Returns:
            dict: evaluation results
        """
        if metric_options is None:
            metric_options = {'thr': 0.5}

        if deprecated_kwargs != {}:
            warnings.warn('Option arguments for metrics has been changed to '
                          '`metric_options`.')
            metric_options = {**deprecated_kwargs}

        if isinstance(metric, str):
            metrics = [metric]
        else:
            metrics = metric
        allowed_metrics = ['mAP', 'CP', 'CR', 'CF1', 'OP', 'OR', 'OF1']
        eval_results = {}
        results = np.vstack(results)
        gt_labels = self.get_gt_labels().numpy()
        num_imgs = len(results)
        assert len(gt_labels) == num_imgs, 'dataset testing results should '\
            'be of the same length as gt_labels.'

        invalid_metrics = set(metrics) - set(allowed_metrics)
        if len(invalid_metrics) != 0:
            raise ValueError(f'metric {invalid_metrics} is not supported.')

        if 'mAP' in metrics:
            mAP_value = mAP(results, gt_labels)
            eval_results['mAP'] = mAP_value
        if len(set(metrics) - {'mAP'}) != 0:
            performance_keys = ['CP', 'CR', 'CF1', 'OP', 'OR', 'OF1']
            performance_values = average_performance(results, gt_labels,
                                                     **metric_options)
            for k, v in zip(performance_keys, performance_values):
                if k in metrics:
                    eval_results[k] = v

        return eval_results
