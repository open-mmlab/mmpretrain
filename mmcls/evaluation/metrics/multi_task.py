# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Sequence

from mmengine.evaluator import BaseMetric

from mmcls.registry import METRICS
from mmcls.structures import MultiTaskDataSample


@METRICS.register_module()
class MultiTasksMetric(BaseMetric):
    """Metrics for MultiTask
    Args:
        task_metrics(dict): a dictionary in the keys are the names of the tasks
            and the values is a list of the metric corresponds to this task
    Examples:
        >>> import torch
        >>> from mmcls.evaluation import MultiTasksMetric
        >>> # -------------------- The Basic Usage --------------------
        >>>task_metrics = {
            'task0': [dict(type='Accuracy', topk=(1, ))],
            'task1': [dict(type='Accuracy', topk=(1, 3)),]
        }
        >>>pred = [
            {
            'gt_task'{
                'task0': torch.tensor([0.7, 0.0, 0.3]),
                'task1': torch.tensor([0.5, 0.2, 0.3])
                },
            'pred_task' : {'task0:0 , task2:2'}
            },
            {'gt_task':
                'task0': torch.tensor([0.0, 0.0, 1.0]),
                'task1': torch.tensor([0.0, 0.0, 1.0])
                }
            'pred_task' : {'task0:2 , task2:2'}
            },
        ]
        >>>metric = MultiTasksMetric(self.task_metrics)
        >>>metric.process(None, self.pred)
        >>>metric.evaluate(2)
        {'task0_accuracy/top1': 100.0,
        'task1_accuracy/top1': 50.0,
        'task1_accuracy/top3': 100.0}

    """

    def __init__(self,
                 task_metrics: Dict,
                 collect_device: str = 'cpu') -> None:
        self.task_metrics = task_metrics
        super().__init__(collect_device=collect_device)

        self._metrics = {}
        for task_name in self.task_metrics.keys():
            self._metrics[task_name] = []
            for metric in self.task_metrics[task_name]:
                self._metrics[task_name].append(METRICS.build(metric))

    def pre_process_nested(self, data_samples: List[MultiTaskDataSample],
                           task_name):
        """Retrieve data_samples corresponds to the task_name for a data_sample
        type MultiTaskDataSample Args :

        data_samples (List[MultiTaskDataSample]):The annotation data of every
        samples. task_name (str)
        """
        task_data_sample = []
        for data_sample in data_samples:
            task_data_sample.append(
                data_sample.to_target_data_sample('MultiTaskDataSample',
                                                  task_name).to_dict())
        return task_data_sample

    def pre_process_cls(self, data_samples: List[MultiTaskDataSample],
                        task_name):
        """Retrieve data_samples corresponds to the task_name for a data_sample
        type ClsDataSample Args :

        data_samples (List[MultiTaskDataSample]):The annotation data of every
        samples. task_name (str)
        """
        task_data_sample_dicts = []
        for data_sample in data_samples:
            task_data_sample_dicts.append(
                data_sample.to_target_data_sample('ClsDataSample',
                                                  task_name).to_dict())
        return task_data_sample_dicts

    def process(self, data_batch, data_samples: Sequence[dict]):
        """Process one batch of data samples.

        The processed results should be stored in ``self.results``, which will
        be used to computed the metrics when all batches have been processed.
        Args:
            data_batch: A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        data_sample_instances = []
        for data_sample in data_samples:
            if 'gt_task' in data_sample:
                data_sample_instances.append(MultiTaskDataSample().set_gt_task(
                    data_sample['gt_task']).set_pred_task(
                        data_sample['pred_task']))
        for task_name in self.task_metrics.keys():
            filtered_data_samples = []
            for data_sample in data_sample_instances:
                sample_mask = data_sample.get_task_mask(task_name)
                if sample_mask:
                    filtered_data_samples.append(data_sample)
            if type(self.task_metrics[task_name]) != dict:
                for metric in self._metrics[task_name]:
                    # Current implementation is only comptaible
                    # With 2 types of metrics :
                    # * Cls Metrics
                    # * Nested Cls Metrics
                    # In order to make it work with other
                    # non-cls heads/metrics, one will have to
                    # override the current implementation
                    if metric.__class__.__name__ != 'MultiTasksMetric':
                        task_data_sample_dicts = self.pre_process_cls(
                            filtered_data_samples, task_name)
                        metric.process(data_batch, task_data_sample_dicts)
                    else:
                        task_data_sample_dicts = self.pre_process_nested(
                            filtered_data_samples, task_name)
                        metric.process(data_batch, task_data_sample_dicts)

    def compute_metrics(self, results: list) -> dict:
        raise Exception('compute metrics should not be used here directly')

    def evaluate(self, size):
        """Evaluate the model performance of the whole dataset after processing
        all batches.

        Args:
            size (int): Length of the entire validation dataset. When batch
                size > 1, the dataloader may pad some data samples to make
                sure all ranks have the same length of dataset slice. The
                ``collect_results`` function will drop the padded data based on
                this size.
        Returns:
            dict: Evaluation metrics dict on the val dataset. The keys are
            "{task_name}_{metric_name}" , and the values
            are corresponding results.
        """
        metrics = {}
        for task_name in self._metrics:
            for metric in self._metrics[task_name]:
                name = metric.__class__.__name__
                if name == 'MultiTasksMetric' or metric.results:
                    results = metric.evaluate(size)
                else:
                    results = {metric.__class__.__name__: 0}
                for key in results:
                    name = f'{task_name}_{key}'
                    if name in results:
                        """Inspired from https://github.com/open-
                        mmlab/mmengine/ bl ob/ed20a9cba52ceb371f7c825131636b9e2
                        747172e/mmengine/evalua tor/evaluator.py#L84-L87."""
                        raise ValueError(
                            'There are multiple metric results with the same'
                            f'metric name {name}. Please make sure all metrics'
                            'have different prefixes.')
                    metrics[name] = results[key]
        return metrics
