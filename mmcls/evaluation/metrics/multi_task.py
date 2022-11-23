# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Sequence

from mmengine.evaluator import BaseMetric
from mmcls.registry import METRICS
from mmcls.structures import MultiTaskDataSample


@METRICS.register_module()
class MultiTasksMetric(BaseMetric):
    """a."""

    def __init__(self,
                 task_metrics: Dict,
                 collect_device: str = 'cpu') -> None:
        self.task_metrics = task_metrics
        super().__init__(collect_device=collect_device)

        self._metrics = {}
        for task_name in self.task_metrics.keys():
            if type(self.task_metrics[task_name]) == list:
                self._metrics[task_name] = []
                for metric in self.task_metrics[task_name]:
                    self._metrics[task_name].append(METRICS.build(metric))
            elif type(self.task_metrics[task_name]) == dict:
                for task_name2 in self.task_metrics[task_name].keys():
                    self._metrics[task_name + '_' + task_name2] = []
                    for metric in self.task_metrics[task_name][task_name2]:
                        self._metrics[task_name + '_' + task_name2].append(
                            METRICS.build(metric))

    def pre_process_nested(self, data_samples, task_name):
        """
        """
        task_data_sample = []
        for data_sample in data_samples:
            task_data_sample.append(
                data_sample.to_target_data_sample('MultiTaskDataSample',
                                                  task_name))
        return task_data_sample

    def pre_process_cls(self, data_samples, task_name):
        """
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
            if type(self.task_metrics[task_name]) != dict:
                task_data_sample_dicts = self.pre_process_cls(
                    data_sample_instances, task_name)
                # à reflichir
                for metric in self._metrics[task_name]:
                    metric.process(data_batch, task_data_sample_dicts)
            else:
                task_data_sample = self.pre_process_nested(
                    data_sample_instances, task_name)
                for task_name2 in self.task_metrics[task_name]:
                    task_data_sample_dicts = self.pre_process_cls(
                        task_data_sample, task_name2)
                    for metric in self._metrics[task_name + '_' + task_name2]:
                        metric.process(data_batch, task_data_sample_dicts)

    def compute_metrics(self, results: list) -> dict:
        raise Exception("compute metrics should not be used here directly")

    def evaluate(self, size):
        metrics = {}
        for task_name in self._metrics:
            for metric in self._metrics[task_name]:
                if metric.results:
                    results = metric.evaluate(size)
                else:
                    results = {metric.__class__.__name__: 0}
                for key in results:
                    name = f'{task_name}_{key}'
                    if name in results:
                        """Inspired from https://github.com/open-mmlab/mmengine/ bl
                        ob/ed20a9cba52ceb371f7c825131636b9e2747172e/mmengine/evalua
                        tor/evaluator.py#L84-L87."""
                        raise ValueError(
                            'There are multiple metric results with the same'
                            f'metric name {name}. Please make sure all metrics'
                            'have different prefixes.')
                    metrics[name] = results[key]
        return metrics
