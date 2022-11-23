# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Sequence
import torch

from mmengine.evaluator import BaseMetric
from mmcls.registry import METRICS


@METRICS.register_module()
class MultiTasksMetric(BaseMetric):
    """a."""

    def __init__(self,
                 task_metrics: Dict,
                 collect_device: str = 'cpu') -> None:
        self.task_metrics = task_metrics
        super().__init__(collect_device=collect_device)

        for task_name in self.task_metrics.keys():
            for metric in self.task_metrics[task_name]:
                setattr(self, metric['type']+'_'+task_name, METRICS.build(
                    metric))

    def process(self, data_batch, data_samples: Sequence[dict]):
        """Process one batch of data samples.

        The processed results should be stored in ``self.results``, which will
        be used to computed the metrics when all batches have been processed.
        Args:
            data_batch: A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for task_name in self.task_metrics.keys():
            task_data_samples = []
            for data_sample in data_samples:
                task_data_sample = {'gt_label': {}, 'pred_label': {}}
                if task_name in data_sample['gt_task']:
                    task_data_sample['gt_label']['label'] = torch.tensor(
                        data_sample['gt_task'][task_name])
                if task_name in data_sample['pred_task']:
                    score = data_sample['pred_task'][task_name]
                    task_data_sample['pred_label']['score'] = score
                if task_data_sample['gt_label']:
                    task_data_samples.append(task_data_sample)
            for metric in self.task_metrics[task_name]:
                class_ = getattr(
                    MultiTasksMetric(self.task_metrics),
                    metric['type']+'_'+task_name
                )
                class_.process(data_batch, task_data_samples)

    def compute_metrics(self, results: List):
        """Compute the metrics from processed results.

        Args:
            results (dict): The processed results of each batch.
        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        # NOTICE: don't access `self.results` from the method.
        Output = {}
        for task_name in self.task_metrics.keys():
            for metric in self.task_metrics[task_name]:
                name_metric = metric['type']+'_'+task_name
                class_ = getattr(
                    MultiTasksMetric(self.task_metrics),
                    name_metric
                    )
                Output[name_metric] = class_.compute_metrics(class_.results)

        return Output

    def evaluate(self, size):
        metrics = {}
        for task_name in self.task_metrics:
            for metric in self.task_metrics[task_name]:
                class_ = getattr(
                    MultiTasksMetric(self.task_metrics),
                    metric['type']+'_'+task_name)
                results = class_.evaluate(size)
                for key, value in results:
                    name = f'{task_name}_{key}'
                    if name in results:
                        """Inspired from https://github.com/open-mmlab/mmengine/ bl
                        ob/ed20a9cba52ceb371f7c825131636b9e2747172e/mmengine/evalua
                        tor/evaluator.py#L84-L87."""
                        raise ValueError(
                            'There are multiple metric results with the same'
                            f'metric name {name}. Please make sure all metrics'
                            'have different prefixes.')
                    metrics[name] = value
            return metrics
