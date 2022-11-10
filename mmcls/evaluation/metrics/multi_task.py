# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Sequence, List
from mmcls.registry import METRICS
from mmengine.registry import METRICS as mmengineMetrics
from mmengine.evaluator import BaseMetric


@METRICS.register_module()
class MultiTasks(BaseMetric):
    """

    """

    def __init__(self,
                 task_metrics: Dict,
                 collect_device: str = 'cpu') -> None:
        self.task_metrics = task_metrics
        super().__init__(collect_device=collect_device)

        for task_name in self.task_metrics.keys():
            globals()['metric_%s' % task_name] = mmengineMetrics.build(
                self.task_metrics[task_name])

    def process(self, data_batch, data_samples: Sequence[dict]):
        """
        Process one batch of data samples.
        The processed results should be stored in ``self.results``, which will
        be used to computed the metrics when all batches have been processed.
        Args:
            data_batch: A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for task_name in self.task_metrics.keys():
            index = list(self.task_metrics.keys()).index(task_name)
            task_data_samples = []
            for data_sample in data_samples.copy():
                if task_name in data_sample['gt_label'].keys():
                    data_sample['gt_label'] = {
                        'label': data_sample['gt_label'][task_name]
                    }
                    if len(data_sample['pred_label']['score']) < 3:
                        raise Exception('score masked')
                    data_sample['pred_label']['score'] = data_sample[
                        'pred_label']['score'][index]
                    task_data_samples.append(data_sample)
            globals()["metric_%s" % task_name].process(data_batch,
                                                       task_data_samples)

    def compute_metrics(self, results: List):
        """
        Compute the metrics from processed results.
        Args:
            results (dict): The processed results of each batch.

        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        # NOTICE: don't access `self.results` from the method.
        Output = {}
        print(results)
        for task_name in self.task_metrics.keys():
            Output[f'metric_{task_name}'] = globals()[
                "metric_%s" % task_name].compute_metrics(results)

        return Output

    def evaluate(self, size):
        metrics = {}
        for task_name in self.task_metrics:
            results = globals()["metric_%s" % task_name].evaluate(size)
            for key, value in results:
                name = f'{task_name}_{key}'
                if name in results:
                    """
                     Inspired from https://github.com/open-mmlab/mmengine/
                     blob/ed20a9cba52ceb371f7c825131636b9e2747172e/mmengine/evaluator/evaluator.py#L84-L87
                    """
                    raise ValueError(
                        'There are multiple metric results with the same '
                        f'metric name {name}. Please make sure all metrics '
                        'have different prefixes.')
            metrics[name] = value
            return metrics
