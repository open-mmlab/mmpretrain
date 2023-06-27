# Copyright (c) OpenMMLab. All rights reserved.
import re
from typing import List, Optional

from mmengine.evaluator import BaseMetric

from mmpretrain.registry import METRICS
from .vqa import _process_digit_article, _process_punctuation


@METRICS.register_module()
class ChartQARelaxACC(BaseMetric):
    '''ChartQARelaxACC.
    Args:

        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Should be modified according to the
            `retrieval_type` for unambiguous results. Defaults to TR.
    '''
    default_prefix = 'ChartQARelaxACC'

    def __init__(self,
                 full_score_weight: float = 0.3,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 relax_thresh: float = 0.05):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.full_score_weight = full_score_weight
        self.relax_thresh = relax_thresh

    def is_digit(self, x: str):
        a = bool(re.match(r'^[+-]?\d+\.\d+$', x))
        b = str(x).isnumeric()
        return any([a, b])

    def process(self, data_batch, data_samples):
        """Process one batch of data samples.

        The processed results should be stored in ``self.results``, which will
        be used to computed the metrics when all batches have been processed.

        Args:
            data_batch: A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for sample in data_samples:
            gt_answer = sample.get('gt_answer')
            sub_set = sample.get('sub_set')

            is_digit = self.is_digit(gt_answer)

            result = {
                'pred_answer': sample.get('pred_answer'),
                'gt_answer': gt_answer,
                'is_digit': is_digit,
                'sub_set': sub_set
            }

            self.results.append(result)

    def compute_metrics(self, results: List):
        """Compute the metrics from processed results.

        Args:
            results (dict): The processed results of each batch.

        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        ChartQA_H_acc = []
        ChartQA_M_acc = []
        for result in results:
            pred_answer = self._process_answer(result['pred_answer'])
            gt_answer = result['gt_answer']
            is_digit = result['is_digit']
            sub_set = result['sub_set']

            if is_digit:
                if self.is_digit(pred_answer):
                    pred_answer = float(pred_answer)
                    gt_answer = float(gt_answer)
                    upper_bound = \
                        max(gt_answer - gt_answer * self.relax_thresh,
                            gt_answer + gt_answer * self.relax_thresh)
                    lower_bound = \
                        min(gt_answer - gt_answer * self.relax_thresh,
                            gt_answer + gt_answer * self.relax_thresh)
                    chart_acc = float(
                        all([
                            pred_answer >= lower_bound,
                            pred_answer <= upper_bound
                        ]))
                else:
                    chart_acc = 0.0
            else:
                chart_acc = float(pred_answer == gt_answer)

            if sub_set == 'ChartQA-H':
                ChartQA_H_acc.append(chart_acc)
            elif sub_set == 'ChartQA-M':
                ChartQA_M_acc.append(chart_acc)
            else:
                raise ValueError(f'Do not support to subset {sub_set}.')

        ChartQA_H_acc = sum(ChartQA_H_acc) / len(ChartQA_H_acc) * 100
        ChartQA_M_acc = sum(ChartQA_M_acc) / len(ChartQA_M_acc) * 100

        accuracy = (ChartQA_H_acc + ChartQA_M_acc) / 2

        metrics = {
            'ChartQA-H acc': ChartQA_H_acc,
            'ChartQA-M acc': ChartQA_M_acc,
            'Overall acc': accuracy
        }

        return metrics

    def _process_answer(self, answer):
        answer = answer.replace('\n', ' ')
        answer = answer.replace('\t', ' ')
        answer = answer.strip()
        answer = _process_punctuation(answer)
        answer = _process_digit_article(answer)
        return answer
