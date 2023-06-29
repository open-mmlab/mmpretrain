# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import torch
from mmengine.evaluator import BaseMetric

from mmpretrain.evaluation.metrics.vqa import (_process_digit_article,
                                               _process_punctuation)
from mmpretrain.registry import METRICS


@METRICS.register_module()
class SparseGTMetrics(BaseMetric):
    """Visual Dialog Acc metric.

    Compute Visual Dialogaccuracy.

    Args:
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Should be modified according to the
            `retrieval_type` for unambiguous results. Defaults to TR.
    """
    default_prefix = 'Visual Dialog'

    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

    def process(self, data_batch, data_samples) -> None:
        """Process one batch of data samples.

        The processed results should be stored in ``self.results``, which will
        be used to computed the metrics when all batches have been processed.

        Args:
            data_batch: A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for sample in data_samples:
            answer_options = sample.get('answer_options')
            
            G = torch.Generator()
            G.manual_seed(0)
            rank = 1 + torch.randperm(len(answer_options), generator=G)
   
            pred_answer = sample.get('pred_answer')
            
            if pred_answer in answer_options:
                answer_index = answer_options.index(pred_answer)
                rank[answer_index] = 1
    
          

            gt_index = sample.get('gt_answer_index')
            gt_rank = rank[gt_index]

            self.results.append(gt_rank)

    def compute_metrics(self, results: List) -> dict:
        """Compute the metrics from processed results.

        Args:
            results (dict): The processed results of each batch.

        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """

        R1 = (torch.tensor(results) <= 1).float().mean()
        R5 = (torch.tensor(results) <= 5).float().mean()
        R10 = (torch.tensor(results) <= 10).float().mean()
        Mean = torch.tensor(results).float().mean()
        MRR = torch.tensor(results).reciprocal().mean()

        metrics = {'R@1': R1.item(), 'R@5': R5.item(), 'R@10': R10.item(), 'Mean': Mean.item(), 'MRR': MRR.item()}
        return metrics

    def _process_answer(self, answer) -> str:
        answer = _process_punctuation(answer)
        answer = _process_digit_article(answer)
        return answer
