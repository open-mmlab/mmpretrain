# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Sequence, Union

import mmengine
import numpy as np
import torch

from mmcls.registry import METRICS
from .single_label import Accuracy, to_tensor


@METRICS.register_module()
class RetrieverRecall(Accuracy):
    r"""Recall evaluation metric for Retriever.
    Args:
        topk (int | Sequence[int]): If the ground truth label matches one of
            the best **k** predictions, the sample will be regard as a positive
            prediction. If the parameter is a tuple, all of top-k recall will
            be calculated and outputted together. Defaults to 1.
        thrs (Sequence[float | None] | float | None): If a float, predictions
            with score lower than the threshold will be regard as the negative
            prediction. If None, not apply threshold. If the parameter is a
            tuple, recall based on all thresholds will be calculated and
            outputted together. Defaults to 0.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    Examples:
        >>> import torch
        >>> from mmcls.evaluation import RetrieverRecall
        >>> # -------------------- The Basic Usage --------------------
        >>> y_pred = [0, 1, 2, 3]
        >>> y_true = [[0, 1], [2], [1], [0, 3]]
        >>> RetrieverRecall.calculate(y_pred, y_true)
        tensor([50.])
        >>> # Calculate the top1 and top5 recall.
        >>> y_score = torch.rand((1000, 10))
        >>> y_true = [[0, 1]] * 1000
        >>> RetrieverRecall.calculate(y_score, y_true, topk=(1, 5))
        [[tensor([20.6000])], [tensor([76.7000])]]
        >>>
        >>> # ------------------- Use with Evalutor -------------------
        >>> from mmcls.structures import ClsDataSample
        >>> from mmengine.evaluator import Evaluator
        >>> data_samples = [
        ...     ClsDataSample().set_gt_label([0, 1]).set_pred_score(
        ...     torch.rand(10))
        ...     for i in range(1000)
        ... ]
        >>> evaluator = Evaluator(metrics=RetrieverRecall(topk=(1, 5)))
        >>> evaluator.process(data_samples)
        >>> evaluator.evaluate(1000)
        {
            'recall/top1': 20.899999618530273,
            'recall/top5': 75.70000457763672
        }
    """
    default_prefix: Optional[str] = 'recall'

    def compute_metrics(self, results: List):
        """Compute the metrics from processed results.

        Args:
            results (dict): The processed results of each batch.
        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        # NOTICE: don't access `self.results` from the method.
        metrics = {}

        # concat
        target = [res['gt_label'] for res in results]
        if 'pred_score' in results[0]:
            pred = torch.stack([res['pred_score'] for res in results])

            try:
                recall = self.calculate(pred, target, self.topk, self.thrs)
            except ValueError as e:
                # If the topk is invalid.
                raise ValueError(
                    str(e) + ' Please check the `val_evaluator` and '
                    '`test_evaluator` fields in your config file.')

            multi_thrs = len(self.thrs) > 1
            for i, k in enumerate(self.topk):
                for j, thr in enumerate(self.thrs):
                    name = f'top{k}'
                    if multi_thrs:
                        name += '_no-thr' if thr is None else f'_thr-{thr:.2f}'
                    metrics[name] = recall[i][j].item()
        else:
            # If only label in the `pred_label`.
            pred = torch.cat([res['pred_label'] for res in results])
            recall = self.calculate(pred, target, self.topk, self.thrs)
            metrics['top1'] = recall.item()

        return metrics

    @staticmethod
    def calculate(
        pred: Union[torch.Tensor, np.ndarray, Sequence],
        target: Sequence,
        topk: Sequence[int] = (1, ),
        thrs: Sequence[Union[float, None]] = (0., ),
    ) -> Union[torch.Tensor, List[List[torch.Tensor]]]:
        """Calculate the recall.
        Args:
            pred (torch.Tensor | np.ndarray | Sequence): The prediction
                results. It can be labels (N, ), or scores of every
                class (N, C).
            target (Sequence): The target of each prediction with shape (N, ).
            thrs (Sequence[float | None]): Predictions with scores under
                the thresholds are considered negative. It's only used
                when ``pred`` is scores. None means no thresholds.
                Defaults to (0., ).
            thrs (Sequence[float]): Predictions with scores under
                the thresholds are considered negative. It's only used
                when ``pred`` is scores. Defaults to (0., ).
        Returns:
            torch.Tensor | List[List[torch.Tensor]]: Recall.
            - torch.Tensor: If the ``pred`` is a sequence of label instead of
              score (number of dimensions is 1). Only return a top-1 recall
              tensor, and ignore the argument ``topk` and ``thrs``.
            - List[List[torch.Tensor]]: If the ``pred`` is a sequence of score
              (number of dimensions is 2). Return the recall on each ``topk``
              and ``thrs``. And the first dim is ``topk``, the second dim is
              ``thrs``.
        """
        if not (isinstance(target, Sequence) and not mmengine.is_str(target)):
            raise TypeError(f'{type(target)} is not an available argument.')

        pred = to_tensor(pred)
        num = len(pred)
        assert len(pred) == len(target), \
            f"The size of pred ({len(pred)}) doesn't match "\
            f'the target ({len(target)}).'

        if pred.ndim == 1:
            # For pred label, ignore topk and recall
            recalls = []
            for p, t in zip(pred, target):
                if p in t:
                    recalls.append(1)
                else:
                    recalls.append(0)
            return torch.Tensor([np.mean(recalls) * 100])
        else:
            # For pred score, calculate on all topk and thresholds.
            pred = pred.float()
            maxk = max(topk)

            if maxk > pred.size(1):
                raise ValueError(
                    f'Top-{maxk} recall is unavailable since the number of '
                    f'categories is {pred.size(1)}.')

            pred_score, pred_label = pred.topk(maxk, dim=1)
            correct = [[] for _ in range(maxk)]
            for labels, t in zip(pred_label, target):
                for i, label in enumerate(labels):
                    if label in t:
                        correct[i].append(1)
                    else:
                        correct[i].append(0)
            correct = to_tensor(correct)
            results = []
            for k in topk:
                results.append([])
                for thr in thrs:
                    # Only prediction values larger than thr are counted
                    # as correct
                    _correct = correct
                    if thr is not None:
                        _correct = _correct & (pred_score.t() > thr)
                    correct_k = _correct[:k].max(dim=0)[0]
                    correct_k = correct_k.reshape(-1).float().sum(
                        0, keepdim=True)
                    recall = correct_k.mul_(100. / num)
                    results[-1].append(recall)
            return results
