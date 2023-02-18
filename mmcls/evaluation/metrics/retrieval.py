# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Sequence, Union

import mmengine
import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmengine.structures import LabelData
from mmengine.utils import is_seq_of

from mmcls.registry import METRICS
from .single_label import to_tensor


@METRICS.register_module()
class RetrievalRecall(BaseMetric):
    r"""Recall evaluation metric for image retrieval.

    Args:
        topk (int | Sequence[int]): If the ground truth label matches one of
            the best **k** predictions, the sample will be regard as a positive
            prediction. If the parameter is a tuple, all of top-k recall will
            be calculated and outputted together. Defaults to 1.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.

    Examples:
        Use in the code:

        >>> import torch
        >>> from mmcls.evaluation import RetrievalRecall
        >>> # -------------------- The Basic Usage --------------------
        >>> y_pred = [[0], [1], [2], [3]]
        >>> y_true = [[0, 1], [2], [1], [0, 3]]
        >>> RetrievalRecall.calculate(
        >>>     y_pred, y_true, topk=1, pred_indices=True, target_indices=True)
        [tensor([50.])]
        >>> # Calculate the recall@1 and recall@5 for non-indices input.
        >>> y_score = torch.rand((1000, 10))
        >>> import torch.nn.functional as F
        >>> y_true = F.one_hot(torch.arange(0, 1000) % 10, num_classes=10)
        >>> RetrievalRecall.calculate(y_score, y_true, topk=(1, 5))
        [tensor(9.3000), tensor(48.4000)]
        >>>
        >>> # ------------------- Use with Evalutor -------------------
        >>> from mmcls.structures import ClsDataSample
        >>> from mmengine.evaluator import Evaluator
        >>> data_samples = [
        ...     ClsDataSample().set_gt_label([0, 1]).set_pred_score(
        ...     torch.rand(10))
        ...     for i in range(1000)
        ... ]
        >>> evaluator = Evaluator(metrics=RetrievalRecall(topk=(1, 5)))
        >>> evaluator.process(data_samples)
        >>> evaluator.evaluate(1000)
        {'retrieval/Recall@1': 20.700000762939453,
         'retrieval/Recall@5': 78.5999984741211}

        Use in OpenMMLab configs:

        .. code:: python

            val/test_evaluator = dict(type='RetrievalRecall', topk=(1, 5))
    """
    default_prefix: Optional[str] = 'retrieval'

    def __init__(self,
                 topk: Union[int, Sequence[int]],
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        topk = (topk, ) if isinstance(topk, int) else topk

        for k in topk:
            if k <= 0:
                raise ValueError('`topk` must be a ingter larger than 0 '
                                 'or seq of ingter larger than 0.')

        self.topk = topk
        super().__init__(collect_device=collect_device, prefix=prefix)

    def process(self, data_batch: Sequence[dict],
                data_samples: Sequence[dict]):
        """Process one batch of data and predictions.

        The processed results should be stored in ``self.results``, which will
        be used to computed the metrics when all batches have been processed.

        Args:
            data_batch (Sequence[dict]): A batch of data from the dataloader.
            predictions (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            pred_label = data_sample['pred_label']
            gt_label = data_sample['gt_label']

            pred = pred_label['score'].clone()
            if 'score' in gt_label:
                target = gt_label['score'].clone()
            else:
                num_classes = pred_label['score'].size()[-1]
                target = LabelData.label_to_onehot(gt_label['label'],
                                                   num_classes)

            # Because the retrieval output logit vector will be much larger
            # compared to the normal classification, to save resources, the
            # evaluation results are computed each batch here and then reduce
            #  all results at the end.
            result = RetrievalRecall.calculate(
                pred.unsqueeze(0), target.unsqueeze(0), topk=self.topk)
            self.results.append(result)

    def compute_metrics(self, results: List):
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        result_metrics = dict()
        for i, k in enumerate(self.topk):
            recall_at_k = sum([r[i].item() for r in results]) / len(results)
            result_metrics[f'Recall@{k}'] = recall_at_k

        return result_metrics

    @staticmethod
    def calculate(pred: Union[np.ndarray, torch.Tensor],
                  target: Union[np.ndarray, torch.Tensor],
                  topk: Union[int, Sequence[int]],
                  pred_indices: (bool) = False,
                  target_indices: (bool) = False) -> float:
        """Calculate the average recall.

        Args:
            pred (torch.Tensor | np.ndarray | Sequence): The prediction
                results. A :obj:`torch.Tensor` or :obj:`np.ndarray` with
                shape ``(N, M)`` or a sequence of index/onehot
                format labels.
            target (torch.Tensor | np.ndarray | Sequence): The prediction
                results. A :obj:`torch.Tensor` or :obj:`np.ndarray` with
                shape ``(N, M)`` or a sequence of index/onehot
                format labels.
            topk (int, Sequence[int]): Predictions with the k-th highest
                scores are considered as positive.
            pred_indices (bool): Whether the ``pred`` is a sequence of
                category index labels. Defaults to False.
            target_indices (bool): Whether the ``target`` is a sequence of
                category index labels. Defaults to False.

        Returns:
            List[float]: the average recalls.
        """
        topk = (topk, ) if isinstance(topk, int) else topk
        for k in topk:
            if k <= 0:
                raise ValueError('`topk` must be a ingter larger than 0 '
                                 'or seq of ingter larger than 0.')

        max_keep = max(topk)
        pred = _format_pred(pred, max_keep, pred_indices)
        target = _format_target(target, target_indices)

        assert len(pred) == len(target), (
            f'Length of `pred`({len(pred)}) and `target` ({len(target)}) '
            f'must be the same.')

        num_samples = len(pred)
        results = []
        for k in topk:
            recalls = torch.zeros(num_samples)
            for i, (sample_pred,
                    sample_target) in enumerate(zip(pred, target)):
                sample_pred = np.array(to_tensor(sample_pred).cpu())
                sample_target = np.array(to_tensor(sample_target).cpu())
                recalls[i] = int(np.in1d(sample_pred[:k], sample_target).max())
            results.append(recalls.mean() * 100)
        return results


def _format_pred(label, topk=None, is_indices=False):
    """format various label to List[indices]."""
    if is_indices:
        assert isinstance(label, Sequence),  \
                '`pred` must be Sequence of indices when' \
                f' `pred_indices` set to True, but get {type(label)}'
        for i, sample_pred in enumerate(label):
            assert is_seq_of(sample_pred, int) or isinstance(
                sample_pred, (np.ndarray, torch.Tensor)), \
                '`pred` should be Sequence of indices when `pred_indices`' \
                f'set to True. but pred[{i}] is {sample_pred}'
            if topk:
                label[i] = sample_pred[:min(topk, len(sample_pred))]
        return label
    if isinstance(label, np.ndarray):
        label = torch.from_numpy(label)
    elif not isinstance(label, torch.Tensor):
        raise TypeError(f'The pred must be type of torch.tensor, '
                        f'np.ndarray or Sequence but get {type(label)}.')
    topk = topk if topk else label.size()[-1]
    _, indices = label.topk(topk)
    return indices


def _format_target(label, is_indices=False):
    """format various label to List[indices]."""
    if is_indices:
        assert isinstance(label, Sequence),  \
                '`target` must be Sequence of indices when' \
                f' `target_indices` set to True, but get {type(label)}'
        for i, sample_gt in enumerate(label):
            assert is_seq_of(sample_gt, int) or isinstance(
                sample_gt, (np.ndarray, torch.Tensor)), \
                '`target` should be Sequence of indices when ' \
                f'`target_indices` set to True. but target[{i}] is {sample_gt}'
        return label

    if isinstance(label, np.ndarray):
        label = torch.from_numpy(label)
    elif isinstance(label, Sequence) and not mmengine.is_str(label):
        label = torch.tensor(label)
    elif not isinstance(label, torch.Tensor):
        raise TypeError(f'The pred must be type of torch.tensor, '
                        f'np.ndarray or Sequence but get {type(label)}.')

    indices = [LabelData.onehot_to_label(sample_gt) for sample_gt in label]
    return indices
