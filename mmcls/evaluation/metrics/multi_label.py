# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from mmengine.logging import MMLogger
from mmeval.metrics import AveragePrecision as _AveragePrecision
from mmeval.metrics import MultiLabelMetric as _MultiLabelMetric

from mmcls.registry import METRICS
from .single_label import to_tensor


class MultiLabelMixin:

    default_prefix: Optional[str] = 'multi-label'

    def __init__(self,
                 prefix: Optional[str] = None,
                 collect_device: Optional[str] = None) -> None:
        logger = MMLogger.get_current_instance()
        self.prefix = prefix or self.default_prefix

        if collect_device is not None:
            logger.warning(
                'DeprecationWarning: The `collect_device` parameter of '
                '`Accuracy` is deprecated, use `dist_backend` instead.')

    def process(self, data_batch, data_samples: Sequence[dict]):
        """Process one batch of data samples.

        The processed results should be stored in ``self.results``, which will
        be used to computed the metrics when all batches have been processed.

        Args:
            data_batch: A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        preds, targets = list(), list()

        for data_sample in data_samples:
            pred_label = data_sample['pred_label']
            gt_label = data_sample['gt_label']
            if 'score' in pred_label:
                preds.append(pred_label['score'].cpu())
            else:
                preds.append(pred_label['label'].reshape(-1).cpu().squeeze())

            if 'score' in gt_label:
                targets.append(gt_label['score'].cpu())
            else:
                targets.append(gt_label['label'].cpu().squeeze())

        # add preds and targets
        self.add(preds, targets)

    def evaluate(self, *args, **kwargs) -> Dict:
        """Returns metric results and reset state.

        This method would be invoked by ``mmengine.Evaluator``.
        """
        metric_results = self.compute(*args, **kwargs)
        metric_results = {
            f'{self.prefix}/{k}': v
            for k, v in metric_results.items()
        }
        self.reset()
        return metric_results

    def calculate(
        self,
        preds: Union[torch.Tensor, np.ndarray, Sequence],
        targets: Union[torch.Tensor, np.ndarray, Sequence],
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Calculate the precision, recall, f1-score.

        Args:
            preds (torch.Tensor | np.ndarray | Sequence): The prediction
                results. A :obj:`torch.Tensor` or :obj:`np.ndarray` with
                shape ``(N, num_classes)`` or a sequence of index/onehot
                format labels.
            targets (torch.Tensor | np.ndarray | Sequence): The prediction
                results. A :obj:`torch.Tensor` or :obj:`np.ndarray` with
                shape ``(N, num_classes)`` or a sequence of index/onehot
                format labels.

        Returns:
            Tuple: The tuple contains precision, recall and f1-score.
            And the type of each item is:

            - torch.Tensor: A tensor for each metric. The shape is (1, ) if
              ``average`` is not None, and (C, ) if ``average`` is None.

        Notes:
            If both ``thr`` and ``topk`` are set, use ``thr` to determine
            positive predictions. If neither is set, use ``thr=0.5`` as
            default.
        """
        preds = [to_tensor(pred) for pred in preds]
        targets = [to_tensor(target).to(torch.int64) for target in targets]
        results = self._compute_metric(preds, targets)
        if len(results) == 1:
            return results[0]
        else:
            return results


@METRICS.register_module()
class MultiLabelMetric(_MultiLabelMetric, MultiLabelMixin):
    """A collection of metrics for multi-label multi-class classification task
    based on confusion matrix.

    It includes precision, recall, f1-score and support.

    Args:
        thr (float, optional): Predictions with scores under the thresholds
            are considered as negative. Defaults to None.
        topk (int, optional): Predictions with the k-th highest scores are
            considered as positive. Defaults to None.
        items (Sequence[str]): The detailed metric items to evaluate. Here is
            the available options:

                - `"precision"`: The ratio tp / (tp + fp) where tp is the
                  number of true positives and fp the number of false
                  positives.
                - `"recall"`: The ratio tp / (tp + fn) where tp is the number
                  of true positives and fn the number of false negatives.
                - `"f1-score"`: The f1-score is the harmonic mean of the
                  precision and recall.
                - `"support"`: The total number of positive of each category
                  in the target.

            Defaults to ('precision', 'recall', 'f1-score').
        average (str | None): The average method. It supports three average
            modes:

                - `"macro"`: Calculate metrics for each category, and calculate
                  the mean value over all categories.
                - `"micro"`: Calculate metrics globally by counting the total
                  true positives, false negatives and false positives.
                - `None`: Return scores of all categories.

            Defaults to "macro".
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.

    Examples:
        >>> import torch
        >>> from mmcls.evaluation import MultiLabelMetric
        >>> # ------ The Basic Usage for category indices labels -------
        >>> y_pred = [[0], [1], [0, 1], [3]]
        >>> y_true = [[0, 3], [0, 2], [1], [3]]
        >>> metric = MultiLabelMetric(num_classes=4)
        >>> # Output precision, recall, f1-score and support
        >>> metric.calculate(y_pred, y_true)
        (tensor(50.), tensor(50.), tensor(45.8333), tensor(6))

        >>> # ----------- The Basic Usage for one-hot labels -----------
        >>> y_pred = torch.tensor([[1, 1, 0, 0],
        ...                        [1, 1, 0, 0],
        ...                        [0, 0, 1, 0],
        ...                        [0, 1, 0, 0],
        ...                        [0, 1, 0, 0]])
        >>> y_true = torch.Tensor([[1, 1, 0, 0],
        ...                        [0, 0, 1, 0],
        ...                        [1, 1, 1, 0],
        ...                        [1, 0, 0, 0],
        ...                        [1, 0, 0, 0]])
        >>> metric.calculate(y_pred, y_true)
        (tensor(43.7500), tensor(31.2500), tensor(33.3333), tensor(8))

        >>> # --------- The Basic Usage for one-hot pred scores ---------
        >>> y_pred = torch.rand(y_true.size())
        >>> # Calculate with different threshold.
        >>> metric = MultiLabelMetric(num_classes=4, thr=0.1)
        >>> metric.calculate(y_pred, y_true)  # doctest: +ELLIPSIS
        (tensor(...), ...)

        >>> # Calculate with topk.
        >>> metric = MultiLabelMetric(num_classes=4, topk=1)
        >>> metric.calculate(y_pred, y_true)  # doctest: +ELLIPSIS
        (tensor(...), ...)

        >>> # ------------------- Use with Evalutor -------------------
        >>> from mmcls.structures import ClsDataSample
        >>> from mmengine.evaluator import Evaluator
        >>> data_sampels = [
        ...     ClsDataSample().set_pred_score(pred).set_gt_score(gt)
        ...     for pred, gt in zip(torch.rand(1000, 5), torch.randint(0, 2, (1000, 5)))]
        >>> evaluator = Evaluator(metrics=MultiLabelMetric(thr=0.5, num_classes=5))
        >>> evaluator.process(data_sampels)
        >>> evaluator.evaluate(1000)  # doctest: +ELLIPSIS
        {
            'multi-label/precision': ...,
            'multi-label/recall': ...,
            'multi-label/f1-score': ...
        }

        >>> # Evaluate on each class by using topk strategy
        >>> evaluator = Evaluator(metrics=MultiLabelMetric(topk=1, average=None, num_classes=5))
        >>> evaluator.process(data_sampels)
        >>> evaluator.evaluate(1000)  # doctest: +ELLIPSIS
        {
            'multi-label/precision_top1_classwise': [...],
            'multi-label/recall_top1_classwise': [...],
            'multi-label/f1-score_top1_classwise': [...]
        }
    """  # noqa: E501

    def __init__(self, prefix: Optional[str] = None, *arg, **kwargs) -> None:

        collect_device = kwargs.pop('collect_device', None)

        super().__init__(*arg, **kwargs)
        MultiLabelMixin.__init__(self, prefix, collect_device)


@METRICS.register_module()
class AveragePrecision(_AveragePrecision, MultiLabelMixin):
    """Calculate the average precision with respect of classes.

    Args:
        average (str | None): The average method. It supports two modes:

                - `"macro"`: Calculate metrics for each category, and calculate
                  the mean value over all categories.
                - `None`: Return scores of all categories.

            Defaults to "macro".
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.

    References
    ----------
    .. [1] `Wikipedia entry for the Average precision
           <https://en.wikipedia.org/w/index.php?title=Information_retrieval&
           oldid=793358396#Average_precision>`_

    Examples:
        >>> import torch
        >>> from mmcls.evaluation import AveragePrecision
        >>> # --------- The Basic Usage for one-hot pred scores ---------
        >>> y_pred = torch.Tensor([[0.9, 0.8, 0.3, 0.2],
        ...                        [0.1, 0.2, 0.2, 0.1],
        ...                        [0.7, 0.5, 0.9, 0.3],
        ...                        [0.8, 0.1, 0.1, 0.2]])
        >>> y_true = torch.Tensor([[1, 1, 0, 0],
        ...                        [0, 1, 0, 0],
        ...                        [0, 0, 1, 0],
        ...                        [1, 0, 0, 0]])
        >>> metric = AveragePrecision()
        >>> metric.calculate(y_pred, y_true)
        tensor(70.833)

        >>> # ------------------- Use with Evalutor -------------------
        >>> from mmcls.structures import ClsDataSample
        >>> from mmengine.evaluator import Evaluator
        >>> data_samples = [
        ...     ClsDataSample().set_pred_score(i).set_gt_score(j)
        ...     for i, j in zip(y_pred, y_true)
        ... ]
        >>> evaluator = Evaluator(metrics=AveragePrecision())
        >>> evaluator.process(data_samples)
        >>> evaluator.evaluate(5)
        {'multi-label/mAP': 70.83333587646484}

        >>> # Evaluate on each class
        >>> evaluator = Evaluator(metrics=AveragePrecision(average=None))
        >>> evaluator.process(data_samples)
        >>> evaluator.evaluate(5)
        {'multi-label/AP_classwise': [100., 83.33, 100., 0.]}
    """

    def __init__(self, prefix: Optional[str] = None, *arg, **kwargs) -> None:

        collect_device = kwargs.pop('collect_device', None)

        super().__init__(*arg, **kwargs)
        MultiLabelMixin.__init__(self, prefix, collect_device)
