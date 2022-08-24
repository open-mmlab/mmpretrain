# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
from mmengine import LabelData, MMLogger
from mmengine.evaluator import BaseMetric

from mmcls.registry import METRICS
from .single_label import _precision_recall_f1_support, to_tensor


@METRICS.register_module()
class MultiLabelMetric(BaseMetric):
    """A collection of metrics for multi-label multi-class classification task
    based on confusion matrix.

    It includes CP, CR, CF1, OP, OR, OF1. The 'C' stands for per-class average,
    just as the average mode of "macro". The 'O' stands for overall average,
    just as the average mode of "micro". P stands for precision, R stands for
    recall and F1 stands for F1-score.

    Notes:
        The metric CF1 (macro-F1-score) is equals to ``(2 * CP * CR) / (CP + CR)``,
         which is widely used in papers. It is different with
         `sklearn.metrics.f1_score(average="macro")`, which equals to
         ``mean(classwise-F1-score_list)``.

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

            Defaults to ('precision', 'recall', 'f1-score').
        average (str | None): The average method. It supports three average
            modes:

                - `"macro"`: Calculate metrics for each category, and calculate
                  the mean value over all categories, including CP, CR and CF1.
                - `"micro"`: Calculate metrics globally by counting the total
                  true positives, false negatives and false positives, including
                  OP, OR and OF1.
                - `"both"`: Calculate metrics both in "macro" and "micro"
                  modes, including CP, CR, CF1, OP, OR and OF1.
                - `None`: Return scores of all categories.

            Defaults to "both".
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.

    Examples:
        >>> import torch
        >>> from mmcls.metrics import MultiLabelMetric
        >>> # ------ The Basic Usage for category indices labels -------
        >>> y_pred = [[0], [1], [0, 1], [3]]
        >>> y_true = [[0, 3], [0, 2], [1], [3]]
        >>> # Output "CP", "CR", "CF1", "OP", "OR", "OF1"
        >>> MultiLabelMetric.calculate(
        ...     y_pred, y_true, pred_indices=True, target_indices=True, num_classes=4)
        (tensor(50.), tensor(50.), tensor(50.), tensor(60.0000), tensor(50.), tensor(54.5455))
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
        >>> MultiLabelMetric.calculate(y_pred, y_true)
        (tensor(43.75), tensor(31.25), tensor(36.46), tensor(42.86), tensor(37.50), tensor(40.00))
        >>> # --------- The Basic Usage for one-hot pred scores ---------
        >>> y_pred = torch.rand(y_true.size())
        >>> y_pred
        tensor([[0.4575, 0.7335, 0.3934, 0.2572],
        [0.1318, 0.1004, 0.8248, 0.6448],
        [0.8349, 0.6294, 0.7896, 0.2061],
        [0.4037, 0.7308, 0.6713, 0.8374],
        [0.3779, 0.4836, 0.0313, 0.0067]])
        >>> # Calculate with  threshold.
        >>> MultiLabelMetric.calculate(y_pred, y_true, thr=0.1)
        (tensor(36.67), tensor(62.50), tensor(46.22), tensor(33.33), tensor(75.), tensor(46.15))
        >>> # Calculate with topk.
        >>> MultiLabelMetric.calculate(y_pred, y_true, topk=1)
        (tensor(25.), tensor(12.50), tensor(16.67), tensor(20.), tensor(12.50), tensor(15.38))
        >>>
        >>> # ------------------- Use with Evalutor -------------------
        >>> from mmcls.structures import ClsDataSample
        >>> from mmengine.evaluator import Evaluator
        >>> # The `data_batch` won't be used in this case, just use a fake.
        >>> data_batch = [
        ...     {'inputs': None,  'data_sample': ClsDataSample()}
        ...     for i in range(1000)]
        >>> pred = [
        ...     ClsDataSample().set_pred_score(torch.rand((5, ))).set_gt_score(torch.randint(2, size=(5, )))
        ...     for i in range(1000)]
        >>> evaluator = Evaluator(metrics=MultiLabelMetric(thr=0.5))
        >>> evaluator.process(data_batch, pred)
        >>> evaluator.evaluate(1000)
        {
            'multi-label/CP': 50.85150909423828,
            'multi-label/CR': 48.266883850097656,
            'multi-label/CF1': 49.52549743652344,
            'multi-label/OP': 50.83981704711914,
            'multi-label/OR': 48.26915740966797,
            'multi-label/OF1': 49.52114486694336
        }
        >>> # Evaluate on each class by using topk strategy
        >>> evaluator = Evaluator(metrics=MultiLabelMetric(topk=1, average=None))
        >>> evaluator.process(data_batch, pred)
        >>> evaluator.evaluate(1000)
        {
            'multi-label/precision_top1_classwise': [52.02, 52.46, 51.01, 55.91, 48.11],
            'multi-label/recall_top1_classwise': [21.62, 18.25, 20.04, 20.43, 20.4],
            'multi-label/f1-score_top1_classwise': [30.54, 27.08, 28.77, 29.93, 28.65]
        }
    """  # noqa: E501
    default_prefix: Optional[str] = 'multi-label'

    def __init__(self,
                 thr: Optional[float] = None,
                 topk: Optional[int] = None,
                 items: Sequence[str] = ('precision', 'recall', 'f1-score'),
                 average: Optional[str] = 'both',
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:

        logger = MMLogger.get_current_instance()
        if thr is None and topk is None:
            thr = 0.5
            logger.warning('Neither thr nor k is given, set thr as 0.5 by '
                           'default.')
        elif thr is not None and topk is not None:
            logger.warning('Both thr and topk are given, '
                           'use threshold in favor of top-k.')

        self.thr = thr
        self.topk = topk
        self.average = average

        for item in items:
            assert item in ['precision', 'recall', 'f1-score'], \
                f'The metric {item} is not supported by `MultiLabelMetric`,' \
                ' please choose from "precision", "recall" and "f1-score".'
        self.items = tuple(items)

        super().__init__(collect_device=collect_device, prefix=prefix)

    def process(self, data_batch: Sequence[dict], predictions: Sequence[dict]):
        """Process one batch of data and predictions.

        The processed results should be stored in ``self.results``, which will
        be used to computed the metrics when all batches have been processed.

        Args:
            data_batch (Sequence[dict]): A batch of data from the dataloader.
            predictions (Sequence[dict]): A batch of outputs from the model.
        """
        for pred in predictions:
            result = dict()
            pred_label = pred['pred_label']
            gt_label = pred['gt_label']

            result['pred_score'] = pred_label['score'].clone()
            num_classes = result['pred_score'].size()[-1]

            if 'score' in gt_label:
                result['gt_score'] = gt_label['score'].clone()
            else:
                result['gt_score'] = LabelData.label_to_onehot(
                    gt_label['label'], num_classes)

            # Save the result to `self.results`.
            self.results.append(result)

    def compute_metrics(self, results: List):
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        # NOTICE: don't access `self.results` from the method. `self.results`
        # are a list of results from multiple batch, while the input `results`
        # are the collected results.
        metrics = {}

        target = torch.stack([res['gt_score'] for res in results])
        pred = torch.stack([res['pred_score'] for res in results])

        metric_res = self.calculate(
            pred,
            target,
            pred_indices=False,
            target_indices=False,
            average=self.average,
            thr=self.thr,
            topk=self.topk)

        def pack_results(*args):
            multi_label_metrics = {}
            if self.average == 'both':
                metric_names = ('CP', 'CR', 'CF1', 'OP', 'OR', 'OF1')
            elif self.average == 'macro':
                metric_names = ('CP', 'CR', 'CF1')
            elif self.average == 'micro':
                metric_names = ('OP', 'OR', 'OF1')
            else:
                metric_names = ('precision', 'recall', 'f1-score')

            for i, (name, value) in enumerate(zip(metric_names, args)):
                if 'precision' in self.items and i % 3 == 0:
                    multi_label_metrics[name] = value
                if 'recall' in self.items and i % 3 == 1:
                    multi_label_metrics[name] = value
                if 'f1-score' in self.items and i % 3 == 2:
                    multi_label_metrics[name] = value

            return multi_label_metrics

        if self.thr:
            suffix = '' if self.thr == 0.5 else f'_thr-{self.thr:.2f}'
            for k, v in pack_results(*metric_res).items():
                metrics[k + suffix] = v
        else:
            for k, v in pack_results(*metric_res).items():
                metrics[k + f'_top{self.topk}'] = v

        result_metrics = dict()
        for k, v in metrics.items():
            if self.average is None:
                result_metrics[k + '_classwise'] = v.detach().cpu().tolist()
            else:
                result_metrics[k] = v.item()
        return result_metrics

    @staticmethod
    def calculate(
        pred: Union[torch.Tensor, np.ndarray, Sequence],
        target: Union[torch.Tensor, np.ndarray, Sequence],
        pred_indices: bool = False,
        target_indices: bool = False,
        average: Optional[str] = 'both',
        thr: Optional[float] = None,
        topk: Optional[int] = None,
        num_classes: Optional[int] = None
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Calculate the precision, recall, f1-score.

        Args:
            pred (torch.Tensor | np.ndarray | Sequence): The prediction
                results. A :obj:`torch.Tensor` or :obj:`np.ndarray` with
                shape ``(N, num_classes)`` or a sequence of index/onehot
                format labels.
            target (torch.Tensor | np.ndarray | Sequence): The prediction
                results. A :obj:`torch.Tensor` or :obj:`np.ndarray` with
                shape ``(N, num_classes)`` or a sequence of index/onehot
                format labels.
            pred_indices (bool): Whether the ``pred`` is a sequence of
                category index labels. If True, ``num_classes`` must be set.
                Defaults to False.
            target_indices (bool): Whether the ``target`` is a sequence of
                category index labels. If True, ``num_classes`` must be set.
                Defaults to False.
            average (str | None): The average method. It supports three average
                modes:

                    - `"macro"`: Calculate metrics for each category, and
                      calculate the mean value over all categories.
                    - `"micro"`: Calculate metrics globally by counting the
                      total true positives, false negatives and false
                      positives.
                    - `"both"`: Calculate metrics both in "macro" and "micro"
                      modes.
                    - `None`: Return scores of all categories.

                Defaults to "both".
            thr (float, optional): Predictions with scores under the thresholds
                are considered as negative. Defaults to None.
            topk (int, optional): Predictions with the k-th highest scores are
                considered as positive. Defaults to None.
            num_classes (Optional, int): The number of classes. If the ``pred``
                is indices instead of onehot, this argument is required.
                Defaults to None.

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
        average_options = ['micro', 'macro', 'both', None]
        assert average in average_options, 'Invalid `average` argument, ' \
            f'please specicy from {average_options}.'

        def _format_label(label, is_indices):
            """format various label to torch.Tensor."""
            if isinstance(label, np.ndarray):
                assert label.ndim == 2, 'The shape `pred` and `target` ' \
                    'array must be (N, num_classes).'
                label = torch.from_numpy(label)
            elif isinstance(label, torch.Tensor):
                assert label.ndim == 2, 'The shape `pred` and `target` ' \
                    'tensor must be (N, num_classes).'
            elif isinstance(label, Sequence):
                if is_indices:
                    assert num_classes is not None, 'For index-type labels, ' \
                        'please specify `num_classes`.'
                    label = torch.stack([
                        LabelData.label_to_onehot(
                            to_tensor(indices), num_classes)
                        for indices in label
                    ])
                else:
                    label = torch.stack(
                        [to_tensor(onehot) for onehot in label])
            else:
                raise TypeError(
                    'The `pred` and `target` must be type of torch.tensor or '
                    f'np.ndarray or sequence but get {type(label)}.')
            return label

        pred = _format_label(pred, pred_indices)
        target = _format_label(target, target_indices).long()

        assert pred.shape == target.shape, \
            f"The size of pred ({pred.shape}) doesn't match "\
            f'the target ({target.shape}).'

        if num_classes is not None:
            assert pred.size(1) == num_classes, \
                f'The shape of `pred` ({pred.shape}) '\
                f"doesn't match the num_classes ({num_classes})."
        num_classes = pred.size(1)

        thr = 0.5 if (thr is None and topk is None) else thr

        if thr is not None:
            # a label is predicted positive if larger than thr
            pos_inds = (pred >= thr).long()
        else:
            # top-k labels will be predicted positive for any example
            _, topk_indices = pred.topk(topk)
            pos_inds = torch.zeros_like(pred).scatter_(1, topk_indices, 1)
            pos_inds = pos_inds.long()

        if average is None:
            return _precision_recall_f1_support(pos_inds, target, average)[:3]

        result = ()
        if average in ('macro', 'both'):
            CP, CR, *_ = _precision_recall_f1_support(pos_inds, target,
                                                      'macro')
            # here calculate the CF1 by using CP and CR, which is widely used
            # in most papers. It is different with `sklearn.metrics.f1_score(average="macro")`, # noqa
            # which equals to mean(classwise-F1-score_list).
            CF1 = 2 * CP * CR / (CP + CR + torch.finfo(torch.float32).eps)
            result += (CP, CR, CF1)
        if average in ('micro', 'both'):
            OP, OR, OF1, _, = _precision_recall_f1_support(
                pos_inds, target, 'micro')
            result += (OP, OR, OF1)

        return result


def _average_precision(pred: torch.Tensor,
                       target: torch.Tensor) -> torch.Tensor:
    r"""Calculate the average precision for a single class.

    AP summarizes a precision-recall curve as the weighted mean of maximum
    precisions obtained for any r'>r, where r is the recall:

    .. math::
        \text{AP} = \sum_n (R_n - R_{n-1}) P_n

    Note that no approximation is involved since the curve is piecewise
    constant.

    Args:
        pred (torch.Tensor): The model prediction with shape
            ``(N, num_classes)``.
        target (torch.Tensor): The target of predictions with shape
            ``(N, num_classes)``.

    Returns:
        torch.Tensor: average precision result.
    """
    assert pred.shape == target.shape, \
        f"The size of pred ({pred.shape}) doesn't match "\
        f'the target ({target.shape}).'

    # a small value for division by zero errors
    eps = torch.finfo(torch.float32).eps

    # sort examples
    sorted_pred_inds = torch.argsort(pred, dim=0, descending=True)
    sorted_target = target[sorted_pred_inds]

    # get indexes when gt_true is positive
    pos_inds = sorted_target == 1

    # Calculate cumulative tp case numbers
    tps = torch.cumsum(pos_inds, 0)
    total_pos = tps[-1].item()  # the last of tensor may change later

    # Calculate cumulative tp&fp(pred_poss) case numbers
    pred_pos_nums = torch.arange(1, len(sorted_target) + 1)
    pred_pos_nums[pred_pos_nums < eps] = eps

    tps[torch.logical_not(pos_inds)] = 0
    precision = tps / pred_pos_nums.float()
    ap = torch.sum(precision, 0) / max(total_pos, eps)
    return ap


@METRICS.register_module()
class AveragePrecision(BaseMetric):
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
        >>> from mmcls.metrics import AveragePrecision
        >>> # --------- The Basic Usage for one-hot pred scores ---------
        >>> y_pred = torch.Tensor([[0.9, 0.8, 0.3, 0.2],
        ...                        [0.1, 0.2, 0.2, 0.1],
        ...                        [0.7, 0.5, 0.9, 0.3],
        ...                        [0.8, 0.1, 0.1, 0.2]])
        >>> y_true = torch.Tensor([[1, 1, 0, 0],
        ...                        [0, 1, 0, 0],
        ...                        [0, 0, 1, 0],
        ...                        [1, 0, 0, 0]])
        >>> AveragePrecision.calculate(y_pred, y_true)
        tensor(70.833)
        >>> # ------------------- Use with Evalutor -------------------
        >>> from mmcls.structures import ClsDataSample
        >>> from mmengine.evaluator import Evaluator
        >>> # The `data_batch` won't be used in this case, just use a fake.
        >>> data_batch = [
        ...     {'inputs': None, 'data_sample': ClsDataSample()}
        ...     for i in range(4)]
        >>> pred = [
        ...     ClsDataSample().set_pred_score(i).set_gt_score(j)
        ...     for i, j in zip(y_pred, y_true)
        ... ]
        >>> evaluator = Evaluator(metrics=AveragePrecision())
        >>> evaluator.process(data_batch, pred)
        >>> evaluator.evaluate(5)
        {'multi-label/mAP': 70.83333587646484}
        >>> # Evaluate on each class
        >>> evaluator = Evaluator(metrics=AveragePrecision(average=None))
        >>> evaluator.process(data_batch, pred)
        >>> evaluator.evaluate(5)
        {'multi-label/AP_classwise': [100., 83.33, 100., 0.]}
    """
    default_prefix: Optional[str] = 'multi-label'

    def __init__(self,
                 average: Optional[str] = 'macro',
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.average = average

    def process(self, data_batch: Sequence[dict], predictions: Sequence[dict]):
        """Process one batch of data and predictions.

        The processed results should be stored in ``self.results``, which will
        be used to computed the metrics when all batches have been processed.

        Args:
            data_batch (Sequence[dict]): A batch of data from the dataloader.
            predictions (Sequence[dict]): A batch of outputs from the model.
        """

        for pred in predictions:
            result = dict()
            pred_label = pred['pred_label']
            gt_label = pred['gt_label']

            result['pred_score'] = pred_label['score']
            num_classes = result['pred_score'].size()[-1]

            if 'score' in gt_label:
                result['gt_score'] = gt_label['score']
            else:
                result['gt_score'] = LabelData.label_to_onehot(
                    gt_label['label'], num_classes)

            # Save the result to `self.results`.
            self.results.append(result)

    def compute_metrics(self, results: List):
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        # NOTICE: don't access `self.results` from the method. `self.results`
        # are a list of results from multiple batch, while the input `results`
        # are the collected results.

        # concat
        target = torch.stack([res['gt_score'] for res in results])
        pred = torch.stack([res['pred_score'] for res in results])

        ap = self.calculate(pred, target, self.average)

        result_metrics = dict()

        if self.average is None:
            result_metrics['AP_classwise'] = ap.detach().cpu().tolist()
        else:
            result_metrics['mAP'] = ap.item()

        return result_metrics

    @staticmethod
    def calculate(pred: Union[torch.Tensor, np.ndarray],
                  target: Union[torch.Tensor, np.ndarray],
                  average: Optional[str] = 'macro') -> torch.Tensor:
        r"""Calculate the average precision for a single class.

        AP summarizes a precision-recall curve as the weighted mean of maximum
        precisions obtained for any r'>r, where r is the recall:

        .. math::
            \text{AP} = \sum_n (R_n - R_{n-1}) P_n

        Note that no approximation is involved since the curve is piecewise
        constant.

        Args:
            pred (torch.Tensor | np.ndarray): The model predictions with
                shape ``(N, num_classes)``.
            target (torch.Tensor | np.ndarray): The target of predictions
                with shape ``(N, num_classes)``.
            average (str | None): The average method. It supports two modes:

                    - `"macro"`: Calculate metrics for each category, and
                      calculate the mean value over all categories.
                    - `None`: Return scores of all categories.

                Defaults to "macro".

        Returns:
            torch.Tensor: the average precision of all classes.
        """
        average_options = ['macro', None]
        assert average in average_options, 'Invalid `average` argument, ' \
            f'please specicy from {average_options}.'

        pred = to_tensor(pred)
        target = to_tensor(target)
        assert pred.ndim == 2 and pred.shape == target.shape, \
            'Both `pred` and `target` should have shape `(N, num_classes)`.'

        num_classes = pred.shape[1]
        ap = pred.new_zeros(num_classes)
        for k in range(num_classes):
            ap[k] = _average_precision(pred[:, k], target[:, k])
        if average == 'macro':
            return ap.mean() * 100.0
        else:
            return ap * 100
