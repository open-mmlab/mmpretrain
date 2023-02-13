# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Sequence, Union

import torch

from mmcls.registry import METRICS
from .single_label import Accuracy


def get_ce_alexnet() -> dict:
    """Returns Corruption Error values for AlexNet."""
    ce_alexnet = dict()
    ce_alexnet['gaussian_noise'] = 0.886428
    ce_alexnet['shot_noise'] = 0.894468
    ce_alexnet['impulse_noise'] = 0.922640
    ce_alexnet['defocus_blur'] = 0.819880
    ce_alexnet['glass_blur'] = 0.826268
    ce_alexnet['motion_blur'] = 0.785948
    ce_alexnet['zoom_blur'] = 0.798360
    ce_alexnet['snow'] = 0.866816
    ce_alexnet['frost'] = 0.826572
    ce_alexnet['fog'] = 0.819324
    ce_alexnet['brightness'] = 0.564592
    ce_alexnet['contrast'] = 0.853204
    ce_alexnet['elastic_transform'] = 0.646056
    ce_alexnet['pixelate'] = 0.717840
    ce_alexnet['jpeg_compression'] = 0.606500

    return ce_alexnet


@METRICS.register_module()
class mCE(Accuracy):
    """Mean Corruption Error (mCE) metric.

    The mCE metric is proposed in `Benchmarking Neural Network Robustness to
    Common Corruptions and Perturbations
    <https://openreview.net/pdf?id=HJz6tiCqYm>`_.

    Args:
        topk (int | Sequence[int]): If the ground truth label matches one of
            the best **k** predictions, the sample will be regard as a positive
            prediction. If the parameter is a tuple, all of top-k accuracy will
            be calculated and outputted together. Defaults to 1.
        thrs (Sequence[float | None] | float | None): If a float, predictions
            with score lower than the threshold will be regard as the negative
            prediction. If None, not apply threshold. If the parameter is a
            tuple, accuracy based on all thresholds will be calculated and
            outputted together. Defaults to 0.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
        ano_file_path (str, optional): The path of the annotation file. This
            file will be used in evaluating the fine-tuned model on OOD
            dataset, e.g. ImageNet-A. Defaults to None.
    """

    def __init__(
        self,
        topk: Union[int, Sequence[int]] = (1, ),
        thrs: Union[float, Sequence[Union[float, None]], None] = 0.,
        collect_device: str = 'cpu',
        prefix: Optional[str] = None,
        ann_file_path: Optional[str] = None,
    ) -> None:
        super().__init__(
            topk=topk,
            thrs=thrs,
            collect_device=collect_device,
            prefix=prefix,
            ann_file_path=ann_file_path)
        self.ce_alexnet = get_ce_alexnet()

    def compute_metrics(self, results: List) -> dict:
        """Compute the metrics from processed results.

        Args:
            results (dict): The processed results of each batch.

        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        # NOTICE: don't access `self.results` from the method.
        metrics = {}

        # extract
        category = [res['img_path'].split('/')[3] for res in results]
        target = [res['gt_label'] for res in results]
        pred = [res['pred_score'] for res in results]

        # categorize
        pred_each_category = {}
        target_each_category = {}
        for c, t, p in zip(category, target, pred):
            if c not in pred_each_category.keys():
                pred_each_category[c] = []
                target_each_category[c] = []
            pred_each_category[c].append(p)
            target_each_category[c].append(t)

        # concat
        pred_each_category = {
            key: torch.stack(pred_each_category[key])
            for key in pred_each_category.keys()
        }
        target_each_category = {
            key: torch.cat(target_each_category[key])
            for key in target_each_category.keys()
        }

        # compute mCE
        mce_for_each_category = []
        for key in pred_each_category.keys():
            if key not in self.ce_alexnet.keys():
                continue
            target_current_category = target_each_category[key]
            pred_current_category = pred_each_category[key]
            try:
                acc = self.calculate(pred_current_category,
                                     target_current_category, self.topk,
                                     self.thrs)
                error = (100 - acc[0][0].item()) / (100. *
                                                    self.ce_alexnet[key])
            except ValueError as e:
                # If the topk is invalid.
                raise ValueError(
                    str(e) + ' Please check the `val_evaluator` and '
                    '`test_evaluator` fields in your config file.')
            mce_for_each_category.append(error)

        metrics['mCE'] = sum(mce_for_each_category) / len(
            mce_for_each_category)

        return metrics
