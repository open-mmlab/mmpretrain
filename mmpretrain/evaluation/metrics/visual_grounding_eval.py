# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch
from mmengine.evaluator import BaseMetric
from torchvision.ops.boxes import box_iou

from mmpretrain.registry import METRICS


@METRICS.register_module()
class VisualGroundingMetric(BaseMetric):
    """Visual Grounding evaluator.

    Calculate the box mIOU and box grounding accuracy for visual grounding
    model.

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
    default_prefix = 'visual-grounding'

    def process(self, data_batch, data_samples):
        """Process one batch of data samples.

        The processed results should be stored in ``self.results``, which will
        be used to computed the metrics when all batches have been processed.

        Args:
            data_batch: A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for preds in data_samples:

            pred_box = preds['pred_bboxes'].squeeze()
            box_gt = torch.Tensor(preds['gt_bboxes']).squeeze()

            result = {
                'box': pred_box.to('cpu').squeeze(),
                'box_target': box_gt.squeeze(),
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
        pred_boxes = torch.stack([each['box'] for each in results])
        gt_boxes = torch.stack([each['box_target'] for each in results])
        iou = box_iou(pred_boxes, gt_boxes)
        accu_num = torch.sum(iou >= 0.5)

        miou = torch.mean(iou)
        acc = accu_num / len(gt_boxes)
        coco_val = {'miou': miou, 'acc': acc}
        return coco_val
