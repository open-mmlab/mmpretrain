# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence

from mmengine.structures import LabelData

from mmcls.registry import METRICS
from .multi_label import AveragePrecision, MultiLabelMetric


@METRICS.register_module()
class VOCMultiLabelMetric(MultiLabelMetric):
    """A collection of metrics for multi-label multi-class classification task
    based on confusion matrix for VOC dataset.

    It includes precision, recall, f1-score and support.

    Args:
        difficult_as_postive (Optional[bool]): Whether to map the difficult
            labels as positive in one-hot ground truth for evaluation. If it
            set to True, map difficult gt labels to positive ones(1), If it
            set to False, map difficult gt labels to negative ones(0).
            Defaults to None, the difficult labels will be set to '-1'.
        **kwarg: Refers to `MultiLabelMetric` for detailed docstrings.
    """  # noqa: E501
    default_prefix: Optional[str] = 'multi-label'

    def __init__(self,
                 difficult_as_positive: Optional[bool] = None,
                 **kwargs) -> None:

        self.difficult_as_positive = difficult_as_positive
        super().__init__(**kwargs)

    def process(self, data_batch, data_samples: Sequence[dict]):
        """Process one batch of data samples.

        The processed results should be stored in ``self.results``, which will
        be used to computed the metrics when all batches have been processed.

        Args:
            data_batch: A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            result = dict()
            pred_label = data_sample['pred_label']
            gt_label = data_sample['gt_label']
            gt_label_difficult = data_sample['gt_label_difficult']

            result['pred_score'] = pred_label['score'].clone()
            num_classes = result['pred_score'].size()[-1]

            if 'score' in gt_label:
                result['gt_score'] = gt_label['score'].clone()
            else:
                result['gt_score'] = LabelData.label_to_onehot(
                    gt_label['label'], num_classes)

                # set difficult label for better eval
                if self.difficult_as_positive is None:
                    result['gt_score'][gt_label_difficult] = -1
                elif self.difficult_as_positive:
                    result['gt_score'][gt_label_difficult] = 1

            # Save the result to `self.results`.
            self.results.append(result)


@METRICS.register_module()
class VOCAveragePrecision(AveragePrecision):
    """Calculate the average precision with respect of classes for VOC dataset.

    Args:
        difficult_as_postive (Optional[bool]): Whether to map the difficult
            labels as positive in one-hot ground truth for evaluation. If it
            set to True, map difficult gt labels to positive ones(1), If it
            set to False, map difficult gt labels to negative ones(0).
            Defaults to None, the difficult labels will be set to '-1'.
        **kwarg: Refers to `MultiLabelMetric` for detailed docstrings.
    """
    default_prefix: Optional[str] = 'multi-label'

    def __init__(self,
                 difficult_as_positive: Optional[bool] = None,
                 **kwargs) -> None:

        self.difficult_as_positive = difficult_as_positive
        super().__init__(**kwargs)

    def process(self, data_batch, data_samples: Sequence[dict]):
        """Process one batch of data samples.

        The processed results should be stored in ``self.results``, which will
        be used to computed the metrics when all batches have been processed.

        Args:
            data_batch: A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """

        for data_sample in data_samples:
            result = dict()
            pred_label = data_sample['pred_label']
            gt_label = data_sample['gt_label']
            gt_label_difficult = data_sample['gt_label_difficult']

            result['pred_score'] = pred_label['score']
            num_classes = result['pred_score'].size()[-1]

            if 'score' in gt_label:
                result['gt_score'] = gt_label['score']
            else:
                result['gt_score'] = LabelData.label_to_onehot(
                    gt_label['label'], num_classes)

                # set difficult label for better eval
                if self.difficult_as_positive is None:
                    result['gt_score'][gt_label_difficult] = -1
                elif self.difficult_as_positive:
                    result['gt_score'][gt_label_difficult] = 1

            # Save the result to `self.results`.
            self.results.append(result)
