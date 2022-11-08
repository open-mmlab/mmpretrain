# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import torch
from mmengine.model import ModuleDict

from mmcls.registry import MODELS
from mmcls.structures import MultiTaskDataSample
from .base_head import BaseHead


@MODELS.register_module()
class MultiTaskHead(BaseHead):
    """Multi task head.

    Args:
        task_heads (dict): Sub heads to use, the key will be use to rename the
            loss components.
        common_cfg (dict): The common settings for all heads. Defaults to an
            empty dict.
        init_cfg (dict, optional): The extra initialization settings.
            Defaults to None.
    """

    def __init__(self, task_heads, common_cfg=dict(), init_cfg=None):
        super(MultiTaskHead, self).__init__(init_cfg=init_cfg)

        assert isinstance(task_heads, dict), 'The `task_heads` argument' \
            "should be a dict, which's keys are task names and values are" \
            'configs of head for the task.'

        self.task_heads = ModuleDict()

        for task_name, head_cfg in task_heads.items():
            sub_head = MODELS.build(head_cfg, default_args=common_cfg)
            self.task_heads[task_name] = sub_head

    def forward(self, feats):
        task_results = ()
        for task_name, head in self.task_heads.items():
            head_res = head.forward(feats)
            task_results = task_results + (head_res, )
        return task_results

    def loss(self, feats: Tuple[torch.Tensor],
             data_samples: List[MultiTaskDataSample], **kwargs) -> dict:
        """Calculate losses from the classification score.

        Args:
            feats (tuple[Tensor]): The features extracted from the backbone.
                Multiple stage inputs are acceptable but only the last stage
                will be used to classify. The shape of every item should be
                ``(num_samples, num_classes)``.
            data_samples (List[MultiTaskDataSample]): The annotation data of
                every samples.
            **kwargs: Other keyword arguments to forward the loss module.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        losses = dict()
        for task_name, head in self.task_heads.items():
            """if 'mask'  in gt_label[task_name].keys()  :

            mask = gt_label[task_name]['mask']
              label = gt_label[task_name]['label']
            else: # a tensor
              label = gt_label[task_name]
              batch_n = label.shape[0]
              mask = to_tensor([True]*batch_n)
            """
            mask = []
            masked_data_samples = []
            for data_sample in data_samples:
                sample_mask = data_sample.get_task_mask(task_name)
                if sample_mask:
                    sample = data_sample.get_task_sample(task_name)
                    masked_data_samples.append(sample)
                mask.append(sample_mask)
            masked_features = tuple()
            if type(feats) is tuple:
                for feature in feats:
                    masked_features = masked_features + (feature[mask], )
            else:
                masked_features = feats[mask]
            head_loss = head.loss(masked_features, masked_data_samples,
                                  **kwargs)

            for k, v in head_loss.items():
                losses[f'{task_name}_{k}'] = v
        return losses

    def predict(
        self,
        feats: Tuple[torch.Tensor],
        data_samples: List[MultiTaskDataSample] = None
    ) -> List[MultiTaskDataSample]:
        """Inference without augmentation.

        Args:
            feats (tuple[Tensor]): The features extracted from the backbone.
                Multiple stage inputs are acceptable but only the last stage
                will be used to classify. The shape of every item should be
                ``(num_samples, num_classes)``.
            data_samples (List[MultiTaskDataSample], optional): The annotation
                data of every samples. If not None, set ``pred_label`` of
                the input data samples. Defaults to None.

        Returns:
            List[MultiTaskDataSample]: A list of data samples which contains
            the predicted results.
        """
        # The part can be traced by torch.fx
        task_results = self(feats)

        # The part can not be traced by torch.fx
        predictions = self._get_predictions(task_results, data_samples)
        return predictions

    def _get_predictions(self, task_results, data_samples):
        """Post-process the output of MultiTaskHead."""
        data_results = list(zip(*task_results))
        if data_samples is None:
            data_samples = []
            for data_result in data_results:
                data_samples.append(
                    MultiTaskDataSample().set_pred_label(data_result))
        else:
            for data_sample, data_result in zip(data_samples, data_results):
                data_sample.set_pred_label(data_result)

        return data_samples
