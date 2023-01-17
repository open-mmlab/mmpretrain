# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
from mmengine.model import ModuleDict

from mmcls.registry import MODELS
from mmcls.structures import MultiTaskDataSample
from .base_head import BaseHead


def loss_convertor(loss_func, task_name):

    def wrapped(inputs, data_samples, **kwargs):
        mask = torch.empty(len(data_samples), dtype=torch.bool)
        task_data_samples = []
        for i, data_sample in enumerate(data_samples):
            assert isinstance(data_sample, MultiTaskDataSample)
            sample_mask = task_name in data_sample
            mask[i] = sample_mask
            if sample_mask:
                task_data_samples.append(data_sample.get(task_name))

        if len(task_data_samples) == 0:
            return {'loss': torch.tensor(0.), 'mask_size': torch.tensor(0.)}

        # Mask the inputs of the task
        def mask_inputs(inputs, mask):
            if isinstance(inputs, Sequence):
                return type(inputs)(
                    [mask_inputs(input, mask) for input in inputs])
            elif isinstance(inputs, torch.Tensor):
                return inputs[mask]

        masked_inputs = mask_inputs(inputs, mask)
        loss_output = loss_func(masked_inputs, task_data_samples, **kwargs)
        loss_output['mask_size'] = mask.sum().to(torch.float)
        return loss_output

    return wrapped


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

    def __init__(self, task_heads, init_cfg=None, **kwargs):
        super(MultiTaskHead, self).__init__(init_cfg=init_cfg)

        assert isinstance(task_heads, dict), 'The `task_heads` argument' \
            "should be a dict, which's keys are task names and values are" \
            'configs of head for the task.'

        self.task_heads = ModuleDict()

        for task_name, sub_head in task_heads.items():
            if not isinstance(sub_head, nn.Module):
                sub_head = MODELS.build(sub_head, default_args=kwargs)
            sub_head.loss = loss_convertor(sub_head.loss, task_name)
            self.task_heads[task_name] = sub_head

    def forward(self, feats):
        """The forward process."""
        return {
            task_name: head(feats)
            for task_name, head in self.task_heads.items()
        }

    def loss(self, feats: Tuple[torch.Tensor],
             data_samples: List[MultiTaskDataSample], **kwargs) -> dict:
        """Calculate losses from the classification score.

        Args:
            feats (tuple[Tensor]): The features extracted from the backbone.
            data_samples (List[MultiTaskDataSample]): The annotation data of
                every samples.
            **kwargs: Other keyword arguments to forward the loss module.

        Returns:
            dict[str, Tensor]: a dictionary of loss components, each task loss
                key will be prefixed by the task_name like "task1_loss"
        """
        losses = dict()
        for task_name, head in self.task_heads.items():
            head_loss = head.loss(feats, data_samples, **kwargs)
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
            data_samples (List[MultiTaskDataSample], optional): The annotation
                data of every samples. If not None, set ``pred_label`` of
                the input data samples. Defaults to None.

        Returns:
            List[MultiTaskDataSample]: A list of data samples which contains
            the predicted results.
        """
        predictions_dict = dict()

        for task_name, head in self.task_heads.items():
            task_samples = head.predict(feats)
            batch_size = len(task_samples)
            predictions_dict[task_name] = task_samples

        if data_samples is None:
            data_samples = [MultiTaskDataSample() for _ in range(batch_size)]

        for task_name, task_samples in predictions_dict.items():
            for data_sample, task_sample in zip(data_samples, task_samples):
                task_sample.set_field(
                    task_name in data_sample.tasks,
                    'eval_mask',
                    field_type='metainfo')

                if task_name in data_sample.tasks:
                    data_sample.get(task_name).update(task_sample)
                else:
                    data_sample.set_field(task_sample, task_name)

        return data_samples
