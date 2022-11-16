# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import torch
from mmengine.model import ModuleDict

from mmcls.registry import MODELS
from mmcls.structures import MultiTaskDataSample
from .base_head import BaseHead


def loss_convertor(func, task_name):
    target_type = func.__annotations__['data_samples'].__args__[0].__name__

    # data_samples = data_samples.to_target_data_sample(target_type, task_name)

    def wrapped(inputs, data_samples, **kwargs):
        mask = []
        task_data_samples = []
        for data_sample in data_samples:
            if isinstance(data_sample, MultiTaskDataSample):
                sample_mask = data_sample.get_task_mask(task_name)
                mask.append(sample_mask)
                if sample_mask:
                    task_data_samples.append(
                        data_sample.to_target_data_sample(
                            target_type, task_name))
        masked_inputs = tuple()
        if type(inputs) is tuple:
            for input in inputs:
                masked_inputs = masked_inputs + (input[mask], )
        else:
            masked_inputs = inputs[mask]
        if len(task_data_samples) == 0:
            return {'loss': torch.tensor(0), 'mask_size': torch.tensor(0)}
        loss_output = func(masked_inputs, task_data_samples, **kwargs)
        loss_output['mask_size'] = sum(mask)
        return loss_output

    return wrapped


def predict_convertor(func, task_name):
    target_type = func.__annotations__['data_samples'].__args__[0].__name__

    def wrapped(inputs, data_samples, **kwargs):
        if data_samples is None:
            return func(inputs, data_samples, **kwargs)
        task_data_samples = [
            data_sample.to_target_data_sample(target_type, task_name)
            for data_sample in data_samples
        ]
        return func(inputs, task_data_samples, **kwargs)

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

    def __init__(self, task_heads, common_cfg=dict(), init_cfg=None):
        super(MultiTaskHead, self).__init__(init_cfg=init_cfg)

        assert isinstance(task_heads, dict), 'The `task_heads` argument' \
            "should be a dict, which's keys are task names and values are" \
            'configs of head for the task.'

        self.task_heads = ModuleDict()

        for task_name, head_cfg in task_heads.items():
            if head_cfg['type'] != 'MultiTaskHead':
                sub_head = MODELS.build(head_cfg, default_args=common_cfg)
            else:
                sub_head = MODELS.build(
                    head_cfg, default_args={'common_cfg': common_cfg})
            sub_head.loss = loss_convertor(sub_head.loss, task_name)
            sub_head.predict = predict_convertor(sub_head.predict, task_name)
            self.task_heads[task_name] = sub_head

    def forward(self, feats):
        """The forward process."""
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
        # The part can be traced by torch.fx
        predictions_ = dict()
        for task_name, head in self.task_heads.items():
            predictions_[task_name] = head.predict(feats, data_samples)
        # The part can not be traced by torch.fx
        predictions = self._get_predictions(predictions_, data_samples)
        return predictions

    def _get_predictions(self, preds_dict, data_samples):
        """Post-process the output of MultiTaskHead."""
        pred_dicts = [
            dict(zip(preds_dict, t)) for t in zip(*preds_dict.values())
        ]
        if data_samples is None:
            data_samples = []
            for pred_dict in pred_dicts:
                data_samples.append(
                    MultiTaskDataSample().set_pred_task(pred_dict))
        else:
            for data_sample, pred_dict in zip(data_samples, pred_dicts):
                data_sample.set_pred_task(pred_dict)

        return data_samples
