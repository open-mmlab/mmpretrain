# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner import ModuleDict

from ..builder import HEADS
from .base_head import BaseHead


@HEADS.register_module()
class MultiTaskClsHead(BaseHead):
    """Multi task head.

    Args:
        sub_heads (dict): Sub heads to use, the key will be use to rename the
            loss components.
        common_cfg (dict): The common settings for all heads. Defaults to an
            empty dict.
        init_cfg (dict, optional): The extra initialization settings.
            Defaults to None.
    """

    def __init__(self, sub_heads, common_cfg=dict(), init_cfg=None):
        super(MultiTaskClsHead, self).__init__(init_cfg=init_cfg)

        assert isinstance(sub_heads, dict), 'The `sub_heads` argument' \
            "should be a dict, which's keys are task names and values are" \
            'configs of head for the task.'

        self.sub_heads = ModuleDict()

        for task_name, head_cfg in sub_heads.items():
            sub_head = HEADS.build(head_cfg, default_args=common_cfg)
            self.sub_heads[task_name] = sub_head

    def forward_train(self, features, gt_label, **kwargs):
        losses = dict()
        for task_name, head in self.sub_heads.items():
            if 'mask'  in gt_label.keys()  :
              mask = gt_label['mask'][task_name]
              label = gt_label['label'][task_name]
            else: # a tensor
              label = gt_label[task_name]
              batch_n = label.shape[0]
              mask = to_tensor([True]*batch_n)
            masked_features = tuple()
            for feature in features :
                masked_features = masked_features + (feature[mask],)
            head_loss = head.forward_train(masked_features, label[mask], **kwargs)
            for k, v in head_loss.items():
                losses[f'{task_name}_{k}'] = v
        return losses

    def pre_logits(self, x):
        results = dict()
        for task_name, head in self.sub_heads.items():
            results[task_name] = head.pre_logits(x)
        return results

    def simple_test(self,
                    x,
                    post_process=True,
                    task_wise_args=dict(),
                    **kwargs):
        """Inference without augmentation.

        Args:
            x (tuple[Tensor]): The input features will be forwarded to every
                heads.
            post_process (bool): Whether to do post-process for each task. If
                True, returns a list of results and each item is a result dict
                for a sample. If False, returns a dict including output without
                post-process of every tasks. Defaults to True.
            task_wise_args (dict): A dict of keyword arguments for different
                heads.

        Returns:
            dict | list[dict]: The inference results. The output type depends
            on ``post_process``, and more details can be found in the examples.

        Examples:
            >>> import torch
            >>> from mmcls.models import HEADS
            >>>
            >>> feats = torch.rand(3, 128)
            >>> cfg = dict(
            ...     type='MultiTaskClsHead',
            ...     sub_heads={
            ...         'task1': dict(num_classes=5),
            ...         'task2': dict(num_classes=10),
            ...     },
            ...     common_cfg=dict(
            ...         type='LinearClsHead',
            ...         in_channels=128,
            ...         loss=dict(type='CrossEntropyLoss')),
            ... )
            >>> head = HEADS.build(cfg)
            >>> # simple_test with post_process
            >>> head.simple_test(feats, post_process=True)
            [{'task1': array([...], dtype=float32),
              'task2': array([...], dtype=float32)},
             {'task1': array([...], dtype=float32),
              'task2': array([...], dtype=float32)},
             {'task1': array([...], dtype=float32),
              'task2': array([...], dtype=float32)}]
            >>> # simple_test without post_process
            >>> head.simple_test(feats, post_process=False)
            {'task1': tensor(...), grad_fn=<...>),
             'task2': tensor(...), grad_fn=<...>}
        """
        results = dict()
        for task_name, head in self.sub_heads.items():
            forward_args = {
                'post_process': post_process,
                **kwargs,
                **task_wise_args.get(task_name, {})
            }
            results[task_name] = head.simple_test(x, **forward_args)

        if post_process:
            # Convert dict of list to list of dict.
            results = [dict(zip(results, t)) for t in zip(*results.values())]

        return results
