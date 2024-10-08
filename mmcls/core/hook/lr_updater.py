# Copyright (c) OpenMMLab. All rights reserved.
from math import cos, pi

from mmcv.runner.hooks import HOOKS, LrUpdaterHook


@HOOKS.register_module()
class CosineAnnealingCooldownLrUpdaterHook(LrUpdaterHook):
    """Cosine annealing learning rate scheduler with cooldown.

    Args:
        min_lr (float, optional): The minimum learning rate after annealing.
            Defaults to None.
        min_lr_ratio (float, optional): The minimum learning ratio after
            nnealing. Defaults to None.
        cool_down_ratio (float): The cooldown ratio. Defaults to 0.1.
        cool_down_time (int): The cooldown time. Defaults to 10.
        by_epoch (bool): If True, the learning rate changes epoch by epoch. If
            False, the learning rate changes iter by iter. Defaults to True.
        warmup (string, optional): Type of warmup used. It can be None (use no
            warmup), 'constant', 'linear' or 'exp'. Defaults to None.
        warmup_iters (int): The number of iterations or epochs that warmup
            lasts. Defaults to 0.
        warmup_ratio (float): LR used at the beginning of warmup equals to
            ``warmup_ratio * initial_lr``. Defaults to 0.1.
        warmup_by_epoch (bool): If True, the ``warmup_iters``
            means the number of epochs that warmup lasts, otherwise means the
            number of iteration that warmup lasts. Defaults to False.

    Note:
        You need to set one and only one of ``min_lr`` and ``min_lr_ratio``.
    """

    def __init__(self,
                 min_lr=None,
                 min_lr_ratio=None,
                 cool_down_ratio=0.1,
                 cool_down_time=10,
                 **kwargs):
        assert (min_lr is None) ^ (min_lr_ratio is None)
        self.min_lr = min_lr
        self.min_lr_ratio = min_lr_ratio
        self.cool_down_time = cool_down_time
        self.cool_down_ratio = cool_down_ratio
        super(CosineAnnealingCooldownLrUpdaterHook, self).__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        if self.by_epoch:
            progress = runner.epoch
            max_progress = runner.max_epochs
        else:
            progress = runner.iter
            max_progress = runner.max_iters

        if self.min_lr_ratio is not None:
            target_lr = base_lr * self.min_lr_ratio
        else:
            target_lr = self.min_lr

        if progress > max_progress - self.cool_down_time:
            return target_lr * self.cool_down_ratio
        else:
            max_progress = max_progress - self.cool_down_time

        return annealing_cos(base_lr, target_lr, progress / max_progress)


def annealing_cos(start, end, factor, weight=1):
    """Calculate annealing cos learning rate.

    Cosine anneal from `weight * start + (1 - weight) * end` to `end` as
    percentage goes from 0.0 to 1.0.

    Args:
        start (float): The starting learning rate of the cosine annealing.
        end (float): The ending learing rate of the cosine annealing.
        factor (float): The coefficient of `pi` when calculating the current
            percentage. Range from 0.0 to 1.0.
        weight (float, optional): The combination factor of `start` and `end`
            when calculating the actual starting learning rate. Default to 1.
    """
    cos_out = cos(pi * factor) + 1
    return end + 0.5 * weight * (start - end) * cos_out
