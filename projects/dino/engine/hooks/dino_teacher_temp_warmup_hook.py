# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmengine.hooks import Hook

from mmpretrain.registry import HOOKS


@HOOKS.register_module()
class DINOTeacherTempWarmupHook(Hook):
    """Warmup teacher temperature for DINO.

    This hook warmups the temperature for teacher to stabilize the training
    process.

    Args:
        warmup_teacher_temp (float): Warmup temperature for teacher.
        teacher_temp (float): Temperature for teacher.
        teacher_temp_warmup_epochs (int): Warmup epochs for teacher
            temperature.
        max_epochs (int): Maximum epochs for training.
    """

    def __init__(self, warmup_teacher_temp: float, teacher_temp: float,
                 teacher_temp_warmup_epochs: int, max_epochs: int) -> None:
        super().__init__()
        self.teacher_temps = np.concatenate(
            (np.linspace(warmup_teacher_temp, teacher_temp,
                         teacher_temp_warmup_epochs),
             np.ones(max_epochs - teacher_temp_warmup_epochs) * teacher_temp))

    def before_train_epoch(self, runner) -> None:
        runner.model.module.head.teacher_temp = self.teacher_temps[
            runner.epoch]
