# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
from mmengine.dist import all_reduce, get_world_size
from mmengine.model import BaseModule

from mmpretrain.registry import MODELS


@MODELS.register_module()
class DINOHead(BaseModule):
    """Implementation for DINO head.

    This module is proposed in `DINO: Emerging Properties in Self-Supervised
    Vision Transformers <https://arxiv.org/abs/2104.14294>`_.

    Args:
        out_channels (int): Output channels of the head.
        num_crops (int): Number of crops.
        student_temp (float): Temperature for student output.
        center_momentum (float): Momentum for center update.
    """

    def __init__(self, out_channels: int, num_crops: int, student_temp: float,
                 center_momentum: float) -> None:
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = 0
        self.center_momentum = center_momentum
        self.num_crops = num_crops
        self.register_buffer('center', torch.zeros(1, out_channels))

    def forward(self, student_output: torch.Tensor,
                teacher_output: torch.Tensor) -> torch.Tensor:

        current_teacher_output = teacher_output
        student_output = student_output / self.student_temp
        student_output = student_output.chunk(self.num_crops, dim=0)

        # teacher centering and sharpening
        teacher_output = F.softmax(
            (teacher_output - self.center) / self.teacher_temp, dim=-1)
        teacher_output = teacher_output.detach().chunk(2, dim=0)

        total_loss = 0
        n_loss_terms = 0

        for i in range(len(teacher_output)):
            for j in range(len(student_output)):
                if i == j:
                    continue
                total_loss += (-teacher_output[i] *
                               student_output[j].log_softmax(dim=-1)).sum(
                                   dim=-1).mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(current_teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output: torch.Tensor) -> None:

        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * get_world_size())

        # ema update batch center
        self.center = self.center * self.center_momentum + batch_center * (
            1 - self.center_momentum)
