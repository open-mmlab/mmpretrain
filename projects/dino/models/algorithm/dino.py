# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Union

import torch
from torch import nn

from mmpretrain.models import BaseSelfSupervisor, CosineEMA
from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample


@MODELS.register_module()
class DINO(BaseSelfSupervisor):
    """Implementation for DINO.

    This module is proposed in `DINO: Emerging Properties in Self-Supervised
    Vision Transformers <https://arxiv.org/abs/2104.14294>`_.

    Args:
        backbone (dict): Config for backbone.
        neck (dict): Config for neck.
        head (dict): Config for head.
        pretrained (str, optional): Path for pretrained model.
            Defaults to None.
        base_momentum (float, optional): Base momentum for momentum update.
            Defaults to 0.99.
        data_preprocessor (dict, optional): Config for data preprocessor.
            Defaults to None.
        init_cfg (list[dict] | dict, optional): Config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 backbone: dict,
                 neck: dict,
                 head: dict,
                 pretrained: Optional[str] = None,
                 base_momentum: float = 0.99,
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[Union[List[dict], dict]] = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            head=head,
            pretrained=pretrained,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

        # create momentum model
        self.teacher = CosineEMA(
            nn.Sequential(self.backbone, self.neck), momentum=base_momentum)
        # weight normalization layer
        self.neck.last_layer = nn.utils.weight_norm(self.neck.last_layer)
        self.neck.last_layer.weight_g.data.fill_(1)
        self.neck.last_layer.weight_g.requires_grad = False
        self.teacher.module[1].last_layer = nn.utils.weight_norm(
            self.teacher.module[1].last_layer)
        self.teacher.module[1].last_layer.weight_g.data.fill_(1)
        self.teacher.module[1].last_layer.weight_g.requires_grad = False

    def loss(self, inputs: torch.Tensor,
             data_samples: List[DataSample]) -> dict:
        global_crops = torch.cat(inputs[:2])
        local_crops = torch.cat(inputs[2:])
        # teacher forward
        teacher_output = self.teacher(global_crops)

        # student forward global
        student_output_global = self.backbone(global_crops)
        student_output_global = self.neck(student_output_global)

        # student forward local
        student_output_local = self.backbone(local_crops)
        student_output_local = self.neck(student_output_local)

        student_output = torch.cat(
            (student_output_global, student_output_local))

        # compute loss
        loss = self.head(student_output, teacher_output)

        return dict(loss=loss)
