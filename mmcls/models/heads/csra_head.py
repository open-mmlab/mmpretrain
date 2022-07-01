# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from ..builder import HEADS
from ..utils import is_tracing
from .cls_head import ClsHead


@HEADS.register_module()
class CSRAClsHead(ClsHead):
    """Class-specific residual attention classifier head.

    Residual Attention: A Simple but Effective Method for Multi-Label
                        Recognition (ICCV 2021)
    Please refer to the `paper <https://arxiv.org/abs/2108.02456>`__ for
    details.

    Args:
        num_classes (int): Number of categories.
        in_channels (int): Number of channels in the input feature map.
        num_heads (int): Number of residual at tensor heads.
        loss (dict): Config of classification loss.
        lam (float): Lambda that combines global average and max pooling
            scores.
        init_cfg (dict | optional): The extra init config of layers.
            Defaults to use dict(type='Normal', layer='Linear', std=0.01).
    """
    temp_settings = {  # softmax temperature settings
        1: [1],
        2: [1, 99],
        4: [1, 2, 4, 99],
        6: [1, 2, 3, 4, 5, 99],
        8: [1, 2, 3, 4, 5, 6, 7, 99]
    }

    def __init__(self,
                 num_classes,
                 in_channels,
                 num_heads,
                 lam,
                 loss=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='mean',
                     loss_weight=1.0),
                 init_cfg=dict(type='Normal', layer='Linear', std=0.01),
                 *args,
                 **kwargs):
        super(CSRAClsHead, self).__init__(
            init_cfg=init_cfg, loss=loss, *args, **kwargs)
        self.temp_list = self.temp_settings[num_heads]
        self.multi_head = nn.ModuleList([
            CSRAModule(in_channels, num_classes, self.temp_list[i], lam)
            for i in range(num_heads)
        ])

    def pre_logits(self, x):
        if isinstance(x, tuple):
            x = x[-1]
        return x

    def post_process(self, pred):
        on_trace = is_tracing()
        if torch.onnx.is_in_onnx_export() or on_trace:
            return pred
        pred = list(pred.detach().cpu().numpy())
        return pred

    def simple_test(self, x, post_process=True, **kwargs):
        logit = 0.
        x = self.pre_logits(x)
        for head in self.multi_head:
            logit += head(x)
        if post_process:
            return self.post_process(logit)
        else:
            return logit

    def forward_train(self, x, gt_label, **kwargs):
        logit = 0.
        x = self.pre_logits(x)
        for head in self.multi_head:
            logit += head(x)
        gt_label = gt_label.type_as(logit)
        losses = self.loss(logit, gt_label, **kwargs)
        return losses


class CSRAModule(nn.Module):  # one basic block

    def __init__(self, input_dim, num_classes, T, lam):
        super(CSRAModule, self).__init__()
        self.T = T  # temperature
        self.lam = lam  # Lambda
        self.head = nn.Conv2d(input_dim, num_classes, 1, bias=False)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        score = self.head(x) / torch.norm(
            self.head.weight, dim=1, keepdim=True).transpose(0, 1)
        score = score.flatten(2)
        base_logit = torch.mean(score, dim=2)

        if self.T == 99:  # max-pooling
            att_logit = torch.max(score, dim=2)[0]
        else:
            score_soft = self.softmax(score * self.T)
            att_logit = torch.sum(score * score_soft, dim=2)

        return base_logit + self.lam * att_logit
