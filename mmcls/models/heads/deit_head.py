# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F

from mmcls.utils import get_root_logger
from ..builder import HEADS
from .vision_transformer_head import VisionTransformerClsHead


@HEADS.register_module()
class DeiTClsHead(VisionTransformerClsHead):

    def __init__(self, *args, **kwargs):
        super(DeiTClsHead, self).__init__(*args, **kwargs)
        if self.hidden_dim is None:
            head_dist = nn.Linear(self.in_channels, self.num_classes)
        else:
            head_dist = nn.Linear(self.hidden_dim, self.num_classes)
        self.layers.add_module('head_dist', head_dist)

    def pre_logits(self, x):
        x = x[-1]
        assert isinstance(x, list) and len(x) == 3
        _, cls_token, dist_token = x

        if self.hidden_dim is None:
            return cls_token, dist_token
        else:
            cls_token = self.layers.act(self.layers.pre_logits(cls_token))
            dist_token = self.layers.act(self.layers.pre_logits(dist_token))
            return cls_token, dist_token

    def simple_test(self, x, softmax=True, post_process=True):
        """Test without augmentation."""
        cls_token, dist_token = self.pre_logits(x)
        cls_score = (self.layers.head(cls_token) +
                     self.layers.head_dist(dist_token)) / 2

        if softmax:
            pred = F.softmax(
                cls_score, dim=1) if cls_score is not None else None
        else:
            pred = cls_score

        if post_process:
            return self.post_process(pred)
        else:
            return pred

    def forward_train(self, x, gt_label):
        logger = get_root_logger()
        logger.warning("MMClassification doesn't support to train the "
                       'distilled version DeiT.')
        cls_token, dist_token = self.pre_logits(x)
        cls_score = (self.layers.head(cls_token) +
                     self.layers.head_dist(dist_token)) / 2
        losses = self.loss(cls_score, gt_label)
        return losses
