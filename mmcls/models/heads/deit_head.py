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
        self.head_dist = nn.Linear(self.in_channels, self.num_classes)

    def simple_test(self, x):
        """Test without augmentation."""
        x = x[-1]
        assert isinstance(x, list) and len(x) == 3
        _, cls_token, dist_token = x
        cls_score = (self.layers(cls_token) + self.head_dist(dist_token)) / 2
        pred = F.softmax(cls_score, dim=1) if cls_score is not None else None

        return self.post_process(pred)

    def forward_train(self, x, gt_label):
        logger = get_root_logger()
        logger.warning("MMClassification doesn't support to train the "
                       'distilled version DeiT.')
        x = x[-1]
        assert isinstance(x, list) and len(x) == 3
        _, cls_token, dist_token = x
        cls_score = (self.layers(cls_token) + self.head_dist(dist_token)) / 2
        losses = self.loss(cls_score, gt_label)
        return losses
