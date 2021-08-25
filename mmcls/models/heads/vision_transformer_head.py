# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_activation_layer, constant_init, kaiming_init

from ..builder import HEADS
from .cls_head import ClsHead


@HEADS.register_module()
class VisionTransformerClsHead(ClsHead):
    """Vision Transformer classifier head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        hidden_dim (int): Number of the dimensions for hidden layer. Only
            available during pre-training. Default None.
        act_cfg (dict): The activation config. Only available during
            pre-training. Defalut Tanh.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 hidden_dim=None,
                 act_cfg=dict(type='Tanh'),
                 *args,
                 **kwargs):
        super(VisionTransformerClsHead, self).__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.act_cfg = act_cfg

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self._init_layers()

    def _init_layers(self):
        if self.hidden_dim is None:
            layers = [('head', nn.Linear(self.in_channels, self.num_classes))]
        else:
            layers = [
                ('pre_logits', nn.Linear(self.in_channels, self.hidden_dim)),
                ('act', build_activation_layer(self.act_cfg)),
                ('head', nn.Linear(self.hidden_dim, self.num_classes)),
            ]
        self.layers = nn.Sequential(OrderedDict(layers))

    def init_weights(self):
        super(VisionTransformerClsHead, self).init_weights()
        # Modified from ClassyVision
        if hasattr(self.layers, 'pre_logits'):
            # Lecun norm
            kaiming_init(
                self.layers.pre_logits, mode='fan_in', nonlinearity='linear')
        constant_init(self.layers.head, 0)

    def simple_test(self, img):
        """Test without augmentation."""
        cls_score = self.layers(img)
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        pred = F.softmax(cls_score, dim=1) if cls_score is not None else None

        return self.post_process(pred)

    def forward_train(self, x, gt_label):
        cls_score = self.layers(x)
        losses = self.loss(cls_score, gt_label)
        return losses
