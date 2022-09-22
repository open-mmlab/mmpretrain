# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F

from mmcls.utils import get_root_logger
from ..builder import HEADS
from .vision_transformer_head import VisionTransformerClsHead


@HEADS.register_module()
class DeiTClsHead(VisionTransformerClsHead):
    """Distilled Vision Transformer classifier head.

    Comparing with the :class:`VisionTransformerClsHead`, this head adds an
    extra linear layer to handle the dist token. The final classification score
    is the average of both linear transformation results of ``cls_token`` and
    ``dist_token``.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        hidden_dim (int): Number of the dimensions for hidden layer.
            Defaults to None, which means no extra hidden layer.
        act_cfg (dict): The activation config. Only available during
            pre-training. Defaults to ``dict(type='Tanh')``.
        init_cfg (dict): The extra initialization configs. Defaults to
            ``dict(type='Constant', layer='Linear', val=0)``.
    """

    def __init__(self, *args, **kwargs):
        super(DeiTClsHead, self).__init__(*args, **kwargs)
        if self.hidden_dim is None:
            head_dist = nn.Linear(self.in_channels, self.num_classes)
        else:
            head_dist = nn.Linear(self.hidden_dim, self.num_classes)
        self.layers.add_module('head_dist', head_dist)

    def pre_logits(self, x):
        if isinstance(x, tuple):
            x = x[-1]
        _, cls_token, dist_token = x

        if self.hidden_dim is None:
            return cls_token, dist_token
        else:
            cls_token = self.layers.act(self.layers.pre_logits(cls_token))
            dist_token = self.layers.act(self.layers.pre_logits(dist_token))
            return cls_token, dist_token

    def simple_test(self, x, softmax=True, post_process=True):
        """Inference without augmentation.

        Args:
            x (tuple[tuple[tensor, tensor, tensor]]): The input features.
                Multi-stage inputs are acceptable but only the last stage will
                be used to classify. Every item should be a tuple which
                includes patch token, cls token and dist token. The cls token
                and dist token will be used to classify and the shape of them
                should be ``(num_samples, in_channels)``.
            softmax (bool): Whether to softmax the classification score.
            post_process (bool): Whether to do post processing the
                inference results. It will convert the output to a list.

        Returns:
            Tensor | list: The inference results.

                - If no post processing, the output is a tensor with shape
                  ``(num_samples, num_classes)``.
                - If post processing, the output is a multi-dimentional list of
                  float and the dimensions are ``(num_samples, num_classes)``.
        """
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
