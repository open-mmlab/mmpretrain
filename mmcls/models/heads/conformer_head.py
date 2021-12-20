# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.utils.weight_init import trunc_normal_

from ..builder import HEADS
from .cls_head import ClsHead


@HEADS.register_module()
class ConformerHead(ClsHead):
    """Linear classifier head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        init_cfg (dict | optional): The extra init config of layers.
            Defaults to use ``dict(type='Normal', layer='Linear', std=0.01)``.
    """

    def __init__(
            self,
            num_classes,
            in_channels,  # [conv_dim, trans_dim]
            init_cfg=dict(type='Normal', layer='Linear', std=0.01),
            *args,
            **kwargs):
        super(ConformerHead, self).__init__(init_cfg=None, *args, **kwargs)

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.init_cfg = init_cfg

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self.conv_cls_head = nn.Linear(self.in_channels[0], num_classes)
        self.trans_cls_head = nn.Linear(self.in_channels[1], num_classes)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def init_weights(self):
        super(ConformerHead, self).init_weights()

        if (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            # Suppress default init if use pretrained model.
            return
        else:
            self.apply(self._init_weights)

    def pre_logits(self, x):
        if isinstance(x, tuple):
            x = x[-1]
        return x

    def simple_test(self, x, softmax=True, post_process=True):
        """Inference without augmentation.

        Args:
            x (tuple[tuple[tensor, tensor]]): The input features.
                Multi-stage inputs are acceptable but only the last stage will
                be used to classify. Every item should be a tuple which
                includes convluation features and transformer features. The
                shape of them should be ``(num_samples, in_channels[0])`` and
                ``(num_samples, in_channels[1])``.
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
        x = self.pre_logits(x)
        # There are two outputs in the Conformer model
        assert len(x) == 2

        conv_cls_score = self.conv_cls_head(x[0])
        tran_cls_score = self.trans_cls_head(x[1])

        if softmax:
            cls_score = conv_cls_score + tran_cls_score
            pred = (
                F.softmax(cls_score, dim=1) if cls_score is not None else None)
            if post_process:
                pred = self.post_process(pred)
        else:
            pred = [conv_cls_score, tran_cls_score]
            if post_process:
                pred = list(map(self.post_process, pred))
        return pred

    def forward_train(self, x, gt_label):
        x = self.pre_logits(x)
        assert isinstance(x, list) and len(x) == 2, \
            'There should be two outputs in the Conformer model'

        conv_cls_score = self.conv_cls_head(x[0])
        tran_cls_score = self.trans_cls_head(x[1])

        losses = self.loss([conv_cls_score, tran_cls_score], gt_label)
        return losses

    def loss(self, cls_score, gt_label):
        num_samples = len(cls_score[0])
        losses = dict()
        # compute loss
        loss = sum([
            self.compute_loss(score, gt_label, avg_factor=num_samples) /
            len(cls_score) for score in cls_score
        ])
        if self.cal_acc:
            # compute accuracy
            acc = self.compute_accuracy(cls_score[0] + cls_score[1], gt_label)
            assert len(acc) == len(self.topk)
            losses['accuracy'] = {
                f'top-{k}': a
                for k, a in zip(self.topk, acc)
            }
        losses['loss'] = loss
        return losses
