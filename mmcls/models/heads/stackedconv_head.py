# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Sequence

import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, ModuleList

from ..builder import HEADS
from .cls_head import ClsHead


class ConvBlock(BaseModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        dropout_rate=0.0,
        norm_cfg=None,
        act_cfg=None,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)

        self.dropout = None

        self.conv = ConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

        if dropout_rate > 0:
            self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


@HEADS.register_module()
class StackedConvClsHead(ClsHead):
    """Classifier head with several conv layer and a output conv layer.

    Args:
        num_classes (int): Number of categories.
        in_channels (int): Number of channels in the input feature map.
        mid_channels (Sequence): Number of channels in the conv layers.
        dropout_rate (float): Dropout rate after each conv layer,
            except the last layer. Defaults to 0.
        norm_cfg (dict, optional): Config dict of normalization layer after
            each conv layer, except the last layer. Defaults to None.
        act_cfg (dict, optional): Config dict of activation function after each
            hidden layer, except the last layer. Defaults to use "ReLU".
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        mid_channels: Sequence,
        dropout_rate: float = 0.0,
        norm_cfg: Dict = None,
        act_cfg: Dict = dict(type="ReLU"),
        **kwargs,
    ):
        super(StackedConvClsHead, self).__init__(**kwargs)
        assert num_classes > 0, (
            f"`num_classes` of StackedConvClsHead must be a positive "
            f"integer, got {num_classes} instead."
        )
        self.num_classes = num_classes

        self.in_channels = in_channels

        assert isinstance(mid_channels, Sequence), (
            f"`mid_channels` of StackedConvClsHead should be a sequence, "
            f"instead of {type(mid_channels)}"
        )
        self.mid_channels = mid_channels

        self.dropout_rate = dropout_rate
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self._init_layers()

    def _init_layers(self):
        self.layers = ModuleList()
        in_channels = self.in_channels
        for hidden_channels in self.mid_channels:
            self.layers.append(
                ConvBlock(
                    in_channels,
                    hidden_channels,
                    dropout_rate=self.dropout_rate,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                )
            )
            in_channels = hidden_channels

        self.layers.append(
            ConvBlock(
                self.mid_channels[-1],
                self.num_classes,
                dropout_rate=0.0,
                norm_cfg=None,
                act_cfg=None,
            )
        )

    def init_weights(self):
        self.layers.init_weights()

    def pre_logits(self, x):
        if isinstance(x, tuple):
            x = x[-1]
        for layer in self.layers[:-1]:
            x = layer(x)
        return x

    @property
    def conv(self):
        return self.layers[-1]

    def simple_test(self, x, softmax=True, post_process=True):
        """Inference without augmentation.

        Args:
            x (tuple[Tensor]): The input features.
                Multi-stage inputs are acceptable but only the last stage will
                be used to classify. The shape of every item should be
                ``(num_samples, in_channels, X, X)``.
            softmax (bool): Whether to softmax the classification score.
            post_process (bool): Whether to do post processing the
                inference results. It will convert the output to a list.

        Returns:
            Tensor | list: The inference results.

                - If no post processing, the output is a tensor with shape
                  ``(num_samples, num_classes)``.
                - If post processing, the output is a multi-dimentional list of
                  float and the dimensions are ``(num_samples, num_classes, X, X)``.
        """
        x = self.pre_logits(x)
        cls_score = self.conv(x)

        if softmax:
            cls_score = cls_score.view(x.size(0), -1)
            pred = (
                F.softmax(cls_score, dim=1) if cls_score is not None else None
            )
        else:
            pred = cls_score

        if post_process:
            return self.post_process(pred)
        else:
            return pred

    def forward_train(self, x, gt_label, **kwargs):
        x = self.pre_logits(x)
        cls_score = self.conv(x)
        cls_score = cls_score.view(x.size(0), -1)

        losses = self.loss(cls_score, gt_label, **kwargs)
        return losses
