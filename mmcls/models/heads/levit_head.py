import torch
import torch.nn as nn

from mmcv.cnn import Linear
from mmengine.model import BaseModule

from mmcls.models.heads import ClsHead
from mmcls.registry import MODELS


class BN_Linear(nn.Sequential):
    def __init__(self,in_feature, out_feature, bias=True, std=0.02):
        super(BN_Linear, self).__init__()
        bn = nn.BatchNorm1d(in_feature)
        linear = Linear(in_feature, out_feature, bias=bias)
        nn.init.trunc_normal_(linear.weight, std)
        if bias:
            nn.init.constant_(linear.bias, 0)
        self.bn = bn
        self.linear = linear
    def fuse(self):
        w = self.bn.weight / (self.bn.running_var + self.bn.eps) ** 0.5
        b = self.bn.bias - self.bn.running_mean * \
            self.bn.weight / (self.bn.running_var + self.bn.eps) ** 0.5
        w = self.linear.weight * w[None, :]
        if self.linear.bias is None:
            b = b @ self.linear.weight.T
        else:
            b = (self.linear.weight @ b[:, None]).view(-1) + self.linear.bias
        m = torch.nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


@MODELS.register_module()
class LeViTClsHead(ClsHead):
    def __init__(self, num_classes=1000, distillation=True, in_channels=None,
                 loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 topk=(1,)):
        super(LeViTClsHead, self).__init__()
        self.topk = topk
        self.loss_module = MODELS.build(loss)
        self.num_classes = num_classes
        self.distillation = distillation
        self.head = BN_Linear(
            in_channels, num_classes) if num_classes > 0 else nn.Identity()
        if distillation:
            self.head_dist = BN_Linear(
                in_channels, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x):
        # x = x[-1]
        # B, C, W, H = x.shape
        # x = x.permute(0, 2, 3, 1).reshape(B, W * H, C)
        x = x.mean(1)  # 2 384
        if self.distillation:
            x = self.head(x), self.head_dist(x)  # 2 16 384 -> 2 1000
            if not self.training:
                x = (x[0] + x[1]) / 2
        else:
            x = self.head(x)
        return x
