import torch
import torch.nn.functional as F

from .linear_head import LinearClsHead
from ..builder import HEADS


@HEADS.register_module()
class AngularPenaltyHead(LinearClsHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 init_cfg=dict(type='Normal', layer='Linear', std=0.01),
                 loss_type='arcface',
                 eps=1e-7,
                 s=None,
                 m=None,
                 *args,
                 **kwargs):
        '''
        Angular Penalty Softmax Loss

        Three 'loss_types' available: ['arcface', 'sphereface', 'cosface']
        These losses are described in the following papers:

        ArcFace: https://arxiv.org/abs/1801.07698
        SphereFace: https://arxiv.org/abs/1704.08063
        CosFace/Ad Margin: https://arxiv.org/abs/1801.05599

        '''
        kwargs['fc_bias'] = False
        super(AngularPenaltyHead, self).__init__(
            num_classes,
            in_channels,
            init_cfg=init_cfg,
            *args,
            **kwargs
        )
        loss_type = loss_type.lower()
        assert loss_type in ['arcface', 'sphereface', 'cosface']
        if loss_type == 'arcface':
            self.s = 64.0 if not s else s
            self.m = 0.5 if not m else m
        if loss_type == 'sphereface':
            self.s = 64.0 if not s else s
            self.m = 1.35 if not m else m
        if loss_type == 'cosface':
            self.s = 30.0 if not s else s
            self.m = 0.4 if not m else m
        self.loss_type = loss_type
        self.eps = eps

    def _additive_angular_margin_penalty(self, cls_score, gt_label):

        if self.loss_type == 'cosface':
            numerator = self.s * (
                    torch.diagonal(
                        cls_score.transpose(0, 1)[gt_label]) - self.m)
        if self.loss_type == 'arcface':
            numerator = self.s * torch.cos(
                torch.acos(
                    torch.clamp(
                        torch.diagonal(cls_score.transpose(0, 1)[gt_label]),
                        - 1. + self.eps, 1 - self.eps
                    )
                ) + self.m)
        if self.loss_type == 'sphereface':
            numerator = self.s * torch.cos(
                self.m * torch.acos(
                    torch.clamp(
                        torch.diagonal(cls_score.transpose(0, 1)[gt_label]),
                        - 1. + self.eps, 1 - self.eps
                    )
                )
            )
        return numerator

    def loss(self, cls_score, gt_label):
        logits = self._additive_angular_margin_penalty(cls_score, gt_label)

        excl = torch.cat(
            [
                torch.cat(
                    (cls_score[i, :y], cls_score[i, y + 1:])).unsqueeze(0)
                for i, y in enumerate(gt_label)
            ],
            dim=0
        )
        denominator = torch.exp(logits) + torch.sum(
            torch.exp(self.s * excl), dim=1)
        L = logits - torch.log(denominator)

        losses = dict()
        if self.cal_acc:
            # compute accuracy
            acc = self.compute_accuracy(cls_score, gt_label)
            assert len(acc) == len(self.topk)
            losses['accuracy'] = {
                f'top-{k}': a
                for k, a in zip(self.topk, acc)
            }
        losses['loss'] = -torch.mean(L)
        return losses

    def forward_train(self, x, gt_label):
        '''
        input shape (N, in_features)
        '''

        for W in self.fc.parameters():
            W = F.normalize(W, p=2, dim=1)

        if isinstance(x, tuple):
            x = x[-1]
        x = F.normalize(x, p=2, dim=1)

        return super(AngularPenaltyHead, self).forward_train(x, gt_label)
