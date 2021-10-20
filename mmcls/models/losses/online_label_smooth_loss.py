r"""
This is an implementation of online label smooth (OLS)
loss from: `Delving Deep into Label Smoothing`
"""

import torch
import torch.nn.functional as F
from mmcv.runner import HOOKS, Hook
from torch import nn
from torch.nn.parallel.distributed import DistributedDataParallel

from ..builder import LOSSES


@LOSSES.register_module()
class OnlineLabelSmoothLoss(nn.Module):
    r"""
    The online label smooth loss following standard PyTorch loss format.

    Args:
        origin_loss (nn.Moudle): the original loss function.
        num_classes (int): the number of classes for classfication task.
        lambda_ols (float): the weight factor for soft label loss.
    """

    def __init__(self, origin_loss, num_classes, lambda_ols=0.5):
        super(OnlineLabelSmoothLoss, self).__init__()

        assert isinstance(origin_loss, nn.Module), origin_loss
        self.origin_loss = origin_loss
        self.num_classes = num_classes
        self.lambda_ols = lambda_ols

        self.soft_labels = torch.zeros(
            num_classes, num_classes, dtype=torch.float32).cuda()
        self.soft_labels[:, :] = 1. / num_classes
        self.soft_labels.requires_grad = False
        self.reset_label_accumulator()

    def reset_label_accumulator(self):
        self.soft_labels_update = torch.zeros(
            self.num_classes, self.num_classes, dtype=torch.float32).cuda()
        self.correct_labels_cnt = torch.zeros(
            self.num_classes, dtype=torch.float32).cuda()
        self.soft_labels_update.requires_grad = False
        self.correct_labels_cnt.requires_grad = False

    def soft_label_regularization(self):
        for class_idx in range(self.num_classes):
            if self.correct_labels_cnt[class_idx].max() < 0.5:
                self.soft_labels[class_idx] = 1. / self.num_classes
            else:
                self.soft_labels[class_idx] = self.soft_labels_update[
                    class_idx] / self.correct_labels_cnt[class_idx]
        self.reset_label_accumulator()

    def update_soft_label(self, input, target):
        with torch.no_grad():
            logits = torch.softmax(input, dim=1)
            sort_args = torch.argsort(logits, dim=1, descending=True)
            for k in range(input.shape[0]):
                if target[k] != sort_args[k, 0]:
                    continue
                self.soft_labels_update[target[k]] += logits[k]
                self.correct_labels_cnt[target[k]] += 1

    def soft_cross_entropy(self, input, target):
        target_prob = torch.zeros_like(input)
        batch = input.shape[0]
        for k in range(batch):
            target_prob[k] = self.soft_labels[target[k]]
        log_like = -F.log_softmax(input, dim=1)
        loss = torch.sum(torch.mul(log_like, target_prob)) / batch
        return loss

    def forward(self, input, target, **kwargs):
        self.update_soft_label(input, target)
        sce_loss = self.soft_cross_entropy(input, target)
        ori_loss = self.origin_loss(input, target, **kwargs)
        return self.lambda_ols * sce_loss + (1 - self.lambda_ols) * ori_loss


@HOOKS.register_module()
class OLSHook(Hook):
    r""" OLS is implemented in the form of custom hook.
         To activate OLS, please add the following cfg:
         ``` python
        custom_hooks = dict(
            type='OLSHook',
            priority='NORMAL',
            lambda_ols=0.5,
        )
        ```
    """

    def __init__(self, lambda_ols=0.5):
        self.lambda_ols = lambda_ols
        self.ols_loss = None

    def before_run(self, runner):
        # Record model and extract num_classes and loss from it
        if isinstance(runner.model, DistributedDataParallel):
            self.model = runner.model.module
        else:
            self.model = runner.model

        # Establish the OnlineLabelSmooth loss
        self.ols_loss = OnlineLabelSmoothLoss(
            origin_loss=self.model.head.compute_loss,
            num_classes=self.model.head.num_classes,
            lambda_ols=self.lambda_ols)

    def before_train_epoch(self, runner):
        self.model.head.compute_loss = self.ols_loss

    def after_train_epoch(self, runner):
        self.model.head.compute_loss = self.ols_loss.origin_loss
        self.ols_loss.soft_label_regularization()
