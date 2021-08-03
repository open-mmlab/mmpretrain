r"""
This is an implementation of online label smooth (OLS) loss from: `Delving Deep into Label Smoothing`
"""

import torch
import torch.nn.functional as F

from mmcv.runner import Hook


class OLSHook(Hook):
    r"""
    OLS is implemented in the form of custom hook. To activate OLS, please add the following cfg:
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

    def update_soft_label(self, output, target):
        with torch.no_grad():
            logits = torch.softmax(output, dim=1)
            sort_args = torch.argsort(logits, dim=1, descending=True)
            for k in range(output.shape[0]):
                if target[k] != sort_args[k, 0]:
                    continue
                self.soft_labels_curr[target[k]] += logits[k]
                self.correct_labels_cnt[target[k]] += 1

    def soft_cross_entropy(self, output, target):
        target_prob = torch.zeros_like(output)
        batch = output.shape[0]
        for k in range(batch):
            target_prob[k] = self.soft_labels[target[k]]
        log_like = -F.log_softmax(output, dim=1)
        loss = torch.sum(torch.mul(log_like, target_prob)) / batch
        return loss

    def compute_ols_loss(self, output, target):
        ols_loss = self.compute_ols_loss(output, target)
        ori_loss = self.compute_loss(output, target)
        return self.lambda_ols * ols_loss + (1 - self.lambda_ols) * ori_loss

    def before_run(self, runner):
        self.num_classes = runner.model.head.num_classes
        self.compute_loss = runner.model.head.compute_loss
        self.soft_labels = torch.zeros(self.num_classes, self.num_classes, dtype=torch.float32).cuda()
        self.soft_labels[:, :] = 1. / self.num_classes
        self.soft_labels.requires_grad = False

    def before_train_epoch(self, runner):
        self.soft_labels_curr = torch.zeros(self.num_classes, self.num_classes, dtype=torch.float32).cuda()
        self.correct_labels_cnt = torch.zeros(self.num_classes, dtype=torch.float32).cuda()

    def before_train_iter(self, runner):
        pass

    def after_train_iter(self, runner):
        pass
