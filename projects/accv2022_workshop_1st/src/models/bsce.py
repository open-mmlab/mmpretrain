import mmengine
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from mmcls.registry import MODELS


@MODELS.register_module(force=True)
class BalancedSoftmax(_Loss):
    """Balanced Softmax Loss."""

    def __init__(self, ann_file):
        super(BalancedSoftmax, self).__init__()
        lines = mmengine.list_from_file(ann_file)
        targets = np.array([int(x.strip().rsplit(' ', 1)[-1]) for x in lines])
        freq = torch.from_numpy(np.bincount(targets))
        freq = freq / freq.sum()
        self.register_buffer('sample_per_class', freq)

    def forward(self,
                input,
                label,
                eight=None,
                avg_factor='mean',
                reduction_override=None,
                **kwargs):
        return balanced_softmax_loss(label, input, self.sample_per_class,
                                     avg_factor)


def balanced_softmax_loss(labels, logits, sample_per_class, reduction):
    """Compute the Balanced Softmax Loss between `logits` and the ground truth
    `labels`.

    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      sample_per_class: A int tensor of size [no of classes].
      reduction: string. One of "none", "mean", "sum"
    Returns:
      loss: A float tensor. Balanced Softmax Loss.
    """
    spc = sample_per_class.type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + spc.log()
    loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
    return loss
