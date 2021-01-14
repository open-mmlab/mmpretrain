import torch.nn as nn

from ...core.evaluation.mean_ap import mAP


class MAP(nn.Module):
    """Module to calculate the mAP
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        return mAP(pred, target)
