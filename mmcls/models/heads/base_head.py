from abc import ABCMeta, abstractmethod

import torch.nn as nn


class BaseHead(nn.Module, metaclass=ABCMeta):
    """Base head.

    """

    def __init__(self):
        super(BaseHead, self).__init__()

    def init_weights(self):
        pass

    @abstractmethod
    def forward_train(self, x, gt_label, **kwargss):
        pass
