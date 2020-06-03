import logging
from abc import ABCMeta, abstractmethod

import torch.nn as nn
from mmcv.runner import load_checkpoint


class BaseBackbone(nn.Module, metaclass=ABCMeta):

    def __init__(self):
        super(BaseBackbone, self).__init__()

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            pass
        else:
            raise TypeError('pretrained must be a str or None')

    @abstractmethod
    def forward(self, x):
        pass

    def train(self, mode=True):
        super(BaseBackbone, self).train(mode)
