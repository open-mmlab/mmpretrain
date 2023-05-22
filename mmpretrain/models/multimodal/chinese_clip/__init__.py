# Copyright (c) OpenMMLab. All rights reserved.
from .bert import BertModelCN
from .chinese_clip import ChineseCLIP, ModifiedResNet

__all__ = ['ChineseCLIP', 'ModifiedResNet', 'BertModelCN']
