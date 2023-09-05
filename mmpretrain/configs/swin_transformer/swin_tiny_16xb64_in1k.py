# Copyright (c) OpenMMLab. All rights reserved.
# This is a BETA new format config file, and the usage may change recently.
from mmengine.config import read_base
from mmengine.model import ConstantInit, TruncNormalInit

from mmpretrain.models import CutMix, LabelSmoothLoss, Mixup

with read_base():
    from .._base_.datasets.imagenet_bs64_swin_224 import *
    from .._base_.default_runtime import *
    from .._base_.models.swin_transformer_base import *
    from .._base_.schedules.imagenet_bs1024_adamw_swin import *

# model settings
model.update(
    backbone=dict(
        arch='tiny', img_size=224, drop_path_rate=0.2, stage_cfgs=None),
    head=dict(
        in_channels=768,
        init_cfg=None,  # suppress the default init_cfg of LinearClsHead.
        loss=dict(
            type=LabelSmoothLoss,
            label_smooth_val=0.1,
            mode='original',
            loss_weight=0),
        topk=None,
        cal_acc=False),
    init_cfg=[
        dict(type=TruncNormalInit, layer='Linear', std=0.02, bias=0.),
        dict(type=ConstantInit, layer='LayerNorm', val=1., bias=0.)
    ],
    train_cfg=dict(
        augments=[dict(type=Mixup, alpha=0.8),
                  dict(type=CutMix, alpha=1.0)]))

# schedule settings
optim_wrapper = dict(clip_grad=dict(max_norm=5.0))
