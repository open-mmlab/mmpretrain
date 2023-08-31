# Copyright (c) OpenMMLab. All rights reserved.
# This is a BETA new format config file, and the usage may change recently.
from mmengine.model import ConstantInit, TruncNormalInit

from mmpretrain.models import (CutMix, GlobalAveragePooling, ImageClassifier,
                               LabelSmoothLoss, LinearClsHead, Mixup,
                               SwinTransformer)

# model settings
model = dict(
    type=ImageClassifier,
    backbone=dict(
        type=SwinTransformer, arch='small', img_size=224, drop_path_rate=0.3),
    neck=dict(type=GlobalAveragePooling),
    head=dict(
        type=LinearClsHead,
        num_classes=1000,
        in_channels=768,
        init_cfg=None,  # suppress the default init_cfg of LinearClsHead.
        loss=dict(type=LabelSmoothLoss, label_smooth_val=0.1, mode='original'),
        cal_acc=False),
    init_cfg=[
        dict(type=TruncNormalInit, layer='Linear', std=0.02, bias=0.),
        dict(type=ConstantInit, layer='LayerNorm', val=1., bias=0.)
    ],
    train_cfg=dict(
        augments=[dict(type=Mixup, alpha=0.8),
                  dict(type=CutMix, alpha=1.0)]),
)
