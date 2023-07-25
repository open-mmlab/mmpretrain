# Copyright (c) OpenMMLab. All rights reserved.
# This is a BETA new format config file, and the usage may change recently.
from mmengine.config import read_base

from mmcv.transforms import (LoadImageFromFile, RandomFlip)

from mmpretrain.datasets import (CenterCrop, LoadImageFromFile,
                                 PackInputs, RandomFlip, RandomResizedCrop,
                                 ResizeEdge)

with read_base():
    from .._base_.models.vit_large_p16 import *
    from .._base_.datasets.imagenet_bs64_pil_resize import *
    from .._base_.schedules.imagenet_bs4096_AdamW import *
    from .._base_.default_runtime import *


# model setting
model = dict(backbone=dict(img_size=384))

# dataset setting
data_preprocessor = dict(
    mean=[127.5, 127.5, 127.5],
    std=[127.5, 127.5, 127.5],
    # convert image from BGR to RGB
    to_rgb=True,
)

train_pipeline = [
    dict(type=LoadImageFromFile),
    dict(type=RandomResizedCrop, scale=384, backend='pillow'),
    dict(type=RandomFlip, prob=0.5, direction='horizontal'),
    dict(type=PackInputs),
]

test_pipeline = [
    dict(type=LoadImageFromFile),
    dict(type=ResizeEdge, scale=384, edge='short', backend='pillow'),
    dict(type=CenterCrop, crop_size=384),
    dict(type=PackInputs),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))

# schedule setting
optim_wrapper = dict(clip_grad=dict(max_norm=1.0))