# Copyright (c) OpenMMLab. All rights reserved.
# This is a BETA new format config file, and the usage may change recently.
from mmengine.config import read_base
from mmengine.hooks import CheckpointHook, LoggerHook
from mmengine.model import PretrainedInit
from torch.optim.adamw import AdamW

from mmpretrain.models import ImageClassifier

with read_base():
    from .._base_.datasets.cub_bs8_384 import *
    from .._base_.default_runtime import *
    from .._base_.models.swin_transformer_base import *
    from .._base_.schedules.cub_bs64 import *

# model settings
checkpoint = 'https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin-large_3rdparty_in21k-384px.pth'  # noqa

model.update(
    backbone=dict(
        arch='large',
        init_cfg=dict(
            type=PretrainedInit, checkpoint=checkpoint, prefix='backbone')),
    head=dict(num_classes=200, in_channels=1536))

# schedule settings
optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,
        type=AdamW,
        lr=5e-6,
        weight_decay=0.0005,
        eps=1e-8,
        betas=(0.9, 0.999)),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        custom_keys={
            '.absolute_pos_embed': dict(decay_mult=0.0),
            '.relative_position_bias_table': dict(decay_mult=0.0)
        }),
    clip_grad=dict(max_norm=5.0),
)

default_hooks = dict(
    # log every 20 intervals
    logger=dict(type=LoggerHook, interval=20),
    # save last three checkpoints
    checkpoint=dict(type=CheckpointHook, interval=1, max_keep_ckpts=3))
