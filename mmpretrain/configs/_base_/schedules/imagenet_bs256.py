# Copyright (c) OpenMMLab. All rights reserved.
# This is a BETA new format config file, and the usage may change recently.
from mmengine.optim import MultiStepLR
from torch.optim import SGD

# optimizer
optim_wrapper = dict(
    optimizer=dict(type=SGD, lr=0.1, momentum=0.9, weight_decay=0.0001))

# learning policy
param_scheduler = dict(
    type=MultiStepLR, by_epoch=True, milestones=[30, 60, 90], gamma=0.1)

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=256)
