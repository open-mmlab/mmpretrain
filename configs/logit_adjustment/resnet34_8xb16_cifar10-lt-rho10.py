_base_ = [
    '../_base_/models/resnet34_cifar.py',
    '../_base_/datasets/cifar10-lt_bs16.py',
    '../_base_/schedules/cifar10_bs128.py',
    '../_base_/default_runtime.py',
]

# model settings
model = dict(head=dict(num_classes=10))

# schedule settings
optim_wrapper = dict(optimizer=dict(weight_decay=0.0005))

param_scheduler = dict(
    type='MultiStepLR',
    by_epoch=True,
    milestones=[60, 120, 160],
    gamma=0.2,
)

train_dataloader = dict(dataset=dict(imbalance_ratio=10))

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3))

custom_hooks = [
    dict(type='PushDataInfoToMessageHubHook', keys=['gt_labels']),
    dict(type='SyncBuffersHook')
]
