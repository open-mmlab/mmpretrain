_base_ = [
    '../_base_/models/resnet50_cifar.py',
    '../_base_/datasets/cifar100-lt_bs16.py',
    '../_base_/schedules/cifar10_bs128.py',
    '../_base_/default_runtime.py',
]

# model settings
model = dict(
    head=dict(
        num_classes=100,
        in_channels=2048,
        type='LogitAdjustLinearClsHead',
        enable_loss_adjustment=False,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))

# schedule settings
optim_wrapper = dict(optimizer=dict(weight_decay=0.0005))

param_scheduler = dict(
    type='MultiStepLR',
    by_epoch=True,
    milestones=[60, 120, 160],
    gamma=0.2,
)

custom_hooks = [dict(type='SendDatasetInfoMessageHubHook', keys=['gt_labels'])]
