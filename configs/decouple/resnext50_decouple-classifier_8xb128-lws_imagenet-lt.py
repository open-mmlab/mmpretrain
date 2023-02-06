_base_ = [
    '../_base_/models/resnext50_32x4d.py',
    '../_base_/datasets/imagenet-lt_bs512_decouple.py',
    '../_base_/schedules/imagenet-lt_bs512_coslr_90e.py',
    '../_base_/default_runtime.py'
]

train_dataloader = dict(
    dataset=dict(
        _delete_=True,
        type='ClassBalancedDataset',
        oversample_thr=0.01,
        dataset={{_base_.train_dataset}}), )

train_epochs = 10

# learning policy
param_scheduler = dict(T_max=train_epochs)

# train, val, test setting
train_cfg = dict(max_epochs=train_epochs)

model = dict(
    pretrained=('work_dirs/resnext50_decouple-representation'
                '_8xb128-instance-balanced_imagenet-lt/epoch_90.pth'),
    backbone=dict(frozen_stages=4),
    head=dict(
        type='LWSHead',
        num_classes=1000,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=10))