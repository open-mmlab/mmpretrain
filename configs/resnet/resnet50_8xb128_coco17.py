_base_ = [
    '../_base_/models/resnet50.py',
    '../_base_/datasets/coco_bs16.py',
    '../_base_/default_runtime.py',
]

# model settings
pretrained = 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_3rdparty-mill_in21k_20220331-faac000b.pth'  # noqa

# use different head for multilabel task
model = dict(
    type='ImageClassifier',
    backbone=dict(
        init_cfg=dict(
            type='Pretrained', checkpoint=pretrained, prefix='backbone')),
    head=dict(
        type='MultiLabelLinearClsHead',
        num_classes=80,
        thr=0.5,
        loss=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)))

# schedule settings
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001))

# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=0.25,
        by_epoch=True,
        begin=0,
        end=5,
        # update by iter
        convert_to_iter_based=True,
    ),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR',
        T_max=95,
        by_epoch=True,
        begin=5,
        end=100,
    )
]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
# base_batch_size = (8 GPUs) x (16 samples per GPU)
auto_scale_lr = dict(base_batch_size=128 * 8)
# dataset settings

train_dataloader = dict(batch_size=128)
val_dataloader = dict(batch_size=128)
test_dataloader = dict(batch_size=128)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=20),
    # save last three checkpoints
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=3,
        save_best='auto',
        rule='greater'))
