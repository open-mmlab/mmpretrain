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
        type='MultiLabelClsHead',
        num_classes=80,
        loss=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)))

# schedule settings
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0),
    # update the final linear by 10 times learning rate.
    paramwise_cfg=dict(custom_keys={'.backbone.classifier': dict(lr_mult=10)}),
)

# learning policy
param_scheduler = dict(type='StepLR', by_epoch=True, step_size=15, gamma=0.1)

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=45, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
# base_batch_size = (8 GPUs) x (16 samples per GPU)
auto_scale_lr = dict(base_batch_size=128)
# dataset settings

train_dataloader = dict(batch_size=64*8)

default_hooks = dict(
    # save last three checkpoints
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=3,
        save_best='auto',
        rule='greater'))