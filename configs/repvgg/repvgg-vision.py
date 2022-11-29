_base_ = [
    '../_base_/models/repvgg-A0_in1k.py',
    '../_base_/schedules/imagenet_bs256_coslr.py',
    '../_base_/default_runtime.py'
]

# schedule settings
optim_wrapper = dict(
    paramwise_cfg=dict(
        bias_decay_mult=0.0,
        custom_keys={
            'branch_3x3.norm': dict(decay_mult=0.0),
            'branch_1x1.norm': dict(decay_mult=0.0),
            'branch_norm.bias': dict(decay_mult=0.0),
        }))

# schedule settings
param_scheduler = dict(
    type='CosineAnnealingLR',
    T_max=120,
    by_epoch=True,
    begin=0,
    end=120,
    convert_to_iter_based=True)

train_cfg = dict(by_epoch=True, max_epochs=120)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3))

# dataset settings
dataset_type = 'VisionImageNet'
data_preprocessor = dict(
    num_classes=1000,
    mean=[0., 0., 0.],
    std=[1., 1., 1.],
    to_rgb=False,
)

train_dataloader = dict(
    batch_size=32,
    num_workers=12,
    dataset=dict(
        type=dataset_type,
        ann_file='./data/imagenet/meta/train.txt',
        data_prefix='./data/imagenet/train'),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=64,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        data_prefix='./data/imagenet/val',
        ann_file='./data/imagenet/meta/val.txt'),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = dict(type='Accuracy', topk=(1, 5))

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator
