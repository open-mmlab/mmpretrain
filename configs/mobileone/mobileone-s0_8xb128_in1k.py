_base_ = [
    '../_base_/datasets/imagenet_bs64_pil_resize.py',
    '../_base_/schedules/imagenet_bs256_coslr.py',
    '../_base_/default_runtime.py'
]

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='MobileOne',
        arch='s0',
        out_indices=(3, ),
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1024,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

# dataset settings
train_dataloader = dict(batch_size=128, num_workers=12)
val_dataloader = dict(batch_size=128, num_workers=12)
test_dataloader = dict(batch_size=128, num_workers=12)
