_base_ = [
    '../_base_/models/resnext50_32x4d.py',
    '../_base_/datasets/imagenet-lt_bs512_decouple.py',
    '../_base_/schedules/imagenet-lt_bs512_coslr_90e.py',
    '../_base_/default_runtime.py'
]

model = dict(
    head=dict(
        type='TauNormHead',
        num_classes=1000,
        in_channels=2048,
        tau=0.7,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))
