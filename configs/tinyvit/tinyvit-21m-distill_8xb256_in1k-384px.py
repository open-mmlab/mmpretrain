_base_ = [
    '../_base_/datasets/imagenet_bs32_pil_bicubic.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py',
    '../_base_/models/tinyvit/tinyvit-21m.py',
]

# model settings
model = dict(
    backbone=dict(
        img_size=(384, 384),
        window_size=[12, 12, 24, 12],
        drop_path_rate=0.1,
    ))

# data settings
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='Resize',
        scale=(384, 384),
        backend='pillow',
        interpolation='bicubic'),
    dict(type='PackInputs'),
]

val_dataloader = dict(dataset=dict(pipeline=test_pipeline))

test_dataloader = val_dataloader
