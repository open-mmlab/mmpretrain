_base_ = [
    '../_base_/datasets/imagenet_bs32_pil_bicubic.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py',
    '../_base_/models/tinyvit/tinyvit-21m-384.py',
]

# data settings
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='Resize',
        scale=(384, 384),
        backend='pillow',
        interpolation='bicubic'),
    dict(type='PackClsInputs'),
]

val_dataloader = dict(dataset=dict(pipeline=test_pipeline))

test_dataloader = val_dataloader
