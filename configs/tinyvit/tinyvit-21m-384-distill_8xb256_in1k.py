_base_ = [
    './tinyvit-21m-224_8xb256_in1k.py',
]

# Model settings
model = dict(
    backbone=dict(arch='tinyvit_21m_384', drop_path_rate=0.1),
    head=dict(in_channels=576))

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
