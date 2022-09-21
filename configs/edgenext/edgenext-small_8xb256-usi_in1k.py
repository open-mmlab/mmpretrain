_base_ = ['./edgenext-small_8xb256_in1k.py']

# dataset setting

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeEdge',
        scale=269,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=256),
    dict(type='PackClsInputs')
]

val_dataloader = dict(dataset=dict(pipeline=test_pipeline))

test_dataloader = val_dataloader
