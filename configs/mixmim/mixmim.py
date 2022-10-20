_base_ = [
    '../_base_/models/mixmim/mixmim_base.py'
    '../_base_/datasets/imagenet_bs64_swin_224.py',
    '../_base_/schedules/imagenet_bs256.py',
    '../_base_/default_runtime.py'
]



# file_client_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         '.data/imagenet/': 'openmmlab:s3://openmmlab/datasets/classification/imagenet/',
#         'data/imagenet/': 'openmmlab:s3://openmmlab/datasets/classification/imagenet/'
#     }))

file_client_args = dict(backend='disk')
data_root = "/data/personal/nus-zwb/ImageNet/"
ann_file = "/home/nus-zwb/research/data/imagenet/meta/val.txt"

dataset_type = 'ImageNet'
preprocess_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)

test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(
        type='ResizeEdge',
        scale=219,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackClsInputs'),
]


val_dataloader = dict(
    batch_size=64,
    num_workers=8,
    pin_memory=True,
    collate_fn=dict(type="default_collate"),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file,
        data_prefix='val',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True,
)
test_dataloader = val_dataloader


