_base_ = '../_base_/default_runtime.py'

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='SwinTransformer',
        arch='base',
        img_size=192,
        out_indices=-1,
        drop_path_rate=0.1,
        stage_cfgs=dict(block_cfgs=dict(window_size=6))),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1024,
        init_cfg=None,  # suppress the default init_cfg of LinearClsHead.
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        cal_acc=False))

dataset_type = 'ImageNet'
data_root = 'data/imagenet/'
extract_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=256, edge='short'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackClsInputs'),
]
extract_dataloader = dict(
    batch_size=8,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root='data/imagenet',
        ann_file='meta/val.txt',
        data_prefix='val',
        pipeline=extract_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
