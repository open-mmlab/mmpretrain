import os.path as osp

# run with 8 GPUs

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearHead',
        num_classes=1000,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))
# dataset settings
dataset_type = 'ImageNet'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
memcached_root = '/mnt/lustre/share/memcached_client/'
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args=dict(
            backend='memcached',
            server_list_cfg=osp.join(memcached_root, 'server_list.conf'),
            client_cfg=osp.join(memcached_root, 'client.conf'))),
    dict(type='ToPIL'),
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomHorizontalFlip', p=0.5),
    dict(type='ToNumpy'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_labels']),
    dict(type='Collect', keys=['img', 'gt_labels'])
]
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args=dict(
            backend='memcached',
            server_list_cfg=osp.join(memcached_root, 'server_list.conf'),
            client_cfg=osp.join(memcached_root, 'client.conf'))),
    dict(type='ToPIL'),
    dict(type='Resize', size=256),
    dict(type='CenterCrop', size=224),
    dict(type='ToNumpy'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_labels']),
    dict(type='Collect', keys=['img', 'gt_labels'])
]
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_prefix='data/imagenet/train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='data/imagenet/val',
        ann_file='data/imagenet/meta/val.txt',
        pipeline=test_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        data_prefix='data/imagenet/val',
        ann_file='data/imagenet/meta/val.txt',
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[30, 60, 90])
# checkpoint saving
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 100
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/imagenet_resnet50_batch256'
load_from = None
resume_from = None
workflow = [('train', 1)]
