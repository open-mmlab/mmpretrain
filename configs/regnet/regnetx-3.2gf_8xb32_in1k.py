_base_ = [
    '../_base_/models/regnet/regnetx_3.2gf.py',
    '../_base_/datasets/imagenet_bs32.py',
    '../_base_/schedules/imagenet_bs256.py', '../_base_/default_runtime.py'
]

# precise bn
precise_bn = dict(num_items=8192, interval=1)

# sgd with nesterov
optimizer = dict(
    type='SGD', lr=0.4, momentum=0.9, weight_decay=5e-5, nesterov=True)

# dataset settings
dataset_type = 'ImageNet'

# normalization params, in order of BGR
NORM_MEAN = [103.53, 116.28, 123.675]
NORM_STD = [57.375, 57.12, 58.395]

# lighting params, in order of RGB
EIGVAL = [0.2175, 0.0188, 0.0045]
EIGVEC = [[-0.5675, 0.7192, 0.4009], [-0.5808, -0.0045, -0.814],
          [-0.5836, -0.6948, 0.4203]]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(
        type='Lighting',
        eigval=EIGVAL,
        eigvec=EIGVEC,
        alphastd=25.5,
        to_rgb=True),
    dict(type='Normalize', mean=NORM_MEAN, std=NORM_STD, to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, -1)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', mean=NORM_MEAN, std=NORM_STD, to_rgb=False),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=8,
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
evaluation = dict(interval=1, metric='accuracy')
