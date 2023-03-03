_base_ = ['../_base_/datasets/voc_bs16.py', '../_base_/default_runtime.py']

# Pre-trained Checkpoint Path
checkpoint = 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet101_8xb32_in1k_20210831-539c63f8.pth'  # noqa
# If you want to use the pre-trained weight of ResNet101-CutMix from the
# originary repo(https://github.com/Kevinz-code/CSRA). Script of
# 'tools/model_converters/torchvision_to_mmpretrain.py' can help you convert
# weight into mmpretrain format. The mAP result would hit 95.5 by using the
# weight. checkpoint = 'PATH/TO/PRE-TRAINED_WEIGHT'

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint, prefix='backbone')),
    neck=None,
    head=dict(
        type='CSRAClsHead',
        num_classes=20,
        in_channels=2048,
        num_heads=1,
        lam=0.1,
        loss=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)))

# dataset setting
data_preprocessor = dict(
    # RGB format normalization parameters
    mean=[0, 0, 0],
    std=[255, 255, 255])

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=448, crop_ratio_range=(0.7, 1.0)),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=448),
    dict(
        type='PackInputs',
        # `gt_label_difficult` is needed for VOC evaluation
        meta_keys=('sample_idx', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction',
                   'gt_label_difficult')),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

# optimizer
# the lr of classifier.head is 10 * base_lr, which help convergence.
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.0002, momentum=0.9, weight_decay=0.0001),
    paramwise_cfg=dict(custom_keys={'head': dict(lr_mult=10)}))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-7,
        by_epoch=True,
        begin=0,
        end=1,
        convert_to_iter_based=True),
    dict(type='StepLR', by_epoch=True, step_size=6, gamma=0.1)
]

train_cfg = dict(by_epoch=True, max_epochs=20, val_interval=1)
val_cfg = dict()
test_cfg = dict()
