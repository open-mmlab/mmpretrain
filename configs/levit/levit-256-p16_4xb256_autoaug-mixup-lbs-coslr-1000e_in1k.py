_base_ = [
    '../_base_/models/levit-256-p16.py',
    '../_base_/datasets/imagenet_bs256_levit_224.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/imagenet_bs1024_adamw_levit.py'
]

dataset_type = 'ImageNet'
data_preprocessor = dict(
    num_classes=1000,
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

bgr_mean = data_preprocessor['mean'][::-1]
bgr_std = data_preprocessor['std'][::-1]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        scale=224,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='RandAugment',
        policies='timm_increasing',
        num_policies=2,
        total_level=10,
        magnitude_level=9,
        magnitude_std=0.5,
        hparams=dict(
            pad_val=[round(x) for x in bgr_mean], interpolation='bicubic')),
    dict(
        type='RandomErasing',
        erase_prob=0.25,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=1 / 3,
        fill_color=bgr_mean,
        fill_std=bgr_std),
    dict(type='PackClsInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeEdge',
        scale=256,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackClsInputs'),
]

train_dataloader = dict(
    batch_size=256,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=r'E:\imagenet',
        ann_file='meta/val.txt',
        data_prefix='ILSVRC2012_img_val',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=256,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=r'E:\imagenet',
        ann_file='meta/val.txt',
        data_prefix='ILSVRC2012_img_val',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = dict(type='Accuracy', topk=(1, 5))

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator

optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=5e-4 * 256 * 4 / 512.0,
        weight_decay=0.025,
        eps=1e-8,
        betas=(0.9, 0.999)),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        custom_keys={
            '.attention_biases': dict(decay_mult=0.0),
        }),
)

# learning policy
# lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
# start_factor=1e-6/lr
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=1e-6 / (5e-4 * 256 * 4 / 512.0),
        by_epoch=True,
        end=6,
        # update by iter
        # convert_to_iter_based=True
    ),
    # main learning rate scheduler
    dict(type='CosineAnnealingLR', eta_min=1e-5, by_epoch=True, begin=5)
]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=1000, val_interval=5)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
# auto_scale_lr = dict(base_batch_size=1024)
model_wrapper_cfg = dict(
    type='MMDistributedDataParallel', find_unused_parameters=True)
