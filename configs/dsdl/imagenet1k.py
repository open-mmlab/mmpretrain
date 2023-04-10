_base_ = [
    '../_base_/models/resnet18.py', '../_base_/schedules/imagenet_bs256.py',
    '../_base_/default_runtime.py'
]

# dataset settings
dataset_type = 'DSDLClsDataset'
data_root = 'data/ImageNet-1K'
img_prefix = 'original'
train_ann = 'dsdl/set-train/train.yaml'
val_ann = 'dsdl/set-val/val.yaml'

data_preprocessor = dict(
    num_classes=1000,
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=224),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackClsInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=256, edge='short'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackClsInputs'),
]

train_dataloader = dict(
    batch_size=32,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=train_ann,
        data_prefix=dict(img_path=img_prefix),
        test_mode=False,
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=32,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=val_ann,
        data_prefix=dict(img_path=img_prefix),
        test_mode=True,
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = dict(type='Accuracy', topk=(1, 5))

test_dataloader = val_dataloader
test_evaluator = val_evaluator
