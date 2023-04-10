_base_ = [
    '../_base_/models/resnet50_cifar.py',
    '../_base_/schedules/cifar10_bs128.py', '../_base_/default_runtime.py'
]

model = dict(head=dict(num_classes=100))

# dataset settings
dataset_type = 'DSDLClsDataset'
data_root = 'data/cifar100'
img_prefix = 'prepared'
train_ann = 'dsdl/set-train/train.yaml'
val_ann = 'dsdl/set-test/test.yaml'

data_preprocessor = dict(
    num_classes=100,
    # RGB format normalization parameters
    mean=[129.304, 124.070, 112.434],
    std=[68.170, 65.392, 70.418],
    to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomCrop', crop_size=32, padding=4),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackClsInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='PackClsInputs'),
]

train_dataloader = dict(
    batch_size=16,
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
    batch_size=16,
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
val_evaluator = dict(type='Accuracy', topk=(1, ))

test_dataloader = val_dataloader
test_evaluator = val_evaluator
