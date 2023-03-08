# Pretrain with Custom Dataset

- [Pretrain with Custom Dataset](#pretrain-with-custom-dataset)
  - [Train MAE on Custom Dataset](#train-mae-on-custom-dataset)
    - [Step-1: Get the path of custom dataset](#step-1-get-the-path-of-custom-dataset)
    - [Step-2: Choose one config as template](#step-2-choose-one-config-as-template)
    - [Step-3: Edit the dataset related config](#step-3-edit-the-dataset-related-config)
  - [Train MAE on COCO Dataset](#train-mae-on-coco-dataset)
  - [Train SimCLR on Custom Dataset](#train-simclr-on-custom-dataset)

In this tutorial, we provide some tips on how to conduct self-supervised learning on your own dataset(without the need of label).

## Train MAE on Custom Dataset

In MMPretrain, We support the `CustomDataset` (similar to the `ImageFolder` in `torchvision`),  which is able to read the images within the specified folder directly. You only need to prepare the path information of the custom dataset and edit the config.

### Step-1: Get the path of custom dataset

It should be like `data/custom_dataset/`

### Step-2: Choose one config as template

Here, we would like to use `configs/mae/mae_vit-base-p16_8xb512-amp-coslr-300e_in1k.py` as the example. We first copy this config file and rename it as `mae_vit-base-p16_8xb512-amp-coslr-300e_${custom_dataset}.py`.

- `custom_dataset`: indicate which dataset you used, e.g.,`in1k` for ImageNet dataset, `coco` for COCO dataset

The content of this config is:

```python
_base_ = [
    '../_base_/models/mae_vit-base-p16.py',
    '../_base_/datasets/imagenet_bs512_mae.py',
    '../_base_/default_runtime.py',
]

# optimizer wrapper
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(
        type='AdamW',
        lr=1.5e-4 * 4096 / 256,
        betas=(0.9, 0.95),
        weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            'ln': dict(decay_mult=0.0),
            'bias': dict(decay_mult=0.0),
            'pos_embed': dict(decay_mult=0.),
            'mask_token': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.)
        }))

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.0001,
        by_epoch=True,
        begin=0,
        end=40,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=260,
        by_epoch=True,
        begin=40,
        end=300,
        convert_to_iter_based=True)
]

# runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=300)
default_hooks = dict(
    # only keeps the latest 3 checkpoints
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3))

randomness = dict(seed=0, diff_rank_seed=True)

# auto resume
resume = True

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=4096)

```

### Step-3: Edit the dataset related config

The dataset related config is defined in `'../_base_/datasets/imagenet_bs512_mae.py'` in `_base_`. We then copy the content of dataset config file into our created file `mae_vit-base-p16_8xb512-coslr-400e_${custom_dataset}.py`.

- Set the `dataset_type = 'CustomDataset'`, and the path of the custom dataset ` data_root = /dataset/my_custom_dataset`.
- Remove the `ann_file` in `train_dataloader`, and edit the `data_prefix` if needed.

And the edited config will be like this:

```python
# >>>>>>>>>>>>>>>>>>>>> Start of Changed >>>>>>>>>>>>>>>>>>>>>>>>>
_base_ = [
    '../_base_/models/mae_vit-base-p16.py',
    '../_base_/datasets/imagenet_mae.py',
    '../_base_/default_runtime.py',
]

# custom dataset
dataset_type = 'CustomDataset'
data_root = 'data/custom_dataset/'

train_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        # ann_file='meta/train.txt', # removed if you don't have the annotation file
        data_prefix=dict(img_path='./'))
# <<<<<<<<<<<<<<<<<<<<<< End of Changed <<<<<<<<<<<<<<<<<<<<<<<<<<<

# optimizer wrapper
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(
        type='AdamW',
        lr=1.5e-4 * 4096 / 256,
        betas=(0.9, 0.95),
        weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            'ln': dict(decay_mult=0.0),
            'bias': dict(decay_mult=0.0),
            'pos_embed': dict(decay_mult=0.),
            'mask_token': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.)
        }))

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.0001,
        by_epoch=True,
        begin=0,
        end=40,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=260,
        by_epoch=True,
        begin=40,
        end=300,
        convert_to_iter_based=True)
]

# runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=300)
default_hooks = dict(
    # only keeps the latest 3 checkpoints
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3))

randomness = dict(seed=0, diff_rank_seed=True)

# auto resume
resume = True

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=4096)
```

By using the edited config file, you are able to train a self-supervised model with MAE algorithm on the custom dataset.

## Train MAE on COCO Dataset

```{note}
You need to install MMDetection to use the `mmdet.CocoDataset` follow this [documentation](https://github.com/open-mmlab/mmdetection/blob/3.x/docs/en/get_started.md)
```

Follow the aforementioned idea, we also present an example of how to train MAE on COCO dataset.  The edited file will be like this:

```python
# >>>>>>>>>>>>>>>>>>>>> Start of Changed >>>>>>>>>>>>>>>>>>>>>>>>>
_base_ = [
    '../_base_/models/mae_vit-base-p16.py',
    '../_base_/datasets/imagenet_mae.py',
    '../_base_/default_runtime.py',
]

# custom dataset
dataset_type = 'mmdet.CocoDataset'
data_root = 'data/coco/'

train_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/')))

# <<<<<<<<<<<<<<<<<<<<<< End of Changed <<<<<<<<<<<<<<<<<<<<<<<<<<<

# optimizer wrapper
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(
        type='AdamW',
        lr=1.5e-4 * 4096 / 256,
        betas=(0.9, 0.95),
        weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            'ln': dict(decay_mult=0.0),
            'bias': dict(decay_mult=0.0),
            'pos_embed': dict(decay_mult=0.),
            'mask_token': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.)
        }))

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.0001,
        by_epoch=True,
        begin=0,
        end=40,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=260,
        by_epoch=True,
        begin=40,
        end=300,
        convert_to_iter_based=True)
]

# runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=300)
default_hooks = dict(
    # only keeps the latest 3 checkpoints
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3))

randomness = dict(seed=0, diff_rank_seed=True)

# auto resume
resume = True

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=4096)
```

## Train SimCLR on Custom Dataset

We provide an example of using SimCLR on custom dataset, the main idea is similar to the [Train MAE on Custom Dataset
](#train-mae-on-custom-dataset).

The template config is `configs/simclr/simclr_resnet50_8xb32-coslr-200e_in1k.py`. And the edited config is:

```python
# >>>>>>>>>>>>>>>>>>>>> Start of Changed >>>>>>>>>>>>>>>>>>>>>>>>>
_base_ = [
    '../_base_/datasets/imagenet_bs32_simclr.py',
    '../_base_/schedules/imagenet_lars_coslr_200e-200e_in1k.py',
    '../_base_/default_runtime.py',
]

# custom dataset
dataset_type = 'CustomDataset'
data_root = 'data/custom_dataset/'

train_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        # ann_file='meta/train.txt',
        data_prefix=dict(img_path='./')))
# <<<<<<<<<<<<<<<<<<<<<< End of Changed <<<<<<<<<<<<<<<<<<<<<<<<<<<

# model settings
model = dict(
    type='SimCLR',
    backbone=dict(
        type='ResNet',
        depth=50,
        norm_cfg=dict(type='SyncBN'),
        zero_init_residual=True),
    neck=dict(
        type='NonLinearNeck',  # SimCLR non-linear neck
        in_channels=2048,
        hid_channels=2048,
        out_channels=128,
        num_layers=2,
        with_avg_pool=True),
    head=dict(
        type='ContrastiveHead',
        loss=dict(type='CrossEntropyLoss'),
        temperature=0.1),
)

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='LARS', lr=0.3, momentum=0.9, weight_decay=1e-6),
    paramwise_cfg=dict(
        custom_keys={
            'bn': dict(decay_mult=0, lars_exclude=True),
            'bias': dict(decay_mult=0, lars_exclude=True),
            # bn layer in ResNet block downsample module
            'downsample.1': dict(decay_mult=0, lars_exclude=True),
        }))

# runtime settings
default_hooks = dict(
    # only keeps the latest 3 checkpoints
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=3))

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=256)
```
