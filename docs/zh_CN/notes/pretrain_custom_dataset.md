# 教程 4: 使用自定义数据集进行预训练

- [教程 4: 使用自定义数据集进行预训练](#教程-4-使用自定义数据集进行预训练)
  - [在自定义数据集上使用 MAE 算法进行预训练](#在自定义数据集上使用-mae-算法进行预训练)
    - [第一步：获取自定义数据路径](#第一步获取自定义数据路径)
    - [第二步：选择一个配置文件作为模板](#第二步选择一个配置文件作为模板)
    - [第三步：修改数据集相关的配置](#第三步修改数据集相关的配置)
  - [在 COCO 数据集上使用 MAE 算法进行预训练](#在-coco-数据集上使用-mae-算法进行预训练)

在本教程中，我们将介绍如何使用自定义数据集(无需标注)进行自监督预训练。

## 在自定义数据集上使用 MAE 算法进行预训练

在 MMPretrain 中, 我们支持用户直接调用 MMPretrain 的 `CustomDataset` (类似于 `torchvision` 的 `ImageFolder`), 该数据集能自动的读取给的路径下的图片。你只需要准备你的数据集路径，并修改配置文件，即可轻松使用 MMPretrain 进行预训练。

### 第一步：获取自定义数据路径

路径应类似这种形式： `data/custom_dataset/`

### 第二步：选择一个配置文件作为模板

在本教程中，我们使用 `configs/selfsup/mae/mae_vit-base-p16_8xb512-coslr-400e_in1k.py`作为一个示例进行讲解。我们首先复制这个配置文件，将新复制的文件命名为`mae_vit-base-p16_8xb512-coslr-400e_${custom_dataset}.py`.

- `custom_dataset`: 表明你用的那个数据集。例如，用 `in1k` 代表ImageNet 数据集，`coco` 代表COCO数据集。

这个配置文件的内容如下：

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

### 第三步：修改数据集相关的配置

数据集相关的配置是定义在 `_base_`的`'../_base_/datasets/imagenet_mae.py'` 文件内。我们直接将其内容复制到刚刚创建的新的配置文件 `mae_vit-base-p16_8xb512-coslr-400e_${custom_dataset}.py` 中.

- 修改`dataset_type = 'CustomDataset'`和` data_root = /dataset/my_custom_dataset`.
- 删除 `train_dataloader`中的 `ann_file` ，同时根据自己的实际情况决定是否需要设定 `data_prefix`。

```{note}
`CustomDataset` 是在 MMPretrain 实现的, 因此我们使用这种方式 `dataset_type=CustomDataset` 来使用这个类。
```

此时，修改后的文件应如下：

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

使用上述配置文件，你就能够轻松的在自定义数据集上使用 `MAE` 算法来进行预训练了。

## 在 COCO 数据集上使用 MAE 算法进行预训练

```{note}
你可能需要参考[文档](https://github.com/open-mmlab/mmdetection/blob/3.x/docs/en/get_started.md)安装 MMDetection 来使用 `mmdet.CocoDataset`。
```

与在自定义数据集上进行预训练类似，我们在本教程中也提供了一个使用 COCO 数据集进行预训练的示例。修改后的文件如下：

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
