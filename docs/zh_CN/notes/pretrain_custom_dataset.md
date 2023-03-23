# 如何在自定义数据集上进行模型预训练

在本教程中，我们提供了一个实践示例和一些有关如何在您自己的数据集上进行训练的技巧。

在 MMPretrain 中，我们支持用户直接调用 MMPretrain 的 `CustomDataset` （类似于 `torchvision` 的 `ImageFolder`）, 该数据集能自动的读取给的路径下的图片。你只需要准备你的数据集路径，并修改配置文件，即可轻松使用 MMPretrain 进行预训练。

## 第一步：准备你的数据集

按照 [准备数据集](../user_guides/dataset_prepare.md) 准备你的数据集。
假设我们的数据集根文件夹路径为 `data/custom_dataset/`

假设我们想使用 MAE 算法进行图像自监督训练，并使用子文件夹格式的 `CustomDataset` 来组织数据集：

```text
data/custom_dataset/
├── sample1.png
├── sample2.png
├── sample3.png
├── sample4.png
└── ...
```

## 第二步：选择一个配置文件作为模板

在本教程中，我们使用 `configs/mae/mae_vit-base-p16_8xb512-amp-coslr-300e_in1k.py` 作为一个示例进行介绍。
首先在同一文件夹下复制一份配置文件，并将其重命名为 `mae_vit-base-p16_8xb512-amp-coslr-300e_custom.py`。

```{tip}
按照惯例，配置名称的最后一个字段是数据集，例如，`in1k` 表示 ImageNet-1k，`coco` 表示 coco 数据集
```

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

## 第三步：修改数据集设置

- 重载数据集设置中的 `type` 为 `'CustomDataset'`
- 重载数据集设置中的 `data_root` 为 `data/custom_dataset`
- 重载数据集设置中的 `ann_file` 为空字符串，这是因为我们使用子文件格式的 `CustomDataset`，需要将配置文件置空
- 重载数据集设置中的 `data_prefix` 为空字符串，这是因为我们希望使用数据集根目录下的所有数据进行训练，并不需要将其拆分为不同子集。

修改后的文件应如下：

```python
_base_ = [
    '../_base_/models/mae_vit-base-p16.py',
    '../_base_/datasets/imagenet_bs512_mae.py',
    '../_base_/default_runtime.py',
]

# >>>>>>>>>>>>>>> 在此重载数据设置 >>>>>>>>>>>>>>>>>>>
train_dataloader = dict(
    dataset=dict(
        type='CustomDataset',
        data_root='data/custom_dataset/',
        ann_file='',       # 我们假定使用子文件夹格式，因此需要将标注文件置空
        data_prefix='',    # 使用 `data_root` 路径下所有数据
        with_label=False,
    )
)
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

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

## 另一个例子：在 COCO 数据集上训练 MAE

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

# >>>>>>>>>>>>>>> 在这里重载数据配置 >>>>>>>>>>>>>>>>>>>
train_dataloader = dict(
    dataset=dict(
        type='mmdet.CocoDataset',
        data_root='data/coco/',
        ann_file='annotations/instances_train2017.json',  # 仅用于加载图片，不会使用标签
        data_prefix=dict(img='train2017/'),
    )
)
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

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
