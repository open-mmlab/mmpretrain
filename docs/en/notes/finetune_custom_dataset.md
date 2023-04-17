# How to Fine-tune with Custom Dataset

In most scenarios, we want to apply a pre-trained model without training from scratch, which might possibly introduce extra uncertainties about the model convergency and therefore, is time-consuming.
The common sense is to learn from previous models trained on large dataset, which can hopefully provide better knowledge than a random beginner. Roughly speaking, this process is as known as fine-tuning.

Models pre-trained on the ImageNet dataset have been demonstrated to be effective for other datasets and other downstream tasks.
Hence, this tutorial provides instructions for users to use the models provided in the [Model Zoo](../modelzoo_statistics.md) for other datasets to obtain better performance.

In this tutorial, we provide a practice example and some tips on how to fine-tune a model on your own dataset.

## Step-1: Prepare your dataset

Prepare your dataset following [Prepare Dataset](../user_guides/dataset_prepare.md).
And the root folder of the dataset can be like `data/custom_dataset/`.

Here, we assume you want to do supervised image-classification training, and use the sub-folder format
`CustomDataset` to organize your dataset as:

```text
data/custom_dataset/
├── train
│   ├── class_x
│   │   ├── x_1.png
│   │   ├── x_2.png
│   │   ├── x_3.png
│   │   └── ...
│   ├── class_y
│   └── ...
└── test
    ├── class_x
    │   ├── test_x_1.png
    │   ├── test_x_2.png
    │   ├── test_x_3.png
    │   └── ...
    ├── class_y
    └── ...
```

## Step-2: Choose one config as template

Here, we would like to use `configs/resnet/resnet50_8xb32_in1k.py` as the example. We first copy this config
file to the same folder and rename it as `resnet50_8xb32-ft_custom.py`.

```{tip}
As a convention, the last field of the config name is the dataset, e.g.,`in1k` for ImageNet dataset, `coco` for COCO dataset
```

The content of this config is:

```python
_base_ = [
    '../_base_/models/resnet50.py',           # model settings
    '../_base_/datasets/imagenet_bs32.py',    # data settings
    '../_base_/schedules/imagenet_bs256.py',  # schedule settings
    '../_base_/default_runtime.py',           # runtime settings
]
```

## Step-3: Edit the model settings

When fine-tuning a model, usually we want to load the pre-trained backbone
weights and train a new classification head from scratch.

To load the pre-trained backbone, we need to change the initialization config
of the backbone and use `Pretrained` initialization function. Besides, in the
`init_cfg`, we use `prefix='backbone'` to tell the initialization function
the prefix of the submodule that needs to be loaded in the checkpoint.

For example, `backbone` here means to load the backbone submodule. And here we
use an online checkpoint, it will be downloaded automatically during training,
you can also download the model manually and use a local path.
And then we need to modify the head according to the class numbers of the new
datasets by just changing `num_classes` in the head.

When new dataset is small and shares the domain with the pre-trained dataset,
we might want to freeze the first several stages' parameters of the
backbone, that will help the network to keep ability to extract low-level
information learnt from pre-trained model. In MMPretrain, you can simply
specify how many stages to freeze by `frozen_stages` argument. For example, to
freeze the first two stages' parameters, just use the following configs:

```{note}
Not all backbones support the `frozen_stages` argument by now. Please check
[the docs](https://mmpretrain.readthedocs.io/en/latest/api.html#module-mmpretrain.models.backbones)
to confirm if your backbone supports it.
```

```python
_base_ = [
    '../_base_/models/resnet50.py',           # model settings
    '../_base_/datasets/imagenet_bs32.py',    # data settings
    '../_base_/schedules/imagenet_bs256.py',  # schedule settings
    '../_base_/default_runtime.py',           # runtime settings
]

# >>>>>>>>>>>>>>> Override model settings here >>>>>>>>>>>>>>>>>>>
model = dict(
    backbone=dict(
        frozen_stages=2,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth',
            prefix='backbone',
        )),
    head=dict(num_classes=10),
)
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
```

```{tip}
Here we only need to set the part of configs we want to modify, because the
inherited configs will be merged and get the entire configs.
```

## Step-4: Edit the dataset settings

To fine-tuning on a new dataset, we need to override some dataset settings, like the type of dataset, data
pipeline, etc.

```python
_base_ = [
    '../_base_/models/resnet50.py',           # model settings
    '../_base_/datasets/imagenet_bs32.py',    # data settings
    '../_base_/schedules/imagenet_bs256.py',  # schedule settings
    '../_base_/default_runtime.py',           # runtime settings
]

# model settings
model = dict(
    backbone=dict(
        frozen_stages=2,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth',
            prefix='backbone',
        )),
    head=dict(num_classes=10),
)

# >>>>>>>>>>>>>>> Override data settings here >>>>>>>>>>>>>>>>>>>
data_root = 'data/custom_dataset'
train_dataloader = dict(
    dataset=dict(
        type='CustomDataset',
        data_root=data_root,
        ann_file='',       # We assume you are using the sub-folder format without ann_file
        data_prefix='train',
    ))
val_dataloader = dict(
    dataset=dict(
        type='CustomDataset',
        data_root=data_root,
        ann_file='',       # We assume you are using the sub-folder format without ann_file
        data_prefix='test',
    ))
test_dataloader = val_dataloader
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
```

## Step-5: Edit the schedule settings (optional)

The fine-tuning hyper parameters vary from the default schedule. It usually
requires smaller learning rate and quicker decaying scheduler epochs.

```python
_base_ = [
    '../_base_/models/resnet50.py',           # model settings
    '../_base_/datasets/imagenet_bs32.py',    # data settings
    '../_base_/schedules/imagenet_bs256.py',  # schedule settings
    '../_base_/default_runtime.py',           # runtime settings
]

# model settings
model = dict(
    backbone=dict(
        frozen_stages=2,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth',
            prefix='backbone',
        )),
    head=dict(num_classes=10),
)

# data settings
data_root = 'data/custom_dataset'
train_dataloader = dict(
    dataset=dict(
        type='CustomDataset',
        data_root=data_root,
        ann_file='',       # We assume you are using the sub-folder format without ann_file
        data_prefix='train',
    ))
val_dataloader = dict(
    dataset=dict(
        type='CustomDataset',
        data_root=data_root,
        ann_file='',       # We assume you are using the sub-folder format without ann_file
        data_prefix='test',
    ))
test_dataloader = val_dataloader

# >>>>>>>>>>>>>>> Override schedule settings here >>>>>>>>>>>>>>>>>>>
# optimizer hyper-parameters
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))
# learning policy
param_scheduler = dict(
    type='MultiStepLR', by_epoch=True, milestones=[15], gamma=0.1)
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
```

```{tip}
Refers to [Learn about Configs](../user_guides/config.md) for more detailed configurations.
```

## Start Training

Now, we have finished the fine-tuning config file as following:

```python
_base_ = [
    '../_base_/models/resnet50.py',           # model settings
    '../_base_/datasets/imagenet_bs32.py',    # data settings
    '../_base_/schedules/imagenet_bs256.py',  # schedule settings
    '../_base_/default_runtime.py',           # runtime settings
]

# model settings
model = dict(
    backbone=dict(
        frozen_stages=2,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth',
            prefix='backbone',
        )),
    head=dict(num_classes=10),
)

# data settings
data_root = 'data/custom_dataset'
train_dataloader = dict(
    dataset=dict(
        type='CustomDataset',
        data_root=data_root,
        ann_file='',       # We assume you are using the sub-folder format without ann_file
        data_prefix='train',
    ))
val_dataloader = dict(
    dataset=dict(
        type='CustomDataset',
        data_root=data_root,
        ann_file='',       # We assume you are using the sub-folder format without ann_file
        data_prefix='test',
    ))
test_dataloader = val_dataloader

# schedule settings
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))
param_scheduler = dict(
    type='MultiStepLR', by_epoch=True, milestones=[15], gamma=0.1)
```

Here we use 8 GPUs on your computer to train the model with the following command:

```shell
bash tools/dist_train.sh configs/resnet/resnet50_8xb32-ft_custom.py 8
```

Also, you can use only one GPU to train the model with the following command:

```shell
python tools/train.py configs/resnet/resnet50_8xb32-ft_custom.py
```

But wait, an important config need to be changed if using one GPU. We need to
change the dataset config as following:

```python
data_root = 'data/custom_dataset'
train_dataloader = dict(
    batch_size=256,
    dataset=dict(
        type='CustomDataset',
        data_root=data_root,
        ann_file='',       # We assume you are using the sub-folder format without ann_file
        data_prefix='train',
    ))
val_dataloader = dict(
    dataset=dict(
        type='CustomDataset',
        data_root=data_root,
        ann_file='',       # We assume you are using the sub-folder format without ann_file
        data_prefix='test',
    ))
test_dataloader = val_dataloader
```

It's because our training schedule is for a batch size of 256. If using 8 GPUs,
just use `batch_size=32` config in the base config file for every GPU, and the total batch
size will be 256. But if using one GPU, you need to change it to 256 manually to
match the training schedule.

However, a larger batch size requires a larger GPU memory, and here are several simple tricks to save the GPU
memory:

1. Enable Automatic-Mixed-Precision training.

   ```shell
   python tools/train.py configs/resnet/resnet50_8xb32-ft_custom.py --amp
   ```

2. Use a smaller batch size, like `batch_size=32` instead of 256, and enable the auto learning rate scaling.

   ```shell
   python tools/train.py configs/resnet/resnet50_8xb32-ft_custom.py --auto-scale-lr
   ```

   The auto learning rate scaling will adjust the learning rate according to the actual batch size and the
   `auto_scale_lr.base_batch_size` (You can find it in the base config
   `configs/_base_/schedules/imagenet_bs256.py`)

```{note}
Most of these tricks may influence the training performance slightly.
```

### Apply pre-trained model with command line

If you don't want to modify the configs, you could use `--cfg-options` to add your pre-trained model path to `init_cfg`.

For example, the command below will also load pre-trained model.

```shell
bash tools/dist_train.sh configs/resnet/resnet50_8xb32-ft_custom.py 8 \
    --cfg-options model.backbone.init_cfg.type='Pretrained' \
    model.backbone.init_cfg.checkpoint='https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_resnet50_8xb512-amp-coslr-100e_in1k/mocov3_resnet50_8xb512-amp-coslr-100e_in1k_20220927-f1144efa.pth' \
    model.backbone.init_cfg.prefix='backbone' \
```
