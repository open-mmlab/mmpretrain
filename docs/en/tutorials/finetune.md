# Tutorial 2: Fine-tune Models

Classification models pre-trained on the ImageNet dataset have been demonstrated to be effective for other datasets and other downstream tasks.
This tutorial provides instructions for users to use the models provided in the [Model Zoo](../model_zoo.md) for other datasets to obtain better performance.

There are two steps to fine-tune a model on a new dataset.

- Add support for the new dataset following [Tutorial 3: Adding New Dataset](new_dataset.md).
- Modify the configs as will be discussed in this tutorial.

Assume we have a ResNet-50 model pre-trained on the ImageNet-2012 dataset and want
to take the fine-tuning on the CIFAR-10 dataset, we need to modify five parts in the
config.

## Inherit base configs

At first, create a new config file
`configs/tutorial/resnet50_finetune_cifar.py` to store our configs. Of course,
the path can be customized by yourself.

To reuse the common parts among different configs, we support inheriting
configs from multiple existing configs. To fine-tune a ResNet-50 model, the new
config needs to inherit `configs/_base_/models/resnet50.py` to build the basic
structure of the model. To use the CIFAR-10 dataset, the new config can also
simply inherit `configs/_base_/datasets/cifar10_bs16.py`. For runtime settings such as
training schedules, the new config needs to inherit
`configs/_base_/default_runtime.py`.

To inherit all above configs, put the following code at the config file.

```python
_base_ = [
    '../_base_/models/resnet50.py',
    '../_base_/datasets/cifar10_bs16.py', '../_base_/default_runtime.py'
]
```

Besides, you can also choose to write the whole contents rather than use inheritance,
like [`configs/lenet/lenet5_mnist.py`](https://github.com/open-mmlab/mmclassification/blob/master/configs/lenet/lenet5_mnist.py).

## Modify model

When fine-tuning a model, usually we want to load the pre-trained backbone
weights and train a new classification head.

To load the pre-trained backbone, we need to change the initialization config
of the backbone and use `Pretrained` initialization function. Besides, in the
`init_cfg`, we use `prefix='backbone'` to tell the initialization
function to remove the prefix of keys in the checkpoint, for example, it will
change `backbone.conv1` to `conv1`. And here we use an online checkpoint, it
will be downloaded during training, you can also download the model manually
and use a local path.

And then we need to modify the head according to the class numbers of the new
datasets by just changing `num_classes` in the head.

```python
model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth',
            prefix='backbone',
        )),
    head=dict(num_classes=10),
)
```

```{tip}
Here we only need to set the part of configs we want to modify, because the
inherited configs will be merged and get the entire configs.
```

Sometimes, we want to freeze the first several layers' parameters of the
backbone, that will help the network to keep ability to extract low-level
information learnt from pre-trained model. In MMClassification, you can simply
specify how many layers to freeze by `frozen_stages` argument. For example, to
freeze the first two layers' parameters, just use the following config:

```python
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
```

```{note}
Not all backbones support the `frozen_stages` argument by now. Please check
[the docs](https://mmclassification.readthedocs.io/en/latest/api.html#module-mmcls.models.backbones)
to confirm if your backbone supports it.
```


## Modify dataset

When fine-tuning on a new dataset, usually we need to modify some dataset
configs. Here, we need to modify the pipeline to resize the image from 32 to
224 to fit the input size of the model pre-trained on ImageNet, and some other
configs.

```python
img_norm_cfg = dict(
    mean=[125.307, 122.961, 113.8575],
    std=[51.5865, 50.847, 51.255],
    to_rgb=False,
)
train_pipeline = [
    dict(type='RandomCrop', size=32, padding=4),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Resize', size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label']),
]
test_pipeline = [
    dict(type='Resize', size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img']),
]
data = dict(
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline),
)
```

## Modify training schedule

The fine-tuning hyper parameters vary from the default schedule. It usually
requires smaller learning rate and less training epochs.

```python
# lr is set for a batch size of 128
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[15])
runner = dict(type='EpochBasedRunner', max_epochs=200)
log_config = dict(interval=100)
```

## Start Training

Now, we have finished the fine-tuning config file as following:

```python
_base_ = [
    '../_base_/models/resnet50.py',
    '../_base_/datasets/cifar10_bs16.py', '../_base_/default_runtime.py'
]

# Model config
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

# Dataset config
img_norm_cfg = dict(
    mean=[125.307, 122.961, 113.8575],
    std=[51.5865, 50.847, 51.255],
    to_rgb=False,
)
train_pipeline = [
    dict(type='RandomCrop', size=32, padding=4),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Resize', size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label']),
]
test_pipeline = [
    dict(type='Resize', size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img']),
]
data = dict(
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline),
)

# Training schedule config
# lr is set for a batch size of 128
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[15])
runner = dict(type='EpochBasedRunner', max_epochs=200)
log_config = dict(interval=100)
```

Here we use 8 GPUs on your computer to train the model with the following
command:

```shell
bash tools/dist_train.sh configs/tutorial/resnet50_finetune_cifar.py 8
```

Also, you can use only one GPU to train the model with the following command:

```shell
python tools/train.py configs/tutorial/resnet50_finetune_cifar.py
```

But wait, an important config need to be changed if using one GPU. We need to
change the dataset config as following:

```python
data = dict(
    samples_per_gpu=128,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline),
)
```

It's because our training schedule is for a batch size of 128. If using 8 GPUs,
just use `samples_per_gpu=16` config in the base config file, and the total batch
size will be 128. But if using one GPU, you need to change it to 128 manually to
match the training schedule.
