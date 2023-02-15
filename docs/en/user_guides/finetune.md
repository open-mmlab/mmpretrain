# Fine-tune Models

In most scenarios, we want to apply a model on new datasets without training from scratch, which might possibly introduce extra uncertainties about the model convergency and therefore, is time-consuming.
The common sense is to learn from previous models trained on large dataset, which can hopefully provide better knowledge than a random beginner. Roughly speaking, this process is as known as fine-tuning.
Classification models pre-trained on the ImageNet dataset have been demonstrated to be effective for other datasets and other downstream tasks.
Hence, this tutorial provides instructions for users to use the models provided in the [Model Zoo](../modelzoo_statistics.md) for other datasets to obtain better performance.

There are two steps to fine-tune a model on a new dataset.

- Add support for the new dataset following [Prepare Dataset](dataset_prepare.md).
- Modify the configs as will be discussed in this tutorial.

Assume we have a ResNet-50 model pre-trained on the ImageNet-2012 dataset and want
to fine-tune on the CIFAR-10 dataset, we need to modify five parts in the config.

## Inherit base configs

At first, create a new config file
`configs/tutorial/resnet50_finetune_cifar.py` to store our fine-tune configs. Of course,
the path can be customized by yourself.

To reuse the common parts among different base configs, we support inheriting
configs from multiple existing configs.Including following four parts：

- Model configs: To fine-tune a ResNet-50 model, the new
  config needs to inherit `configs/_base_/models/resnet50.py` to build the basic structure of the model.
- Dataset configs: To use the CIFAR-10 dataset, the new config can simply
  inherit `configs/_base_/datasets/cifar10_bs16.py`.
- Schedule configs: The new config can inherit `_base_/schedules/cifar10_bs128.py`
  for CIFAR-10 dataset with a batch size of 128.
- Runtime configs: For runtime settings such as basic hooks, etc.,
  the new config needs to inherit `configs/_base_/default_runtime.py`.

To inherit all configs above, put the following code at the config file.

```python
_base_ = [
    '../_base_/models/resnet50.py',
    '../_base_/datasets/cifar10_bs16.py',
    '../_base_/schedules/cifar10_bs128.py',
    '../_base_/default_runtime.py',
]
```

Besides, you can also choose to write the whole contents rather than use inheritance.
Refers to [`configs/lenet/lenet5_mnist.py`](https://github.com/open-mmlab/mmclassification/blob/master/configs/lenet/lenet5_mnist.py) for more details.

## Modify model configs

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

When new dataset is small and shares the domain with the pre-trained dataset,
we might want to freeze the first several stages' parameters of the
backbone, that will help the network to keep ability to extract low-level
information learnt from pre-trained model. In MMClassification, you can simply
specify how many stages to freeze by `frozen_stages` argument. For example, to
freeze the first two stages' parameters, just use the following configs:

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
[the docs](https://mmclassification.readthedocs.io/en/1.x/api.html#module-mmcls.models.backbones)
to confirm if your backbone supports it.
```

## Modify dataset configs

When fine-tuning on a new dataset, usually we need to modify some dataset
configs. Here, we need to modify the pipeline to resize the image from 32 to
224 to fit the input size of the model pre-trained on ImageNet, and modify
dataloaders correspondingly.

```python
# data pipeline settings
train_pipeline = [
    dict(type='RandomCrop', crop_size=32, padding=4),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='Resize', scale=224),
    dict(type='PackClsInputs'),
]
test_pipeline = [
    dict(type='Resize', scale=224),
    dict(type='PackClsInputs'),
]
# dataloader settings
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader
```

## Modify training schedule configs

The fine-tuning hyper parameters vary from the default schedule. It usually
requires smaller learning rate and quicker decaying scheduler epochs.

```python
# lr is set for a batch size of 128
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))
# learning policy
param_scheduler = dict(
    type='MultiStepLR', by_epoch=True, milestones=[15], gamma=0.1)
```

```{tip}
Refers to [Learn about Configs](config.md) for more detailed configurations.
```

## Start Training

Now, we have finished the fine-tuning config file as following:

```python
_base_ = [
    '../_base_/models/resnet50.py',
    '../_base_/datasets/cifar10_bs16.py',
    '../_base_/schedules/cifar10_bs128.py',
    '../_base_/default_runtime.py',
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
# data pipeline settings
train_pipeline = [
    dict(type='RandomCrop', crop_size=32, padding=4),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='Resize', scale=224),
    dict(type='PackClsInputs'),
]
test_pipeline = [
    dict(type='Resize', scale=224),
    dict(type='PackClsInputs'),
]
# dataloader settings
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

# Training schedule config
# lr is set for a batch size of 128
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))
# learning policy
param_scheduler = dict(
    type='MultiStepLR', by_epoch=True, milestones=[15], gamma=0.1)
```

Here we use 8 GPUs on your computer to train the model with the following command:

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
train_dataloader = dict(
    batch_size=128,
    dataset=dict(pipeline=train_pipeline),
)
val_dataloader = dict(
    batch_size=128,
    dataset=dict(pipeline=test_pipeline),
)
test_dataloader = val_dataloader
```

It's because our training schedule is for a batch size of 128. If using 8 GPUs,
just use `batch_size=16` config in the base config file for every GPU, and the total batch
size will be 128. But if using one GPU, you need to change it to 128 manually to
match the training schedule.

## Evaluate the fine-tuned model on ImageNet variants

It's a common practice to evaluate the ImageNet-(1K, 21K) fine-tuned model on the ImageNet-1K validation set. This set
shares similar data distribution with the training set, but in real world, the inference data is more likely to share
different data distribution with the training set. To have a full evaluation of model's performance on
out-of-distribution datasets, research community introduces the ImageNet-variant datasets, which shares different data
distribution with that of ImageNet-(1K, 21K)., MMClassification supports evaluating the fine-tuned model on
[ImageNet-Adversarial (A)](https://arxiv.org/abs/1907.07174), [ImageNet-Rendition (R)](https://arxiv.org/abs/2006.16241),
[ImageNet-Corruption (C)](https://arxiv.org/abs/1903.12261), and [ImageNet-Sketch (S)](https://arxiv.org/abs/1905.13549).
You can follow these steps below to have a try:

### Prepare the datasets

You can download these datasets from [OpenDataLab](https://opendatalab.com/) and refactor these datasets under the
`data` folder in the following format:

```text
   imagenet-a
        ├── meta
        │   └── val.txt
        ├── val
   imagenet-r
        ├── meta
        │   └── val.txt
        ├── val/
   imagenet-s
        ├── meta
        │   └── val.txt
        ├── val/
   imagenet-c
        ├── meta
        │   └── val.txt
        ├── val/
```

`val.txt` is the annotation file, which should have the same style as that of ImageNet-1K. You can refer to
[prepare_dataset](https://mmclassification.readthedocs.io/en/1.x/user_guides/dataset_prepare.html) to generate the
annotation file or you can refer to this [script](https://github.com/open-mmlab/mmclassification/tree/dev-1.x/projects/example_project/ood_eval/generate_imagenet_variant_annotation.py).

### Configure the dataset and test evaluator

Once the dataset is ready, you need to configure the `dataset` and `test_evaluator`. You have two options to
write the default settings:

#### 1. Change the configuration file directly

There are few modifications to the config file, but change the `data_root` of the test dataloader and pass the
annotation file to the `test_evaluator`.

```python
# You should replace imagenet-x below with imagenet-c, imagenet-r, imagenet-a
# or imagenet-s
test_dataloader=dict(dataset=dict(data_root='data/imagenet-x'))
test_evaluator=dict(ann_file='data/imagenet-x/meta/val.txt')
```

#### 2. Overwrite the default settings from command line

For example, you can overwrite the default settings by passing `--cfg-options`:

```bash
--cfg-options test_dataloader.dataset.data_root='data/imagenet-x' \
              test_evaluator.ann_file='data/imagenet-x/meta/val.txt'
```

### Start test

This step is the common test step, you can follow this [guide](https://mmclassification.readthedocs.io/en/1.x/user_guides/train_test.html)
to evaluate your fine-tuned model on out-of-distribution datasets.

To make it easier, we also provide an off-the-shelf config files, for [ImageNet-C](https://github.com/open-mmlab/mmclassification/tree/dev-1.x/projects/example_project/ood_eval/vit_ood-eval_toy-example.py) and [ImageNet-C](https://github.com/open-mmlab/mmclassification/tree/dev-1.x/projects/example_project/ood_eval/vit_ood-eval_toy-example_imagnet-c.py), and you can have a try.
