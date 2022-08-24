# Learn about Configs

MMClassification mainly uses python files as configuration files, it is mainly divided into multiple modules and can be inherited. All configuration files are placed under the [`configs`](https://github.com/open-mmlab/mmclassification/tree/master/configs) folder, the directory structure is as follows:

```text
MMClassification/
    ├── configs/
    │   ├── _base_/                       # primitive configuration folder
    │   │   ├── datasets/                      # primitive datasets
    │   │   ├── models/                        # primitive models
    │   │   ├── schedules/                     # primitive schedules
    │   │   └── default_runtime.py             # primitive runtime
    │   ├── resnet/                       # ResNet Algorithms Folder
    │   ├── swin_transformer/             # Swin Algorithms Folder
    │   ├── vision_transformer/           # ViT Algorithms Folder
    │   ├── ...
    └── ...
```

If you wish to inspect the config file, you may run `python tools/misc/print_config.py /PATH/TO/CONFIG` to see the complete config.

This article mainly explains the naming convention and the structure of configuration files, and how to modify it based on the existing configuration files. We also take [ResNet50 primitive configuration file](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet50_8xb32_in1k.py) as an example and explained line by line.

<!-- TOC -->

- [Naming Convention](#naming-convention)
- [Config Structure](#config-structure)
- [Inherit and Modify Config File](#inherit-and-modify-config-file)
  - [Use intermediate variables in configs](#use-intermediate-variables-in-configs)
  - [Ignore some fields in the base configs](#ignore-some-fields-in-the-base-configs)
  - [Use some fields in the base configs](#use-some-fields-in-the-base-configs)
- [Modify config in command](#modify-config-in-command)
- [Import user-defined modules](#import-user-defined-modules)
- [FAQ](#faq)

<!-- TOC -->

## Naming Convention

We follow the below convention to name config files. Contributors are advised to follow the same style. The config file names are divided into four parts: algorithm info, module information, training information and data information. Logically, different parts are concatenated by underscores `'_'`, and words in the same part are concatenated by dashes `'-'`.

```text
{algorithm info}_{module info}_{training info}_{data info}.py
```

- `algorithm info`：algorithm information, model name and neural network architecture, such as resnet, etc.;
- `module info`： module information is used to represent some special neck, head and pretrain information;
- `training info`：Training information, some training schedule, including batch size, lr schedule, data augment and the like;
- `data info`：Data information, dataset name, input size and so on, such as imagenet, cifar, etc.;

### Algorithm information

The main algorithm name and the corresponding branch architecture information. E.g：

- `resnet50`
- `mobilenet-v3-large`
- `vit-small-patch32`   : `patch32` represents the size of the partition in `ViT` algorithm;
- `seresnext101-32x4d`  : `SeResNet101` network structure, `32x4d` means that `groups` and `width_per_group` are 32 and 4 respectively in `Bottleneck`;

### Module information

Some special `neck`, `head` and `pretrain` information. In classification tasks, `pretrain` information is the most commonly used:

- `in21k-pre` : pre-trained on ImageNet21k;
- `in21k-pre-3rd-party` : pre-trained on ImageNet21k and the checkpoint is converted from a third-party repository;

### Training information

Training schedule, including training type, `batch size`, `lr schedule`, data augment, special loss functions and so on:

- format `{gpu x batch_per_gpu}`, such as `8xb32`

Training type (mainly seen in the transformer network, such as the `ViT` algorithm, which is usually divided into two training type: pre-training and fine-tuning):

- `ft` : configuration file for fine-tuning
- `pt` : configuration file for pretraining

Training recipe. Usually, only the part that is different from the original paper will be marked. These methods will be arranged in the order `{pipeline aug}-{train aug}-{loss trick}-{scheduler}-{epochs}`.

- `coslr-200e` : use cosine scheduler to train 200 epochs
- `autoaug-mixup-lbs-coslr-50e` : use `autoaug`, `mixup`, `label smooth`, `cosine scheduler` to train 50 epochs

### Data information

- `in1k` : `ImageNet1k` dataset, default to use the input image size of 224x224;
- `in21k` : `ImageNet21k` dataset, also called `ImageNet22k` dataset, default to use the input image size of 224x224;
- `in1k-384px` : Indicates that the input image size is 384x384;
- `cifar100`

### Config File Name Example

```text
repvgg-D2se_deploy_4xb64-autoaug-lbs-mixup-coslr-200e_in1k.py
```

- `repvgg-D2se`:  Algorithm information
  - `repvgg`: The main algorithm.
  - `D2se`: The architecture.
- `deploy`: Module information, means the backbone is in the deploy state.
- `4xb64-autoaug-lbs-mixup-coslr-200e`: Training information.
  - `4xb64`: Use 4 GPUs and the size of batches per GPU is 64.
  - `autoaug`: Use `AutoAugment` in training pipeline.
  - `lbs`: Use label smoothing loss.
  - `mixup`: Use `mixup` training augment method.
  - `coslr`: Use cosine learning rate scheduler.
  - `200e`: Train the model for 200 epochs.
- `in1k`: Dataset information. The config is for `ImageNet1k` dataset and the input size is `224x224`.

```{note}
Some configuration files currently do not follow this naming convention, and related files will be updated in the near future.
```

### Checkpoint Naming Convention

The naming of the weight mainly includes the configuration file name, date and hash value.

```text
{config_name}_{date}-{hash}.pth
```

## Config Structure

There are four kinds of basic component file in the `configs/_base_` folders, namely：

- [models](https://github.com/open-mmlab/mmclassification/tree/master/configs/_base_/models)
- [datasets](https://github.com/open-mmlab/mmclassification/tree/master/configs/_base_/datasets)
- [schedules](https://github.com/open-mmlab/mmclassification/tree/master/configs/_base_/schedules)
- [runtime](https://github.com/open-mmlab/mmclassification/blob/master/configs/_base_/default_runtime.py)

You can easily build your own training config file by inherit some base config files. And the configs that are composed by components from `_base_` are called _primitive_.

For easy understanding, we use [ResNet50 primitive config](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet50_8xb32_in1k.py) as a example and comment the meaning of each line. For more detaile, please refer to the API documentation.

```python
_base_ = [                                    # _base_ can be a list or a str
    '../_base_/models/resnet50.py',           # model
    '../_base_/datasets/imagenet_bs32.py',    # data
    '../_base_/schedules/imagenet_bs256.py',  # training schedule
    '../_base_/default_runtime.py'            # runtime setting
]
```

The four parts are explained separately below, and the above-mentioned ResNet50 primitive config are also used as an example.

### model

The parameter `"model"` is a python dictionary in the configuration file, which mainly includes information such as network structure and loss function:

- `type`： Classifier name, MMCls supports `ImageClassifier`, refer to [API documentation](https://mmclassification.readthedocs.io/en/latest/api/models.html#classifier). The supported classification algorithms can be viewed in [`model zoo`](https://mmclassification.readthedocs.io/en/latest/model_zoo.html).
- `data_preprocessor`: The component before model to preprocess the inputs, e.g., `ClsDataPreprocessor`, refer to [API documentation](TODO:).
- `backbone`： Backbone config, MMCls supports `ResNet`, `Swin Transformer`, `Vision Transformer` etc., For available options, refer to [API documentation](https://mmclassification.readthedocs.io/en/latest/api/models.html#backbones).
- `neck`：Neck network name, MMCls supports `GlobalAveragePooling` etc., For available options, refer to [API documentation](https://mmclassification.readthedocs.io/en/latest/api/models.html#necks).
- `head`: Head network name, MMCls supports single-label and multi-label classification head networks, available options refer to [API documentation](https://mmclassification.readthedocs.io/en/latest/api/models.html#heads).
  - `loss`: Loss function type, MMCls supports `CrossEntropyLoss`, `LabelSmoothLoss` etc., For available options, refer to [API documentation](https://mmclassification.readthedocs.io/en/latest/api/models.html#losses).
- `train_cfg`：Training augment config, MMCls support `Mixup`, `CutMix` etc., please refer [API documentation](TODO:) for more batch augments.

```{note}
The 'type' in the configuration file is not a constructed parameter, but a class name.
```

Following is the model configuration of ResNet50 primitive config in ['configs/_base_/models/resnet50.py'](https://github.com/open-mmlab/mmclassification/blob/master/configs/_base_/models/resnet50.py):

```python
model = dict(
    type='ImageClassifier',     # Classifier name
    backbone=dict(
        type='ResNet',          # Backbones name
        depth=50,               # depth of backbone, ResNet has options of 18, 34, 50, 101, 152.
        num_stages=4,           # number of stages，The feature maps generated by these states are used as the input for the subsequent neck and head.
        out_indices=(3, ),      # The output index of the output feature maps.
        frozen_stages=-1,       # the stage to be frozen, '-1' means not be forzen
        style='pytorch'),        # The style of backbone, 'pytorch' means that stride 2 layers are in 3x3 conv, 'caffe' means stride 2 layers are in 1x1 convs.
    neck=dict(type='GlobalAveragePooling'),    # neck network name
    head=dict(
        type='LinearClsHead',     # linear classification head，
        num_classes=1000,         # The number of output categories, consistent with the number of categories in the dataset
        in_channels=2048,         # The number of input channels, consistent with the output channel of the neck
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0), # Loss function configuration information
        topk=(1, 5),              # Evaluation index, Top-k accuracy rate, here is the accuracy rate of top1 and top5
    ))
```

### data

The part `"data"` in the config includes information to construct dataloader and evaluator:

- `preprocess_cfg`: Model input preprocessing configuration, same as `model.data_preprocessor` but with higher priority.
- `train_evaluator | val_evaluator | test_evaluator`: To build the evaluator, refer to the [API documentation](TODO:).
- `train_dataloader | val_dataloader | test_dataloader`: build dataloader
  - `samples_per_gpu`: the BatchSize of each GPU when building the dataloader
  - `workers_per_gpu`: the number of threads per GPU when building dataloader
  - `sampler`: sampler configuration
  - `dataset`: Construct a dataset.
    - `type`: dataset type, MMClassification supports `ImageNet`, `Cifar` and other datasets, refer to [API documentation](https://mmclassification.readthedocs.io/en/latest/api.html#module-mmcls.datasets)
    - `pipeline`: data processing pipeline, refer to the related tutorial document [How to Design a Data Processing Pipeline](https://mmclassification.readthedocs.io/en/latest/tutorials/data_pipeline.html)

Following is the data configuration of ResNet50 primitive config in ['configs/_base_/datasets/imagenet_bs32.py'](https://github.com/open-mmlab/mmclassification/blob/master/configs/_base_/datasets/imagenet_bs32.py)：

```python
dataset_type = 'ImageNet'
# preprocessing configuration
preprocess_cfg = dict(
    # Input image data channels in 'RGB' order
    mean=[123.675, 116.28, 103.53],    # Input image normalized channel mean in RGB order
    std=[58.395, 57.12, 57.375],       # Input image normalized channel std in RGB order
    to_rgb=True,                       # Whether to flip the channel from BGR to RGB or RGB to BGR
)

train_pipeline = [
    dict(type='LoadImageFromFile'),     # read image
    dict(type='RandomResizedCrop', scale=224),     # Random scaling and cropping
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),   # random horizontal flip
    dict(type='PackClsInputs'),         # prepare images and labels
]

test_pipeline = [
    dict(type='LoadImageFromFile'),     # read image
    dict(type='ResizeEdge', scale=256, edge='short'),  # Scale the short side to 256
    dict(type='CenterCrop', crop_size=224),     # center crop
    dict(type='PackClsInputs'),                 # prepare images and labels
]

# Construct training set dataloader
train_dataloader = dict(
    batch_size=32,                     # batchsize per GPU
    num_workers=5,                     # Number of threads per GPU
    dataset=dict(                      # training dataset
        type=dataset_type,
        data_root='data/imagenet',
        ann_file='meta/train.txt',
        data_prefix='train',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),   # default sampler
    persistent_workers=True,                             # Whether to keep the process, can shorten the preparation time of each epoch
)

# Construct the validation set dataloader
val_dataloader = dict(
    batch_size=32,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root='data/imagenet',
        ann_file='meta/val.txt',
        data_prefix='val',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True,
)
# Build the validation set evaluator, using the accuracy as the indicator
val_evaluator = dict(type='Accuracy', topk=(1, 5))

test_dataloader = val_dataloader  # Construct the test set dataloader, which is directly the same as val_dataloader
test_evaluator = val_evaluator    # Construct the test set devaluator, which is directly the same as val_evaluator
```

```note
'model.data_preprocessor' can be defined either in `model=dict(data_preprocessor=dict())` or using the `preprocess_cfg` definition here, if both of them exist, use the `preprocess_cfg` configuration.
```

### training schedule

Mainly contains training strategy settings：

- `optim_wrapper`: Optimizer Wrapper Settings Information
  - `optimizer`: Supports all `pytorch` optimizers, refer to the relevant [MMEngine](TODO:) documentation.
  - `paramwise_cfg`: To customize different optimization parameters, refer to the relevant [Learning Policy Documentation](TODO:) document.
- `param_scheduler`: Learning rate policy, supports "CosineAnnealing", "Step", "Cyclic", etc.， [MMEngine](TODO:)
- `train_cfg | val_cfg`: For the configuration of the runner when training and validation, refer to the relevant [MMEngine](TODO:) documentation.

Following is the schedule configuration of ResNet50 primitive config in ['configs/_base_/datasets/imagenet_bs32.py'](https://github.com/open-mmlab/mmclassification/blob/master/configs/_base_/datasets/imagenet_bs32.py)：

```python
# Optimizer configuration, supports all PyTorch optimizers
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001))

# The tuning strategy of the learning rate parameter
# Scheduler policy, also supports CosineAnnealing, Cyclic, etc.,
# 'MultiStepLR' step in 30, 60, 90 epochs, lr = lr * gamma
param_scheduler = dict(
    type='MultiStepLR', by_epoch=True, milestones=[30, 60, 90], gamma=0.1)

# Training configuration, iterate 100 epochs, and perform validation set evaluation after each training epoch
# 'by_epoch=True' uses EpochBaseLoop by default, 'by_epoch=False' uses IterBaseLoop by default
# Refer to MMEngine for more information on Runners and Loops
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# When using automatic learning rate adjustment,
# the batch_size of the benchmark is equal to base_num_GPU * base_batch_pre_GPU
auto_scale_lr = dict(base_batch_size=256)
```

### runtime setting

This part mainly includes saving the checkpoint strategy, log configuration, training parameters, breakpoint weight path, working directory, etc..

Here is the running configuration ['configs/_base_/default_runtime.py'](https://github.com/open-mmlab/mmclassification/blob/master/configs/_base_/default_runtime.py) file used by almost all algorithms:

```python
# defaults to use registries in mmcls
default_scope = 'mmcls'

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type='IterTimerHook'),

    # print log every 100 iterations.
    logger=dict(type='LoggerHook', interval=100),

    # enable the parameter scheduler.
    param_scheduler=dict(type='ParamSchedulerHook'),

    # save checkpoint per epoch.
    checkpoint=dict(type='CheckpointHook', interval=1),

    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type='DistSamplerSeedHook'),

    # validation results visualization, set True to enable it.
    visualization=dict(type='VisualizationHook', enable=False),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,

    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),

    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

# set visualizer
vis_backends = [dict(type='LocalVisBackend')]  # use local HDD backend
visualizer = dict(
    type='ClsVisualizer', vis_backends=vis_backends, name='visualizer')

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False
```

## Inherit and Modify Config File

For easy understanding, we recommend contributors to inherit from existing methods.

For all configs under the same folder, it is recommended to have only **one** _primitive_ config. All other configs should inherit from the _primitive_ config. In this way, the maximum of inheritance level is 3.

For example, if your config file is based on ResNet with some other modification, you can first inherit the basic ResNet structure, dataset and other training setting by specifying `_base_ ='./resnet50_8xb32_in1k.py'` (The path relative to your config file), and then modify the necessary parameters in the config file. A more specific example, now we want to use almost all configs in `configs/resnet/resnet50_8xb32_in1k.py`, but using `CutMix` train batch augment and changing the number of training epochs from 100 to 300, modify when to decay the learning rate, and modify the dataset path, you can create a new config file `configs/resnet/resnet50_8xb32-300e_in1k.py` with content as below:

```python
_base_ = './resnet50_8xb32_in1k.py'

# using CutMix batch augment
model = dict(
    train_cfg=dict(
        augments=dict(type='CutMix', alpha=1.0, num_classes=1000, prob=1.0)
    )
)

# trains more epochs
train_cfg = dict(max_epochs=300, val_interval=10)  # Train for 300 epochs, evaluate every 10 epochs
param_scheduler = dict(step=[150, 200, 250])   # The learning rate adjustment has also changed

# Use your own dataset directory
train_dataloader = dict(
    dataset=dict(data_root='mydata/imagenet/train'),
)
val_dataloader = dict(
    batch_size=64,                  # No backpropagation during inference, larger batchsize can be used
    dataset=dict(data_root='mydata/imagenet/val'),
)
test_dataloader = dict(
    batch_size=64,                  # No backpropagation during inference, larger batchsize can be used
    dataset=dict(data_root='mydata/imagenet/val'),
)
```

### Use intermediate variables in configs

Some intermediate variables are used in the configuration file. The intermediate variables make the configuration file clearer and easier to modify.

For example, `train_pipeline` / `test_pipeline` is the intermediate variable of the data pipeline. We first need to define `train_pipeline` / `test_pipeline`, and then pass them to `train_dataloader` / `test_dataloader`. If you want to modify the size of the input image during training and testing, you need to modify the intermediate variables of `train_pipeline` / `test_pipeline`.

```python
bgr_mean = [103.53, 116.28, 123.675]  # mean in BGR order
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=224, backend='pillow', interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='RandAugment',
        policies='timm_increasing',
        num_policies=2,
        total_level=10,
        magnitude_level=6,
        magnitude_std=0.5,
        hparams=dict(pad_val=[round(x) for x in bgr_mean], interpolation='bicubic')),
    dict(type='PackClsInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=236, edge='short', backend='pillow', interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackClsInputs')
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=val_pipeline))
test_dataloader = dict(dataset=dict(pipeline=val_pipeline))
```

### Ignore some fields in the base configs

Sometimes, you need to set `_delete_=True` to ignore some domain content in the basic configuration file. You can refer to [mmcv](TODO:) for more instructions.

The following is an example. If you wangt to use cosine schedule in the above ResNet50 case, just using inheritance and directly modify it will report `get unexcepected keyword'step'` error, because the `'step'` field of the basic config in `param_scheduler` domain information is reserved, and you need to add `_delete_ =True` to ignore the content of `param_scheduler` related fields in the basic configuration file:

```python
_base_ = '../../configs/resnet/resnet50_8xb32_in1k.py'

# the learning rate scheduler
param_scheduler = [
    # In the first phase performs a warm up learning rate adjustment.
    # The first stage begin is 0 and end is 5, which means [0, 5)
    dict(
        type='LinearLR',      # warm up learning rate policy type
        start_factor=0.25,    # Initial learning rate = lr * start_factor
        by_epoch=True,        # begin and end represent epoch, if False, iter
        begin=0,              # start epoch sequence index
        end=5,                # End epoch sequence index, epoch 5 no longer use this strategy
        convert_to_iter_based=True), # Whether to update based on iter
    # The second stage performs cos learning rate adjustment.
    # In the second stage, begin is 5 and end is 100, which means [5, 100)
    dict(
        type='CosineAnnealingLR', # Use CosineAnnealingLR, half-cosine function
        T_max=95,                 # The period is 95.
        by_epoch=True,            # T_max, begin and end represent epoch, and if False, use IterBase
        begin=5,
        end=100,
    )
]
```

### Use some fields in the base configs

Sometimes, you may refer to some fields in the `_base_` config, so as to avoid duplication of definitions. You can refer to [MMEngine](TODO:) for some more instructions.

The following is an example of using auto augment in the training data preprocessing pipeline， refer to [`configs/resnest/resnest50_32xb64_in1k.py`](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnest/resnest50_32xb64_in1k.py). When defining `train_pipeline`, just add the definition file name of auto augment to `_base_`, and then use `{{_base_.auto_increasing_policies}}` to reference the variables:

```python
_base_ = [
    '../_base_/models/resnest50.py', '../_base_/datasets/imagenet_bs64.py',
    '../_base_/default_runtime.py', './_randaug_policies.py',
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandAugment',
        policies={{_base_.policies}}, # This uses the `policies` parameter from _base_.
        num_policies=2,
        magnitude_level=12),
    dict(type='EfficientNetRandomCrop', scale=224, backend='pillow'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='ColorJitter', brightness=0.4, contrast=0.4, saturation=0.4),
    dict(
        type='Lighting',
        eigval=EIGVAL,
        eigvec=EIGVEC,
        alphastd=0.1,
        to_rgb=False),
    dict(type='PackClsInputs'),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
```

## Modify config in command

When users use the script "tools/train.py" or "tools/test.py" to submit tasks or use some other tools, they can directly modify the content of the configuration file used by specifying the `--cfg-options` parameter.

- Update config keys of dict chains.

  The config options can be specified following the order of the dict keys in the original config.
  For example, `--cfg-options model.backbone.norm_eval=False` changes the all BN modules in model backbones to `train` mode.

- Update keys inside a list of configs.

  Some config dicts are composed as a list in your config. For example, the training pipeline `data.train.pipeline` is normally a list
  e.g. `[dict(type='LoadImageFromFile'), dict(type='TopDownRandomFlip', flip_prob=0.5), ...]`. If you want to change `'flip_prob=0.5'` to `'flip_prob=0.0'` in the pipeline,
  you may specify `--cfg-options data.train.pipeline.1.flip_prob=0.0`.

- Update values of list/tuples.

  If the value to be updated is a list or a tuple. For example, the config file normally sets `val_evaluator = dict(type='Accuracy', topk=(1, 5))`. If you want to change the field `topk`, you may specify `--cfg-options val_evaluator.topk="(1,3)"`. Note that the quotation mark " is necessary to support list/tuple data types, and that **NO** white space is allowed inside the quotation marks in the specified value.

## Import user-defined modules

After studying the follow-up tutorials [ADDING NEW DATASET](https://mmclassification.readthedocs.io/en/latest/tutorials/new_dataset.html), [CUSTOM DATA PIPELINES](https://mmclassification.readthedocs.io/en/latest/tutorials/data_pipeline.html), [ADDING NEW MODULES](https://mmclassification.readthedocs.io/en/latest/tutorials/new_modules.html). You may use MMClassification to complete your project and create new classes of datasets, models, data enhancements, etc. in the project. In order to streamline the code, you can use MMClassification as a third-party library, you just need to keep your own extra code and import your own custom module in the configuration files. For examples, you may refer to [OpenMMLab Algorithm Competition Project](https://github.com/zhangrui-wolf/openmmlab-competition-2021).

Add the following code to your own configuration files:

```python
custom_imports = dict(
    imports=['your_dataset_class',
             'your_transforme_class',
             'your_model_class',
             'your_module_class'],
    allow_failed_imports=False)
```

## FAQ

- None
