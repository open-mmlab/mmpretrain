# Tutorial 1: Learn about Configs

MMClassification mainly uses python files as configs. The design of our configuration file system integrates modularity and inheritance, facilitating users to conduct various experiments. All configuration files are placed in the `configs` folder, which mainly contains the primitive configuration folder of `_base_` and many algorithm folders such as `resnet`, `swin_transformer`, `vision_transformer`, etc.

If you wish to inspect the config file, you may run `python tools/misc/print_config.py /PATH/TO/CONFIG` to see the complete config.

<!-- TOC -->

- [Config  File and Checkpoint Naming Convention](#config-file-and-checkpoint-naming-convention)
- [Config File Structure](#config-file-structure)
- [Inherit and Modify Config File](#inherit-and-modify-config-file)
  - [Use intermediate variables in configs](#use-intermediate-variables-in-configs)
  - [Ignore some fields in the base configs](#ignore-some-fields-in-the-base-configs)
  - [Use some fields in the base configs](#use-some-fields-in-the-base-configs)
- [Modify config through script arguments](#modify-config-through-script-arguments)
- [Import user-defined modules](#import-user-defined-modules)
- [FAQ](#faq)

<!-- TOC -->

## Config File and Checkpoint Naming Convention

We follow the below convention to name config files. Contributors are advised to follow the same style. The config file names are divided into four parts: algorithm info, module information, training information and data information. Logically, different parts are concatenated by underscores `'_'`, and words in the same part are concatenated by dashes `'-'`.

```
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

```
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

```
{config_name}_{date}-{hash}.pth
```

## Config File Structure

There are four kinds of basic component file in the `configs/_base_` folders, namely：

- [models](https://github.com/open-mmlab/mmclassification/tree/master/configs/_base_/models)
- [datasets](https://github.com/open-mmlab/mmclassification/tree/master/configs/_base_/datasets)
- [schedules](https://github.com/open-mmlab/mmclassification/tree/master/configs/_base_/schedules)
- [runtime](https://github.com/open-mmlab/mmclassification/blob/master/configs/_base_/default_runtime.py)

You can easily build your own training config file by inherit some base config files. And the configs that are composed by components from `_base_` are called _primitive_.

For easy understanding, we use [ResNet50 primitive config](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet50_8xb32_in1k.py) as a example and comment the meaning of each line. For more detaile, please refer to the API documentation.

```python
_base_ = [
    '../_base_/models/resnet50.py',           # model
    '../_base_/datasets/imagenet_bs32.py',    # data
    '../_base_/schedules/imagenet_bs256.py',  # training schedule
    '../_base_/default_runtime.py'            # runtime setting
]
```

The four parts are explained separately below, and the above-mentioned ResNet50 primitive config are also used as an example.

### model

The parameter `"model"` is a python dictionary in the configuration file, which mainly includes information such as network structure and loss function:

- `type` ： Classifier name, MMCls supports `ImageClassifier`, refer to [API documentation](https://mmclassification.readthedocs.io/en/latest/api/models.html#classifier).
- `backbone` ： Backbone configs, refer to [API documentation](https://mmclassification.readthedocs.io/en/latest/api/models.html#backbones) for available options.
- `neck` ：Neck network name, MMCls supports `GlobalAveragePooling`, please refer to [API documentation](https://mmclassification.readthedocs.io/en/latest/api/models.html#necks).
- `head`: Head network name, MMCls supports single-label and multi-label classification head networks, available options refer to [API documentation](https://mmclassification.readthedocs.io/en/latest/api/models.html#heads).
  - `loss`: Loss function type, supports `CrossEntropyLoss`, [`LabelSmoothLoss`](https://github.com/open-mmlab/mmclassification/blob/master/configs/_base_/models/resnet50_label_smooth.py) etc., For available options, refer to [API documentation](https://mmclassification.readthedocs.io/en/latest/api/models.html#losses).
- `train_cfg` ：Training augment config, MMCls supports [`mixup`](https://github.com/open-mmlab/mmclassification/blob/master/configs/_base_/models/resnet50_mixup.py), [`cutmix`](https://github.com/open-mmlab/mmclassification/blob/master/configs/_base_/models/resnet50_cutmix.py) and other augments.

```{note}
The 'type' in the configuration file is not a constructed parameter, but a class name.
```

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

The parameter `"data"` is a python dictionary in the configuration file, which mainly includes information to construct dataloader:

- `samples_per_gpu` : the BatchSize of each GPU when building the dataloader
- `workers_per_gpu` : the number of threads per GPU when building dataloader
- `train ｜ val ｜ test` : config to construct dataset
  - `type`: Dataset name, MMCls supports `ImageNet`, `Cifar` etc., refer to [API documentation](https://mmclassification.readthedocs.io/en/latest/api/datasets.html)
  - `data_prefix` : Dataset root directory
  - `pipeline` :  Data processing pipeline, refer to related tutorial [CUSTOM DATA PIPELINES](https://mmclassification.readthedocs.io/en/latest/tutorials/data_pipeline.html)

The parameter `evaluation` is also a dictionary, which is the configuration information of `evaluation hook`, mainly including evaluation interval, evaluation index, etc..

```python
# dataset settings
dataset_type = 'ImageNet'  # dataset name，
img_norm_cfg = dict(        # Image normalization config to normalize the input images
    mean=[123.675, 116.28, 103.53],  # Mean values used to pre-training the pre-trained backbone models
    std=[58.395, 57.12, 57.375],     # Standard variance used to pre-training the pre-trained backbone models
    to_rgb=True)                     # Whether to invert the color channel, rgb2bgr or bgr2rgb.
# train data pipeline
train_pipeline = [
    dict(type='LoadImageFromFile'),                # First pipeline to load images from file path
    dict(type='RandomResizedCrop', size=224),      # RandomResizedCrop
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),  # Randomly flip the picture horizontally with a probability of 0.5
    dict(type='Normalize', **img_norm_cfg),        # normalization
    dict(type='ImageToTensor', keys=['img']),      # convert image from numpy into torch.Tensor
    dict(type='ToTensor', keys=['gt_label']),      # convert gt_label into torch.Tensor
    dict(type='Collect', keys=['img', 'gt_label']) # Pipeline that decides which keys in the data should be passed to the detector
]
# test data pipeline
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, -1)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])             # do not pass gt_label while testing
]
data = dict(
    samples_per_gpu=32,     # Batch size of a single GPU
    workers_per_gpu=2,      # Worker to pre-fetch data for each single GPU
    train=dict(  # Train dataset config
    train=dict(            # train data config
        type=dataset_type,                  # dataset name
        data_prefix='data/imagenet/train',  # Dataset root, when ann_file does not exist, the category information is automatically obtained from the root folder
        pipeline=train_pipeline),           # train data pipeline
    val=dict(              # val data config
        type=dataset_type,
        data_prefix='data/imagenet/val',
        ann_file='data/imagenet/meta/val.txt',   #  ann_file existes, the category information is obtained from file
        pipeline=test_pipeline),
    test=dict(             # test data config
        type=dataset_type,
        data_prefix='data/imagenet/val',
        ann_file='data/imagenet/meta/val.txt',
        pipeline=test_pipeline))
evaluation = dict(       # The config to build the evaluation hook, refer to https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/evaluation/eval_hooks.py#L7 for more details.
    interval=1,          # Evaluation interval
    metric='accuracy')   # Metrics used during evaluation
```

### training schedule

Mainly include optimizer settings, `optimizer hook` settings, learning rate schedule and `runner` settings:

- `optimizer`: optimizer setting , support all optimizers in `pytorch`, refer to related [mmcv](https://mmcv.readthedocs.io/en/latest/_modules/mmcv/runner/optimizer/default_constructor.html#DefaultOptimizerConstructor) documentation.
- `optimizer_config`: `optimizer hook` configuration file, such as setting gradient limit, refer to related [mmcv](https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/optimizer.py#L8) code.
- `lr_config`: Learning rate scheduler, supports "CosineAnnealing", "Step", "Cyclic", etc. refer to related [mmcv](https://mmcv.readthedocs.io/en/latest/_modules/mmcv/runner/hooks/lr_updater.html#LrUpdaterHook) documentation for more options.
- `runner`: For `runner`, please refer to `mmcv` for [`runner`](https://mmcv.readthedocs.io/en/latest/understand_mmcv/runner.html) introduction document.

```python
# he configuration file used to build the optimizer, support all optimizers in PyTorch.
optimizer = dict(type='SGD',         # Optimizer type
                lr=0.1,              # Learning rate of optimizers, see detail usages of the parameters in the documentation of PyTorch
                momentum=0.9,        # Momentum
                weight_decay=0.0001) # Weight decay of SGD
# Config used to build the optimizer hook, refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/optimizer.py#L8 for implementation details.
optimizer_config = dict(grad_clip=None)  # Most of the methods do not use gradient clip
# Learning rate scheduler config used to register LrUpdater hook
lr_config = dict(policy='step',          # The policy of scheduler, also support CosineAnnealing, Cyclic, etc. Refer to details of supported LrUpdater from https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py#L9.
                 step=[30, 60, 90])      # Steps to decay the learning rate
runner = dict(type='EpochBasedRunner',   # Type of runner to use (i.e. IterBasedRunner or EpochBasedRunner)
            max_epochs=100)    # Runner that runs the workflow in total max_epochs. For IterBasedRunner use `max_iters`
```

### runtime setting

This part mainly includes saving the checkpoint strategy, log configuration, training parameters, breakpoint weight path, working directory, etc..

```python
# Config to set the checkpoint hook, Refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py for implementation.
checkpoint_config = dict(interval=1)    # The save interval is 1
# config to register logger hook
log_config = dict(
    interval=100,                       # Interval to print the log
    hooks=[
        dict(type='TextLoggerHook'),           # The Tensorboard logger is also supported
        # dict(type='TensorboardLoggerHook')
    ])

dist_params = dict(backend='nccl')   # Parameters to setup distributed training, the port can also be set.
log_level = 'INFO'             # The output level of the log.
resume_from = None             # Resume checkpoints from a given path, the training will be resumed from the epoch when the checkpoint's is saved.
workflow = [('train', 1)]      # Workflow for runner. [('train', 1)] means there is only one workflow and the workflow named 'train' is executed once.
work_dir = 'work_dir'          # Directory to save the model checkpoints and logs for the current experiments.
```

## Inherit and Modify Config File

For easy understanding, we recommend contributors to inherit from existing methods.

For all configs under the same folder, it is recommended to have only **one** _primitive_ config. All other configs should inherit from the _primitive_ config. In this way, the maximum of inheritance level is 3.

For example, if your config file is based on ResNet with some other modification, you can first inherit the basic ResNet structure, dataset and other training setting by specifying `_base_ ='./resnet50_8xb32_in1k.py'` (The path relative to your config file), and then modify the necessary parameters in the config file. A more specific example, now we want to use almost all configs in `configs/resnet/resnet50_8xb32_in1k.py`, but change the number of training epochs from 100 to 300, modify when to decay the learning rate, and modify the dataset path, you can create a new config file `configs/resnet/resnet50_8xb32-300e_in1k.py` with content as below:

```python
_base_ = './resnet50_8xb32_in1k.py'

runner = dict(max_epochs=300)
lr_config = dict(step=[150, 200, 250])

data = dict(
    train=dict(data_prefix='mydata/imagenet/train'),
    val=dict(data_prefix='mydata/imagenet/train', ),
    test=dict(data_prefix='mydata/imagenet/train', )
)
```

### Use intermediate variables in configs

Some intermediate variables are used in the configuration file. The intermediate variables make the configuration file clearer and easier to modify.

For example, `train_pipeline` / `test_pipeline` is the intermediate variable of the data pipeline. We first need to define `train_pipeline` / `test_pipeline`, and then pass them to `data`. If you want to modify the size of the input image during training and testing, you need to modify the intermediate variables of `train_pipeline` / `test_pipeline`.

```python
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=384, backend='pillow',),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=384, backend='pillow'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
```

### Ignore some fields in the base configs

Sometimes, you need to set `_delete_=True` to ignore some domain content in the basic configuration file. You can refer to [mmcv](https://mmcv.readthedocs.io/en/latest/understand_mmcv/config.html#inherit-from-base-config-with-ignored-fields) for more instructions.

The following is an example. If you want to use cosine schedule in the above ResNet50 case, just using inheritance and directly modify it will report `get unexcepected keyword'step'` error, because the `'step'` field of the basic config in `lr_config` domain information is reserved, and you need to add `_delete_ =True` to ignore the content of `lr_config` related fields in the basic configuration file:

```python
_base_ = '../../configs/resnet/resnet50_8xb32_in1k.py'

lr_config = dict(
    _delete_=True,
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    by_epoch=True,
    warmup_iters=5,
    warmup_ratio=0.1
)
```

### Use some fields in the base configs

Sometimes, you may refer to some fields in the `_base_` config, so as to avoid duplication of definitions. You can refer to [mmcv](https://mmcv.readthedocs.io/en/latest/understand_mmcv/config.html#reference-variables-from-base) for some more instructions.

The following is an example of using auto augment in the training data preprocessing pipeline， refer to [`configs/_base_/datasets/imagenet_bs64_autoaug.py`](https://github.com/open-mmlab/mmclassification/blob/master/configs/_base_/datasets/imagenet_bs64_autoaug.py). When defining `train_pipeline`, just add the definition file name of auto augment to `_base_`, and then use `{{_base_.auto_increasing_policies}}` to reference the variables:

```python
_base_ = ['./pipelines/auto_aug.py']

# dataset settings
dataset_type = 'ImageNet'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='AutoAugment', policies={{_base_.auto_increasing_policies}}),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [...]
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=2,
    train=dict(..., pipeline=train_pipeline),
    val=dict(..., pipeline=test_pipeline))
evaluation = dict(interval=1, metric='accuracy')
```

## Modify config through script arguments

When users use the script "tools/train.py" or "tools/test.py" to submit tasks or use some other tools, they can directly modify the content of the configuration file used by specifying the `--cfg-options` parameter.

- Update config keys of dict chains.

  The config options can be specified following the order of the dict keys in the original config.
  For example, `--cfg-options model.backbone.norm_eval=False` changes the all BN modules in model backbones to `train` mode.

- Update keys inside a list of configs.

  Some config dicts are composed as a list in your config. For example, the training pipeline `data.train.pipeline` is normally a list
  e.g. `[dict(type='LoadImageFromFile'), dict(type='TopDownRandomFlip', flip_prob=0.5), ...]`. If you want to change `'flip_prob=0.5'` to `'flip_prob=0.0'` in the pipeline,
  you may specify `--cfg-options data.train.pipeline.1.flip_prob=0.0`.

- Update values of list/tuples.

  If the value to be updated is a list or a tuple. For example, the config file normally sets `workflow=[('train', 1)]`. If you want to
  change this key, you may specify `--cfg-options workflow="[(train,1),(val,1)]"`. Note that the quotation mark " is necessary to
  support list/tuple data types, and that **NO** white space is allowed inside the quotation marks in the specified value.

## Import user-defined modules

```{note}
This part may only be used when using MMClassification as a third party library to build your own project, and beginners can skip it.
```

After studying the follow-up tutorials [ADDING NEW DATASET](https://mmclassification.readthedocs.io/en/latest/tutorials/new_dataset.html), [CUSTOM DATA PIPELINES](https://mmclassification.readthedocs.io/en/latest/tutorials/data_pipeline.html), [ADDING NEW MODULES](https://mmclassification.readthedocs.io/en/latest/tutorials/new_modules.html). You may use MMClassification to complete your project and create new classes of datasets, models, data enhancements, etc. in the project. In order to streamline the code, you can use MMClassification as a third-party library, you just need to keep your own extra code and import your own custom module in the configuration files. For examples, you may refer to [OpenMMLab Algorithm Competition Project](https://github.com/zhangrui-wolf/openmmlab-competition-2021) .

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
