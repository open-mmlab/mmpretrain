# Learn about Configs

To manage various configurations in a deep-learning experiment, we use a kind of config file to record all of
these configurations. This config system has a modular and inheritance design, and more details can be found in
{external+mmengine:doc}`the tutorial in MMEngine <advanced_tutorials/config>`.

Usually, we use python files as config file. All configuration files are placed under the [`configs`](https://github.com/open-mmlab/mmpretrain/tree/main/configs) folder, and the directory structure is as follows:

```text
MMPretrain/
    ├── configs/
    │   ├── _base_/                       # primitive configuration folder
    │   │   ├── datasets/                      # primitive datasets
    │   │   ├── models/                        # primitive models
    │   │   ├── schedules/                     # primitive schedules
    │   │   └── default_runtime.py             # primitive runtime setting
    │   ├── beit/                         # BEiT Algorithms Folder
    │   ├── mae/                          # MAE Algorithms Folder
    │   ├── mocov2/                       # MoCoV2 Algorithms Folder
    │   ├── resnet/                       # ResNet Algorithms Folder
    │   ├── swin_transformer/             # Swin Algorithms Folder
    │   ├── vision_transformer/           # ViT Algorithms Folder
    │   ├── ...
    └── ...
```

If you wish to inspect the config file, you may run `python tools/misc/print_config.py /PATH/TO/CONFIG` to see the complete config.

This article mainly explains the structure of configuration files, and how to modify it based on the existing configuration files. We will take [ResNet50 config file](https://github.com/open-mmlab/mmpretrain/blob/main/configs/resnet/resnet50_8xb32_in1k.py) as an example and explain it line by line.

## Config Structure

There are four kinds of basic component files in the `configs/_base_` folders, namely：

- [models](https://github.com/open-mmlab/mmpretrain/tree/main/configs/_base_/models)
- [datasets](https://github.com/open-mmlab/mmpretrain/tree/main/configs/_base_/datasets)
- [schedules](https://github.com/open-mmlab/mmpretrain/tree/main/configs/_base_/schedules)
- [runtime](https://github.com/open-mmlab/mmpretrain/blob/main/configs/_base_/default_runtime.py)

We call all the config files in the `_base_` folder as _primitive_ config files. You can easily build your training config file by inheriting some primitive config files.

For easy understanding, we use [ResNet50 config file](https://github.com/open-mmlab/mmpretrain/blob/main/configs/resnet/resnet50_8xb32_in1k.py) as an example and comment on each line.

```python
_base_ = [                                    # This config file will inherit all config files in `_base_`.
    '../_base_/models/resnet50.py',           # model settings
    '../_base_/datasets/imagenet_bs32.py',    # data settings
    '../_base_/schedules/imagenet_bs256.py',  # schedule settings
    '../_base_/default_runtime.py'            # runtime settings
]
```

We will explain the four primitive config files separately below.

### Model settings

This primitive config file includes a dict variable `model`, which mainly includes information such as network structure and loss function:

- `type`: The type of model to build, we support several tasks.
  - For image classification tasks, it's usually `ImageClassifier` You can find more details in the [API documentation](mmpretrain.models.classifiers).
  - For self-supervised leanrning, there are several `SelfSupervisors`, such as `MoCoV2`, `BEiT`, `MAE`, etc. You can find more details in the [API documentation](mmpretrain.models.selfsup).
  - For image retrieval tasks, it's usually `ImageToImageRetriever` You can find more details in the [API documentation](mmpretrain.models.retrievers).

Usually, we use the **`type` field** to specify the class of the component and use other fields to pass
the initialization arguments of the class. The {external+mmengine:doc}`registry tutorial <advanced_tutorials/registry>` describes it in detail.

Here, we use the config fields of [`ImageClassifier`](mmpretrain.models.classifiers.ImageClassifier) as an example to
describe the initialization arguments as below:

- `backbone`: The settings of the backbone. The backbone is the main network to extract features of the inputs, like `ResNet`, `Swin Transformer`, `Vision Transformer` etc. All available backbones can be found in the [API documentation](mmpretrain.models.backbones).
  - For self-supervised leanrning, some of the backbones are re-implemented, you can find more details in the [API documentation](mmpretrain.models.selfsup).
- `neck`: The settings of the neck. The neck is the intermediate module to connect the backbone and the head, like `GlobalAveragePooling`. All available necks can be found in the [API documentation](mmpretrain.models.necks).
- `head`: The settings of the task head. The head is the task-related component to do a specified task, like image classification or self-supervised training. All available heads can be found in the [API documentation](mmpretrain.models.heads).
  - `loss`: The loss function to optimize, like `CrossEntropyLoss`, `LabelSmoothLoss`, `PixelReconstructionLoss` and etc. All available losses can be found in the [API documentation](mmpretrain.models.losses).
- `data_preprocessor`: The component before the model forwarding to preprocess the inputs. See the [documentation](mmpretrain.models.utils.data_preprocessor) for more details.
- `train_cfg`: The extra settings of `ImageClassifier` during training. In `ImageClassifier`, we mainly use it to specify batch augmentation settings, like `Mixup` and `CutMix`. See the [documentation](mmpretrain.models.utils.batch_augments) for more details.

Following is the model primitive config of the ResNet50 config file in [`configs/_base_/models/resnet50.py`](https://github.com/open-mmlab/mmpretrain/blob/main/configs/_base_/models/resnet50.py):

```python
model = dict(
    type='ImageClassifier',     # The type of the main model (here is for image classification task).
    backbone=dict(
        type='ResNet',          # The type of the backbone module.
        # All fields except `type` come from the __init__ method of class `ResNet`
        # and you can find them from https://mmpretrain.readthedocs.io/en/latest/api/generated/mmpretrain.models.backbones.ResNet.html
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        frozen_stages=-1,
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),    # The type of the neck module.
    head=dict(
        type='LinearClsHead',     # The type of the classification head module.
        # All fields except `type` come from the __init__ method of class `LinearClsHead`
        # and you can find them from https://mmpretrain.readthedocs.io/en/latest/api/generated/mmpretrain.models.heads.LinearClsHead.html
        num_classes=1000,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))
```

### Data settings

This primitive config file includes information to construct the dataloader and evaluator:

- `data_preprocessor`: Model input preprocessing configuration, same as `model.data_preprocessor` but with lower priority.
- `train_evaluator | val_evaluator | test_evaluator`: To build the evaluator or metrics, refer to the [tutorial](mmpretrain.evaluation).
- `train_dataloader | val_dataloader | test_dataloader`: The settings of dataloaders
  - `batch_size`: The batch size of each GPU.
  - `num_workers`: The number of workers to fetch data of each GPU.
  - `sampler`: The settings of the sampler.
  - `persistent_workers`: Whether to persistent workers after finishing one epoch.
  - `dataset`: The settings of the dataset.
    - `type`: The type of the dataset, we support `CustomDataset`, `ImageNet` and many other datasets, refer to [documentation](mmpretrain.datasets).
    - `pipeline`: The data transform pipeline. You can find how to design a pipeline in [this tutorial](https://mmpretrain.readthedocs.io/en/latest/tutorials/data_pipeline.html).

Following is the data primitive config of the ResNet50 config in [`configs/_base_/datasets/imagenet_bs32.py`](https://github.com/open-mmlab/mmpretrain/blob/main/configs/_base_/datasets/imagenet_bs32.py)：

```python
dataset_type = 'ImageNet'
# preprocessing configuration
data_preprocessor = dict(
    # Input image data channels in 'RGB' order
    mean=[123.675, 116.28, 103.53],    # Input image normalized channel mean in RGB order
    std=[58.395, 57.12, 57.375],       # Input image normalized channel std in RGB order
    to_rgb=True,                       # Whether to flip the channel from BGR to RGB or RGB to BGR
)

train_pipeline = [
    dict(type='LoadImageFromFile'),     # read image
    dict(type='RandomResizedCrop', scale=224),     # Random scaling and cropping
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),   # random horizontal flip
    dict(type='PackInputs'),         # prepare images and labels
]

test_pipeline = [
    dict(type='LoadImageFromFile'),     # read image
    dict(type='ResizeEdge', scale=256, edge='short'),  # Scale the short side to 256
    dict(type='CenterCrop', crop_size=224),     # center crop
    dict(type='PackInputs'),                 # prepare images and labels
]

# Construct training set dataloader
train_dataloader = dict(
    batch_size=32,                     # batchsize per GPU
    num_workers=5,                     # Number of workers to fetch data per GPU
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
# The settings of the evaluation metrics for validation. We use the top1 and top5 accuracy here.
val_evaluator = dict(type='Accuracy', topk=(1, 5))

test_dataloader = val_dataloader  # The settings of the dataloader for the test dataset, which is the same as val_dataloader
test_evaluator = val_evaluator    # The settings of the evaluation metrics for test, which is the same as val_evaluator
```

```{note}
The data preprocessor can be defined either in the subfield of `model`, or a using the `data_preprocessor` definition here, if both of them exist, use the `model.data_preprocessor` configuration.
```

### Schedule settings

This primitive config file mainly contains training strategy settings and the settings of training, val and
test loops:

- `optim_wrapper`: The settings of the optimizer wrapper. We use the optimizer wrapper to customize the
  optimization process.
  - `optimizer`: Supports all `pytorch` optimizers, refers to the relevant {external+mmengine:doc}`MMEngine documentation <tutorials/optim_wrapper>`.
  - `paramwise_cfg`: To set different optimization arguments according to the parameters' type or name, refer to the relevant [learning policy documentation](../advanced_guides/schedule.md).
  - `accumulative_counts`: Optimize parameters after several backward steps instead of one backward step. You
    can use it to simulate large batch size by small batch size.
- `param_scheduler`: Optimizer parameters policy. You can use it to specify learning rate and momentum curves during training. See the {external+mmengine:doc}`documentation <tutorials/param_scheduler>` in MMEngine for more details.
- `train_cfg | val_cfg | test_cfg`: The settings of the training, validation and test loops, refer to the relevant {external+mmengine:doc}`MMEngine documentation <design/runner>`.

Following is the schedule primitive config of the ResNet50 config in [`configs/_base_/datasets/imagenet_bs32.py`](https://github.com/open-mmlab/mmpretrain/blob/main/configs/_base_/datasets/imagenet_bs32.py)：

```python
optim_wrapper = dict(
    # Use SGD optimizer to optimize parameters.
    optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001))

# The tuning strategy of the learning rate.
# The 'MultiStepLR' means to use multiple steps policy to schedule the learning rate (LR).
param_scheduler = dict(
    type='MultiStepLR', by_epoch=True, milestones=[30, 60, 90], gamma=0.1)

# Training configuration, iterate 100 epochs, and perform validation after every training epoch.
# 'by_epoch=True' means to use `EpochBaseTrainLoop`, 'by_epoch=False' means to use IterBaseTrainLoop.
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
# Use the default val loop settings.
val_cfg = dict()
# Use the default test loop settings.
test_cfg = dict()

# This schedule is for the total batch size 256.
# If you use a different total batch size, like 512 and enable auto learning rate scaling.
# We will scale up the learning rate to 2 times.
auto_scale_lr = dict(base_batch_size=256)
```

### Runtime settings

This part mainly includes saving the checkpoint strategy, log configuration, training parameters, breakpoint weight path, working directory, etc.

Here is the runtime primitive config file ['configs/_base_/default_runtime.py'](https://github.com/open-mmlab/mmpretrain/blob/main/configs/_base_/default_runtime.py) file used by almost all configs:

```python
# defaults to use registries in mmpretrain
default_scope = 'mmpretrain'

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

    # set sampler seed in a distributed environment.
    sampler_seed=dict(type='DistSamplerSeedHook'),

    # validation results visualization, set True to enable it.
    visualization=dict(type='VisualizationHook', enable=False),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,

    # set multi-process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),

    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

# set visualizer
vis_backends = [dict(type='LocalVisBackend')]  # use local HDD backend
visualizer = dict(
    type='UniversalVisualizer', vis_backends=vis_backends, name='visualizer')

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False
```

## Inherit and Modify Config File

For easy understanding, we recommend contributors inherit from existing config files. But do not abuse the
inheritance. Usually, for all config files, we recommend the maximum inheritance level is 3.

For example, if your config file is based on ResNet with some other modification, you can first inherit the
basic ResNet structure, dataset and other training settings by specifying `_base_ ='./resnet50_8xb32_in1k.py'`
(The path relative to your config file), and then modify the necessary parameters in the config file. A more
specific example, now we want to use almost all configs in `configs/resnet/resnet50_8xb32_in1k.py`, but using
`CutMix` train batch augment and changing the number of training epochs from 100 to 300, modify when to decay
the learning rate, and modify the dataset path, you can create a new config file
`configs/resnet/resnet50_8xb32-300e_in1k.py` with content as below:

```python
# create this file under 'configs/resnet/' folder
_base_ = './resnet50_8xb32_in1k.py'

# using CutMix batch augment
model = dict(
    train_cfg=dict(
        augments=dict(type='CutMix', alpha=1.0)
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
    batch_size=64,                  # No back-propagation during validation, larger batch size can be used
    dataset=dict(data_root='mydata/imagenet/val'),
)
test_dataloader = dict(
    batch_size=64,                  # No back-propagation during test, larger batch size can be used
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
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=236, edge='short', backend='pillow', interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs')
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=val_pipeline))
test_dataloader = dict(dataset=dict(pipeline=val_pipeline))
```

### Ignore some fields in the base configs

Sometimes, you need to set `_delete_=True` to ignore some domain content in the basic configuration file. You can refer to the {external+mmengine:doc}`documentation in MMEngine <advanced_tutorials/config>` for more instructions.

The following is an example. If you want to use cosine schedule in the above ResNet50 case, just using inheritance and directly modifying it will report `get unexpected keyword 'step'` error, because the `'step'` field of the basic config in `param_scheduler` domain information is reserved, and you need to add `_delete_ =True` to ignore the content of `param_scheduler` related fields in the basic configuration file:

```python
_base_ = '../../configs/resnet/resnet50_8xb32_in1k.py'

# the learning rate scheduler
param_scheduler = dict(type='CosineAnnealingLR', by_epoch=True, _delete_=True)
```

### Use some fields in the base configs

Sometimes, you may refer to some fields in the `_base_` config, to avoid duplication of definitions. You can refer to {external+mmengine:doc}`MMEngine <advanced_tutorials/config>` for some more instructions.

The following is an example of using auto augment in the training data preprocessing pipeline, refer to [`configs/resnest/resnest50_32xb64_in1k.py`](https://github.com/open-mmlab/mmpretrain/blob/main/configs/resnest/resnest50_32xb64_in1k.py). When defining `train_pipeline`, just add the definition file name of auto augment to `_base_`, and then use `_base_.auto_increasing_policies` to reference the variables in the primitive config:

```python
_base_ = [
    '../_base_/models/resnest50.py', '../_base_/datasets/imagenet_bs64.py',
    '../_base_/default_runtime.py', './_randaug_policies.py',
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandAugment',
        policies=_base_.policies, # This uses the `policies` parameter in the primitive config.
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
    dict(type='PackInputs'),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
```

## Modify config in command

When you use the script "tools/train.py" or "tools/test.py" to submit tasks or use some other tools, they can directly modify the content of the configuration file used by specifying the `--cfg-options` argument.

- Update config keys of dict chains.

  The config options can be specified following the order of the dict keys in the original config.
  For example, `--cfg-options model.backbone.norm_eval=False` changes the all BN modules in model backbones to `train` mode.

- Update keys inside a list of configs.

  Some config dicts are composed as a list in your config. For example, the training pipeline `data.train.pipeline` is normally a list
  e.g. `[dict(type='LoadImageFromFile'), dict(type='TopDownRandomFlip', flip_prob=0.5), ...]`. If you want to change `'flip_prob=0.5'` to `'flip_prob=0.0'` in the pipeline,
  you may specify `--cfg-options data.train.pipeline.1.flip_prob=0.0`.

- Update values of list/tuples.

  If the value to be updated is a list or a tuple. For example, the config file normally sets `val_evaluator = dict(type='Accuracy', topk=(1, 5))`. If you want to change the field `topk`, you may specify `--cfg-options val_evaluator.topk="(1,3)"`. Note that the quotation mark " is necessary to support list/tuple data types and that **NO** white space is allowed inside the quotation marks in the specified value.
