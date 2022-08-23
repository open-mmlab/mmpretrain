# 教程 1：如何编写配置文件

MMClassification 主要使用 python 文件作为配置文件。其配置文件系统的设计将模块化与继承整合进来，方便用户进行各种实验。所有配置文件都放置在 `configs` 文件夹下，主要包含 `_base_` 原始配置文件夹 以及 `resnet`, `swin_transformer`，`vision_transformer` 等诸多算法文件夹。

本文主要讲解 MMClassification 配置文件的命名和结构，以及如何基于已有的配置文件修改，并以 [ResNet50 原始配置文件](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet50_8xb32_in1k.py) 逐行解释。

可以使用 `python tools/misc/print_config.py /PATH/TO/CONFIG` 命令来查看完整的配置信息，从而方便检查所对应的配置文件。

<!-- TOC -->

- [配置文件以及权重命名规则](#配置文件以及权重命名规则)
- [配置文件结构](#配置文件结构)
- [继承并修改配置文件](#继承并修改配置文件)
  - [使用配置文件里的中间变量](#使用配置文件里的中间变量)
  - [忽略基础配置文件里的部分内容](#忽略基础配置文件里的部分内容)
  - [引用基础配置文件里的变量](#引用基础配置文件里的变量)
- [通过命令行参数修改配置信息](#通过命令行参数修改配置信息)
- [导入用户自定义模块](#导入用户自定义模块)
- [常见问题](#常见问题)

<!-- TOC -->

## 配置文件以及权重命名规则

MMClassification 按照以下风格进行配置文件命名，代码库的贡献者需要遵循相同的命名规则。文件名总体分为四部分：算法信息，模块信息，训练信息和数据信息。逻辑上属于不同部分的单词之间用下划线 `'_'` 连接，同一部分有多个单词用短横线 `'-'` 连接。

```text
{algorithm info}_{module info}_{training info}_{data info}.py
```

- `algorithm info`：算法信息，算法名称或者网络架构，如 resnet 等；
- `module info`： 模块信息，因任务而异，用以表示一些特殊的 neck、head 和 pretrain 信息；
- `training info`：一些训练信息，训练策略设置，包括 batch size，schedule 以及数据增强等；
- `data info`：数据信息，数据集名称、模态、输入尺寸等，如 imagenet, cifar 等；

### 算法信息

指论文中的算法名称缩写，以及相应的分支架构信息。例如：

- `resnet50`
- `mobilenet-v3-large`
- `vit-small-patch32`   : `patch32` 表示 `ViT` 切分的分块大小
- `seresnext101-32x4d`  : `SeResNet101` 基本网络结构，`32x4d` 表示在 `Bottleneck` 中  `groups` 和 `width_per_group` 分别为32和4

### 模块信息

指一些特殊的 `neck` 、`head` 或者 `pretrain` 的信息， 在分类中常见为预训练信息，比如：

- `in21k-pre` : 在 `ImageNet21k` 上预训练
- `in21k-pre-3rd-party` : 在 `ImageNet21k` 上预训练，其权重来自其他仓库

### 训练信息

训练策略的一些设置，包括训练类型、 `batch size`、 `lr schedule`、 数据增强以及特殊的损失函数等等,比如:
Batch size 信息：

- 格式为`{gpu x batch_per_gpu}`, 如 `8xb32`

训练类型(主要见于 transformer 网络，如 `ViT` 算法，这类算法通常分为预训练和微调两种模式):

- `ft` : Finetune config，用于微调的配置文件
- `pt` : Pretrain config，用于预训练的配置文件

训练策略信息，训练策略以复现配置文件为基础，此基础不必标注训练策略。但如果在此基础上进行改进，则需注明训练策略，按照应用点位顺序排列，如：`{pipeline aug}-{train aug}-{loss trick}-{scheduler}-{epochs}`

- `coslr-200e` : 使用 cosine scheduler, 训练 200 个 epoch
- `autoaug-mixup-lbs-coslr-50e` : 使用了 `autoaug`、`mixup`、`label smooth`、`cosine scheduler`, 训练了 50 个轮次

### 数据信息

- `in1k` : `ImageNet1k` 数据集，默认使用 `224x224` 大小的图片
- `in21k` : `ImageNet21k` 数据集，有些地方也称为 `ImageNet22k` 数据集，默认使用 `224x224` 大小的图片
- `in1k-384px` : 表示训练的输出图片大小为 `384x384`
- `cifar100`

### 配置文件命名案例

```text
repvgg-D2se_deploy_4xb64-autoaug-lbs-mixup-coslr-200e_in1k.py
```

- `repvgg-D2se`:  算法信息
  - `repvgg`: 主要算法名称。
  - `D2se`: 模型的结构。
- `deploy`:模块信息，该模型为推理状态。
- `4xb64-autoaug-lbs-mixup-coslr-200e`: 训练信息
  - `4xb64`: 使用4块 GPU 并且 每块 GPU 的批大小为64。
  - `autoaug`: 使用 `AutoAugment` 数据增强方法。
  - `lbs`: 使用 `label smoothing` 损失函数。
  - `mixup`: 使用 `mixup` 训练增强方法。
  - `coslr`: 使用 `cosine scheduler` 优化策略。
  - `200e`: 训练 200 轮次。
- `in1k`: 数据信息。 配置文件用于 `ImageNet1k` 数据集上使用 `224x224` 大小图片训练。

### 权重命名规则

权重的命名主要包括配置文件名，日期和哈希值。

```text
{config_name}_{date}-{hash}.pth
```

## 配置文件结构

在 `configs/_base_` 文件夹下有 4 个基本组件类型，分别是：

- [模型(model)](https://github.com/open-mmlab/mmclassification/tree/master/configs/_base_/models)
- [数据(data)](https://github.com/open-mmlab/mmclassification/tree/master/configs/_base_/datasets)
- [训练策略(schedule)](https://github.com/open-mmlab/mmclassification/tree/master/configs/_base_/schedules)
- [运行设置(runtime)](https://github.com/open-mmlab/mmclassification/blob/master/configs/_base_/default_runtime.py)

你可以通过继承一些基本配置文件轻松构建自己的训练配置文件。由来自 `_base_` 的组件组成的配置称为 _primitive_。

为了帮助用户对 MMClassification 检测系统中的完整配置和模块有一个基本的了解，我们使用 [ResNet50 原始配置文件](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet50_8xb32_in1k.py) 作为案例进行说明并注释每一行含义。更详细的用法和各个模块对应的替代方案，请参考 API 文档。

```python
_base_ = [                                    # _base_ 可为一个 list 或者一个 str
    '../_base_/models/resnet50.py',           # 模型
    '../_base_/datasets/imagenet_bs32.py',    # 数据
    '../_base_/schedules/imagenet_bs256.py',  # 训练策略
    '../_base_/default_runtime.py'            # 默认运行设置
]
```

下面对这四个部分分别进行说明，仍然以上述 ResNet50 原始配置文件作为案例。

### 模型

`model` 在配置文件中为一个 `python` 字典，主要包括网络结构、损失函数等信息：

- `type`： 分类器名称, 目前 MMClassification 只支持 `ImageClassifier`， 参考 [API 文档](https://mmclassification.readthedocs.io/zh_CN/latest/api.html#module-mmcls.models.classifiers)， 目前支持的分类算法可查看 [`model zoo`](https://mmclassification.readthedocs.io/en/latest/model_zoo.html)。
- `data_preprocessor` : 图像输入的预处理模块，包括图像数据转换以及归一化等操作，如 `ClsDataPreprocessor`, 参考 [API 文档](https://mmclassification.readthedocs.io/zh_CN/latest/api.html#module-mmcls.models.datapreprocessors)。
- `backbone`： 主干网类型，目前支持 `ResNet`, `Swin Transformer`, `Vision Transformer` 等。可用选项参考 [API 文档](https://mmclassification.readthedocs.io/zh_CN/latest/api.html#module-mmcls.models.backbones)。
- `neck`： 颈网络类型，目前 MMClassification 支持 `GlobalAveragePooling` 等，参考 [API 文档](https://mmclassification.readthedocs.io/zh_CN/latest/api.html#module-mmcls.models.necks)。
- `head`： 头网络类型， 包括单标签分类与多标签分类头网络，可用选项参考 [API 文档](https://mmclassification.readthedocs.io/zh_CN/latest/api.html#module-mmcls.models.heads)。
  - `loss`： 损失函数类型， 支持 `CrossEntropyLoss`, `LabelSmoothLoss` 等，可用选项参考 [API 文档](https://mmclassification.readthedocs.io/zh_CN/latest/api.html#module-mmcls.models.losses)。
- `train_cfg`：训练配置, 目前支持 `Mixup`, `CutMix` 等训练增强, 可用选项参考 [API 文档](https://mmclassification.readthedocs.io/zh_CN/latest/api.html#models.utils.augment.html)。

```{note}
配置文件中的 'type' 不是构造时的参数，而是类名。
```

以下是 ResNet50 基本配置的模型配置['configs/_base_/models/resnet50.py'](https://github.com/open-mmlab/mmclassification/blob/master/configs/_base_/models/resnet50.py)：

```python
model = dict(
    type='ImageClassifier',     # 分类器类型， 目前只有 'ImageClassifier'
    backbone=dict(
        type='ResNet',          # 主干网络类型
        depth=50,               # 主干网网络深度， ResNet 一般有18, 34, 50, 101, 152 可以选择
        num_stages=4,           # 主干网络状态(stages)的数目，这些状态产生的特征图作为后续的 head 的输入。
        out_indices=(3, ),      # 输出的特征图输出索引。
        frozen_stages=-1,       # 网络 fine-tune 时，冻结网络的截止 stage，若num_stages=4，backbone包含stem 与 4 个 stages。frozen_stages为-1时，不冻结网络； 为0时，冻结 stem； 为1时，冻结 stem 和 stage1； 为4时，冻结整个backbone
        style='pytorch'),       # 主干网络的风格，'pytorch' 意思是步长为2的层为 3x3 卷积， 'caffe' 意思是步长为2的层为 1x1 卷积。
    neck=dict(type='GlobalAveragePooling'),    # 颈网络类型
    head=dict(
        type='LinearClsHead',     # 线性分类头，
        num_classes=1000,         # 输出类别数，这与数据集的类别数一致
        in_channels=2048,         # 输入通道数，这与 neck 的输出通道一致
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0), # 损失函数配置信息
        topk=(1, 5),              # 评估指标，Top-k 准确率， 这里为 top1 与 top5 准确率
    ))
```

### 数据

配置中的 `data` 部分包括构建数据加载器和评估的信息，如下：

- `preprocess_cfg`: 模型输入预处理配置, 与 `model.data_preprocessor` 相同，但优先级更高。
- `train_evaluator | val_evaluator | test_evaluator`: 构建评估器，参考 [API 文档](TODO:)。
- `train_dataloader | val_dataloader | test_dataloader`: 构建 dataloader
  - `samples_per_gpu`: 每个 GPU 的 Batchsize
  - `workers_per_gpu`: 每个 GPU 的线程数
  - `sampler`: 采样器配置。
  - `dataset`: 构造数据集。
    - `type`:  数据集类型， MMClassification 支持 `ImageNet`、 `Cifar` 等数据集 ，参考 [API 文档](https://mmclassification.readthedocs.io/zh_CN/latest/api.html#module-mmcls.datasets)
    - `pipeline`:  数据处理流水线，参考相关教程文档 [如何设计数据处理流水线](https://mmclassification.readthedocs.io/zh_CN/latest/tutorials/data_pipeline.html)

以下是 ResNet50 基本配置的数据配置 ['configs/_base_/datasets/imagenet_bs32.py'](https://github.com/open-mmlab/mmclassification/blob/master/configs/_base_/datasets/imagenet_bs32.py)：

```python
dataset_type = 'ImageNet'
# 预处理配置
preprocess_cfg = dict(
    # 输入的图片数据通道以 'RGB' 顺序
    mean=[123.675, 116.28, 103.53],    # 输入图像归一化的 RGB 通道均值
    std=[58.395, 57.12, 57.375],       # 输入图像归一化的 RGB 通道均值
    to_rgb=True,                       # 是否将通道翻转，从 BGR 转为 RGB 或者 RGB 转为 BGR
)

train_pipeline = [
    dict(type='LoadImageFromFile'),     # 读取图像
    dict(type='RandomResizedCrop', scale=224),     # 随机放缩裁剪
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),   # 随机水平翻转
    dict(type='PackClsInputs'),         # 准备图像以及标签
]

test_pipeline = [
    dict(type='LoadImageFromFile'),     # 读取图像
    dict(type='ResizeEdge', scale=256, edge='short'),  # 短边对其256进行放缩
    dict(type='CenterCrop', crop_size=224),     # 中心裁剪
    dict(type='PackClsInputs'),                 # 准备图像以及标签
]

# 构造训练集 dataloader
train_dataloader = dict(
    batch_size=32,                     # 每张GPU的 batchsize
    num_workers=5,                     # 每个GPU的线程数
    dataset=dict(                      # 训练数据集
        type=dataset_type,
        data_root='data/imagenet',
        ann_file='meta/train.txt',
        data_prefix='train',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),   # 默认采样器
    persistent_workers=True,                             # 是否保持进程，可以缩短每个epoch的准备时间
)

# 构造验证集 dataloader
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
# 构建验证集评估器，使用准确率为指标
val_evaluator = dict(type='Accuracy', topk=(1, 5))

test_dataloader = val_dataloader  # 构造验证集 dataloader，这里直接与 val_dataloader相同
test_evaluator = val_evaluator    # 构造验证集评估器，这里直接与 val_evaluator 相同
```

```note
'model.data_preprocessor' 既可以在 `model=dict(data_preprocessor=dict())`中定义，也可以使用此处的 `preprocess_cfg` 定义, 同时配置时，使用 `preprocess_cfg` 的配置。
```

### 训练策略

主要包含训练策略设置：

- `optim_wrapper`: 优化器设置信息,
  - `optimizer`: 支持 `pytorch` 所有的优化器，参考相关 [MMEngine](TODO:) 文档。
  - `paramwise_cfg`: 定制不同参数的学习率以及动量，参考相关 [学习策略文档](TODO:) 文档。
- `param_scheduler` : 学习率策略，支持 "CosineAnnealing"、 "Step"、 "Cyclic" 等等，参考相关 [mmcv](https://mmcv.readthedocs.io/zh_CN/latest/_modules/mmcv/runner/hooks/lr_updater.html#LrUpdaterHook) 文档
- `train_cfg | val_cfg`: 训练以及验证的配置，参考相关 [MMEngine](TODO:) 文档。

以下是 ResNet50 基本配置的模型配置['configs/_base_/schedules/imagenet_bs256.py'](https://github.com/open-mmlab/mmclassification/blob/master/configs/_base_/schedules/imagenet_bs256.py)：

```python
# 优化器配置，支持所有 PyTorch 的优化器
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001))

# 学习率参数的调整策略
# 调度流程(scheduler)的策略，也支持 CosineAnnealing, Cyclic, 等,
# 'MultiStepLR' 在 30， 60，90 个 epoch 时， lr = lr * gamma
param_scheduler = dict(
    type='MultiStepLR', by_epoch=True, milestones=[30, 60, 90], gamma=0.1)

# 训练的配置, 迭代 100 个epoch，每一个训练 epoch 后都做验证集评估
# 'by_epoch=True' 默认使用 EpochBaseLoop,  'by_epoch=False' 默认使用 IterBaseLoop
# 参考 MMEngine 获取更多 Runner 和 Loop 的信息
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# 使用自动调整学习率时，基准的 batch_size， 等于 base_num_GPU * base_batch_pre_GPU
auto_scale_lr = dict(base_batch_size=256)
```

### 运行设置

本部分主要包括保存权重策略、日志配置、训练参数、断点权重路径和工作目录等等。

以下是几乎所有算法都使用的运行配置['configs/_base_/default_runtime.py'](https://github.com/open-mmlab/mmclassification/blob/master/configs/_base_/default_runtime.py)文件：

```python
# 默认所有注册器使用的域
default_scope = 'mmcls'

# 配置默认的hook
default_hooks = dict(
    # 记录每次迭代的时间。
    timer=dict(type='IterTimerHook'),

    # 每 100 次迭代打印一次日志。
    logger=dict(type='LoggerHook', interval=100),

    # 启用默认参数调度hook。
    param_scheduler=dict(type='ParamSchedulerHook'),

    # 每个epoch保存检查点。
    checkpoint=dict(type='CheckpointHook', interval=1),

    # 在分布式环境中设置采样器种子。
    sampler_seed=dict(type='DistSamplerSeedHook'),

    # 验证结果可视化，默认不启用，设置 True 时启用。
    visualization=dict(type='VisualizationHook', enable=False),
)

# 配置环境
env_cfg = dict(
   # 是否开启cudnn benchmark
    cudnn_benchmark=False,

    # 设置多进程参数
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),

    # 设置分布式参数
    dist_cfg=dict(backend='nccl'),
)

# 设置可视化工具
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='ClsVisualizer', vis_backends=vis_backends, name='visualizer')

# 设置日志级别
log_level = 'INFO'

# 从哪个检查点加载
load_from = None

# 是否从加载的检查点恢复训练
resume = False
```

## 继承并修改配置文件

为了精简代码、更快的修改配置文件以及便于理解，我们建议继承现有方法。

对于在同一算法文件夹下的所有配置文件，MMClassification 推荐只存在 **一个** 对应的 _原始配置_ 文件。
所有其他的配置文件都应该继承 _原始配置_ 文件，这样就能保证配置文件的最大继承深度为 3。

例如，如果在 ResNet 的基础上做了一些修改，用户首先可以通过指定 `_base_ = './resnet50_8xb32_in1k.py'`（相对于你的配置文件的路径），来继承基础的 ResNet 结构、数据集以及其他训练配置信息，然后修改配置文件中的必要参数以完成继承。如想在基础 resnet50 的基础上使用 `CutMix` 训练增强，将训练轮数由 100 改为 300 和修改学习率衰减轮数，同时修改数据集路径，可以建立新的配置文件 `configs/resnet/resnet50_8xb32-300e_in1k.py`， 文件中写入以下内容:

```python
_base_ = './resnet50_8xb32_in1k.py'

# 模型在之前的基础上使用 CutMix 训练增强
model = dict(
    train_cfg=dict(
        augments=dict(type='CutMix', alpha=1.0, num_classes=1000, prob=1.0)
    )
)

# 优化策略在之前基础上训练更多个 epoch
train_cfg = dict(max_epochs=300, val_interval=10)  # 训练300个 epoch，每10个 epoch 评估一次
param_scheduler = dict(step=[150, 200, 250])   # 学习率调整也有所变动

# 使用自己的数据集目录
train_dataloader = dict(
    dataset=dict(data_root='mydata/imagenet/train'),
)
val_dataloader = dict(
    batch_size=64,                  # 推理时没有反向传播，可以使用更大的 batchsize
    dataset=dict(data_root='mydata/imagenet/val'),
)
test_dataloader = dict(
    batch_size=64,                  # 推理时没有反向传播，可以使用更大的 batchsize
    dataset=dict(data_root='mydata/imagenet/val'),
)
```

### 使用配置文件里的中间变量

用一些中间变量，中间变量让配置文件更加清晰，也更容易修改。

例如数据集里的 `train_pipeline` / `test_pipeline` 是作为数据流水线的中间变量。我们首先要定义 `train_pipeline` / `test_pipeline`，然后将它们传递到 `xx_dataloader` 中。如果想修改训练或测试时输入图片的大小，就需要修改 `train_pipeline` / `test_pipeline` 这些中间变量。

```python
bgr_mean = [103.53, 116.28, 123.675]
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

### 忽略基础配置文件里的部分内容

有时，您需要设置 `_delete_=True` 去忽略基础配置文件里的一些域内容。 可以参照 [MMEngine](TODO:) 来获得一些简单的指导。

以下是一个简单应用案例。 如果在上述 ResNet50 案例中 使用 cosine schedule ,使用继承并直接修改会报 `get unexcepected keyword 'step'` 错, 因为基础配置文件 `param_scheduler` 域信息的 `'step'` 字段被保留下来了，需要加入 `_delete_=True` 去忽略基础配置文件里的 `param_scheduler` 相关域内容：

```python
_base_ = '../../configs/resnet/resnet50_8xb32_in1k.py'

# 训练参数调整
param_scheduler = [
    # 第一个阶段执行 warm up 学习率调整。 第一个阶段 begin 为 0，end 为 5，表示 [0, 5)
    dict(
        type='LinearLR',      # warm up 学习率策略类型
        start_factor=0.25,    # 初始学习率 = lr * start_factor
        by_epoch=True,        # begin 和 end 表示 epoch，如果为 False， 则表示 iter
        begin=0,              # 开始 epoch 序号
        end=5,                # 结束 epoch 序号，epoch 5 不再使用此策略
        convert_to_iter_based=True),  # 是否以iter为基础更新
    # 第二个阶段执行 cos 学习率调整。 第二个阶段 begin 为 5，end 为 100，表示 [5, 100)
    dict(
        type='CosineAnnealingLR', # 使用 CosineAnnealingLR， 半余弦函数
        T_max=95,                 # 半余弦函数的周期为 95。
        by_epoch=True,            # T_max，begin 和 end 表示 epoch，如果为 False， 则表示 iter
        begin=5,
        end=100,
    )
]
```

### 引用基础配置文件里的变量

有时，您可以引用 `_base_` 配置信息的一些域内容，这样可以避免重复定义。 可以参照 [mmcv](https://mmcv.readthedocs.io/zh_CN/latest/understand_mmcv/config.html#reference-variables-from-base) 来获得一些简单的指导。

以下是一个简单应用案例，在训练数据预处理流水线中使用 auto augment 数据增强，参考配置文件 [`configs/resnest/resnest50_32xb64_in1k.py`](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnest/resnest50_32xb64_in1k.py)。 在定义 `train_pipeline` 时，可以直接在 `_base_` 中加入定义 auto augment 数据增强的文件命名，再通过 `{{_base_.auto_increasing_policies}}` 引用变量：

```python
_base_ = [
    '../_base_/models/resnest50.py', '../_base_/datasets/imagenet_bs64.py',
    '../_base_/default_runtime.py', './_randaug_policies.py',
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandAugment',
        policies={{_base_.policies}},    # 这里使用了 _base_ 里的 `policies` 参数。
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

## 通过命令行参数修改配置信息

当用户使用脚本 "tools/train.py" 或者 "tools/test.py" 提交任务，以及使用一些工具脚本时，可以通过指定 `--cfg-options` 参数来直接修改所使用的配置文件内容。

- 更新配置文件内的字典

  可以按照原始配置文件中字典的键的顺序指定配置选项。
  例如，`--cfg-options model.backbone.norm_eval=False` 将主干网络中的所有 BN 模块更改为 `train` 模式。

- 更新配置文件内列表的键

  一些配置字典在配置文件中会形成一个列表。例如，训练流水线 `data.train.pipeline` 通常是一个列表。
  例如，`[dict(type='LoadImageFromFile'), dict(type='TopDownRandomFlip', flip_prob=0.5), ...]` 。如果要将流水线中的 `'flip_prob=0.5'` 更改为 `'flip_prob=0.0'`，您可以这样指定 `--cfg-options data.train.pipeline.1.flip_prob=0.0` 。

- 更新列表/元组的值。

  当配置文件中需要更新的是一个列表或者元组，例如，配置文件通常会设置 `val_evaluator = dict(type='Accuracy', topk=(1, 5))`，用户如果想更改 `topk`，
  需要指定 `--cfg-options val_evaluator.topk="(1,3)"`。注意这里的引号 " 对于列表以及元组数据类型的修改是必要的，
  并且 **不允许** 引号内所指定的值的书写存在空格。

## 导入用户自定义模块

在学习完后续教程 [如何添加新数据集](https://mmclassification.readthedocs.io/zh_CN/latest/tutorials/new_dataset.html)、[如何设计数据处理流程](https://mmclassification.readthedocs.io/zh_CN/latest/tutorials/data_pipeline.html) 、[如何增加新模块](https://mmclassification.readthedocs.io/zh_CN/latest/tutorials/new_modules.html) 后，您可能使用 MMClassification 完成自己的项目并在项目中自定义了数据集、模型、数据增强等。为了精简代码，可以将 MMClassification 作为一个第三方库，只需要保留自己的额外的代码，并在配置文件中导入自定义的模块。案例可以参考 [OpenMMLab 算法大赛项目](https://github.com/zhangrui-wolf/openmmlab-competition-2021)。

只需要在你的配置文件中添加以下代码：

```python
custom_imports = dict(
    imports=['your_dataset_class',
             'your_transforme_class',
             'your_model_class',
             'your_module_class'],
    allow_failed_imports=False)
```

## 常见问题

- 无
