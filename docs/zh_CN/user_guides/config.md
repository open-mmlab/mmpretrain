# 学习配置文件

为了管理深度学习实验的各种设置，我们使用配置文件来记录所有这些配置。这种配置文件系统具有模块化和继承特性，更多细节可以在{external+mmengine:doc}`MMEngine 中的教程 <advanced_tutorials/config>`。

MMPretrain 主要使用 python 文件作为配置文件，所有配置文件都放置在 [`configs`](https://github.com/open-mmlab/mmpretrain/tree/main/configs) 文件夹下，目录结构如下所示：

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

可以使用 `python tools/misc/print_config.py /PATH/TO/CONFIG` 命令来查看完整的配置信息，从而方便检查所对应的配置文件。

本文主要讲解 MMPretrain 配置文件的命名和结构，以及如何基于已有的配置文件修改，并以 [ResNet50 配置文件](https://github.com/open-mmlab/mmpretrain/blob/main/configs/resnet/resnet50_8xb32_in1k.py) 逐行解释。

## 配置文件结构

在 `configs/_base_` 文件夹下有 4 个基本组件类型，分别是：

- [模型(model)](https://github.com/open-mmlab/mmpretrain/tree/main/configs/_base_/models)
- [数据(data)](https://github.com/open-mmlab/mmpretrain/tree/main/configs/_base_/datasets)
- [训练策略(schedule)](https://github.com/open-mmlab/mmpretrain/tree/main/configs/_base_/schedules)
- [运行设置(runtime)](https://github.com/open-mmlab/mmpretrain/blob/main/configs/_base_/default_runtime.py)

你可以通过继承一些基本配置文件轻松构建自己的训练配置文件。我们称这些被继承的配置文件为 _原始配置文件_，如 `_base_` 文件夹中的文件一般仅作为原始配置文件。

下面使用 [ResNet50 配置文件](https://github.com/open-mmlab/mmpretrain/blob/main/configs/resnet/resnet50_8xb32_in1k.py) 作为案例进行说明并注释每一行含义。

```python
_base_ = [                                    # 此配置文件将继承所有 `_base_` 中的配置
    '../_base_/models/resnet50.py',           # 模型配置
    '../_base_/datasets/imagenet_bs32.py',    # 数据配置
    '../_base_/schedules/imagenet_bs256.py',  # 训练策略配置
    '../_base_/default_runtime.py'            # 默认运行设置
]
```

我们将在下面分别解释这四个原始配置文件。

### 模型配置

模型原始配置文件包含一个 `model` 字典数据结构，主要包括网络结构、损失函数等信息：

- `type`：算法类型，我们支持了多种任务
  - 对于图像分类任务，通常为 `ImageClassifier`，更多细节请参考 [API 文档](mmpretrain.models.classifiers)。
  - 对于自监督任务，有多种类型的算法，例如 `MoCoV2`, `BEiT`, `MAE` 等。更多细节请参考 [API 文档](mmpretrain.models.selfsup)。
  - 对于图像检索任务，通常为 `ImageToImageRetriever`，更多细节请参考 [API 文档](mmpretrain.models.retrievers).

通常，我们使用 **`type`字段** 来指定组件的类，并使用其他字段来传递类的初始化参数。{external+mmengine:doc}`注册器教程 <advanced_tutorials/registry>` 对其进行了详细描述。

这里我们以 [`ImageClassifier`](mmpretrain.models.classifiers.ImageClassifier) 的配置字段为例，对初始化参数进行说明：

- `backbone`： 主干网络设置，主干网络为主要的特征提取网络，比如 `ResNet`, `Swin Transformer`, `Vision Transformer` 等等。更多可用选项请参考 [API 文档](mmpretrain.models.backbones)。
  - 对于自监督学习，有些主干网络需要重新实现，您可以在 [API 文档](mmpretrain.models.selfsup) 中获取更多细节。
- `neck`： 颈网络设置，颈网络主要是连接主干网和头网络的中间部分，比如 `GlobalAveragePooling` 等，更多可用选项请参考 [API 文档](mmpretrain.models.necks)。
- `head`： 头网络设置，头网络主要是与具体任务关联的部件，如图像分类、自监督训练等，更多可用选项请参考 [API 文档](mmpretrain.models.heads)。
  - `loss`： 损失函数设置， 支持 `CrossEntropyLoss`, `LabelSmoothLoss`, `PixelReconstructionLoss` 等，更多可用选项参考 [API 文档](mmpretrain.models.losses)。
- `data_preprocessor`: 图像输入的预处理模块，输入在进入模型前的预处理操作，例如 `ClsDataPreprocessor`, 有关详细信息，请参阅 [API 文档](mmpretrain.models.utils.data_preprocessor)。
- `train_cfg`: `ImageClassifier` 的额外训练配置。在 `ImageClassifier` 中，我们使用这一参数指定批数据增强设置，比如 `Mixup` 和 `CutMix`。详见[文档](mmpretrain.models.utils.batch_augments)。

以下是 ResNet50 的模型配置['configs/_base_/models/resnet50.py'](https://github.com/open-mmlab/mmpretrain/blob/main/configs/_base_/models/resnet50.py)：

```python
model = dict(
    type='ImageClassifier',     # 主模型类型（对于图像分类任务，使用 `ImageClassifier`）
    backbone=dict(
        type='ResNet',          # 主干网络类型
        # 除了 `type` 之外的所有字段都来自 `ResNet` 类的 __init__ 方法
        # 可查阅 https://mmpretrain.readthedocs.io/zh_CN/latest/api/generated/mmpretrain.models.backbones.ResNet.html
        depth=50,
        num_stages=4,           # 主干网络状态(stages)的数目，这些状态产生的特征图作为后续的 head 的输入。
        out_indices=(3, ),      # 输出的特征图输出索引。
        frozen_stages=-1,       # 冻结主干网的层数
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),    # 颈网络类型
    head=dict(
        type='LinearClsHead',         # 分类颈网络类型
        # 除了 `type` 之外的所有字段都来自 `LinearClsHead` 类的 __init__ 方法
        # 可查阅 https://mmpretrain.readthedocs.io/zh_CN/latest/api/generated/mmpretrain.models.heads.LinearClsHead.html
        num_classes=1000,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0), # 损失函数配置信息
        topk=(1, 5),                 # 评估指标，Top-k 准确率， 这里为 top1 与 top5 准确率
    ))
```

### 数据

数据原始配置文件主要包括预处理设置、dataloader 以及 评估器等设置：

- `data_preprocessor`: 模型输入预处理配置，与 `model.data_preprocessor` 相同，但优先级更低。
- `train_evaluator | val_evaluator | test_evaluator`: 构建评估器，参考 [API 文档](mmpretrain.evaluation)。
- `train_dataloader | val_dataloader | test_dataloader`: 构建 dataloader
  - `batch_size`: 每个 GPU 的 batch size
  - `num_workers`: 每个 GPU 的线程数
  - `sampler`: 采样器配置
  - `dataset`: 数据集配置
    - `type`:  数据集类型， MMPretrain 支持 `ImageNet`、 `Cifar` 等数据集 ，参考 [API 文档](mmpretrain.datasets)
    - `pipeline`:  数据处理流水线，参考相关教程文档 [如何设计数据处理流水线](../advanced_guides/pipeline.md)

以下是 ResNet50 的数据配置 ['configs/_base_/datasets/imagenet_bs32.py'](https://github.com/open-mmlab/mmpretrain/blob/main/configs/_base_/datasets/imagenet_bs32.py)：

```python
dataset_type = 'ImageNet'
# 预处理配置
data_preprocessor = dict(
    # 输入的图片数据通道以 'RGB' 顺序
    mean=[123.675, 116.28, 103.53],    # 输入图像归一化的 RGB 通道均值
    std=[58.395, 57.12, 57.375],       # 输入图像归一化的 RGB 通道标准差
    to_rgb=True,                       # 是否将通道翻转，从 BGR 转为 RGB 或者 RGB 转为 BGR
)

train_pipeline = [
    dict(type='LoadImageFromFile'),     # 读取图像
    dict(type='RandomResizedCrop', scale=224),     # 随机放缩裁剪
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),   # 随机水平翻转
    dict(type='PackInputs'),         # 准备图像以及标签
]

test_pipeline = [
    dict(type='LoadImageFromFile'),     # 读取图像
    dict(type='ResizeEdge', scale=256, edge='short'),  # 缩放短边尺寸至 256px
    dict(type='CenterCrop', crop_size=224),     # 中心裁剪
    dict(type='PackInputs'),                 # 准备图像以及标签
]

# 构造训练集 dataloader
train_dataloader = dict(
    batch_size=32,                     # 每张 GPU 的 batchsize
    num_workers=5,                     # 每个 GPU 的线程数
    dataset=dict(                      # 训练数据集
        type=dataset_type,
        data_root='data/imagenet',
        ann_file='meta/train.txt',
        data_prefix='train',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),   # 默认采样器
    persistent_workers=True,                             # 是否保持进程，可以缩短每个 epoch 的准备时间
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
# 验证集评估设置，使用准确率为指标， 这里使用 topk1 以及 top5 准确率
val_evaluator = dict(type='Accuracy', topk=(1, 5))

test_dataloader = val_dataloader  # test dataloader 配置，这里直接与 val_dataloader 相同
test_evaluator = val_evaluator    # 测试集的评估配置，这里直接与 val_evaluator 相同
```

```{note}
预处理配置（`data_preprocessor`）既可以作为 `model` 的一个子字段，也可以定义在外部的 `data_preprocessor` 字段，
同时配置时，优先使用 `model.data_preprocessor` 的配置。
```

### 训练策略

训练策略原始配置文件主要包括预优化器设置和训练、验证及测试的循环控制器(LOOP)：

- `optim_wrapper`: 优化器装饰器配置信息，我们使用优化器装饰配置优化进程。
  - `optimizer`: 支持 `pytorch` 所有的优化器，参考相关 {external+mmengine:doc}`MMEngine <tutorials/optim_wrapper>` 文档。
  - `paramwise_cfg`: 根据参数的类型或名称设置不同的优化参数，参考相关 [学习策略文档](../advanced_guides/schedule.md) 文档。
  - `accumulative_counts`: 积累几个反向传播后再优化参数，你可以用它通过小批量来模拟大批量。
- `param_scheduler` : 学习率策略，你可以指定训练期间的学习率和动量曲线。有关详细信息，请参阅 MMEngine 中的 {external+mmengine:doc}`文档 <tutorials/param_scheduler>`。
- `train_cfg | val_cfg | test_cfg`: 训练、验证以及测试的循环执行器配置，请参考相关的{external+mmengine:doc}`MMEngine 文档 <design/runner>`。

以下是 ResNet50 的训练策略配置['configs/_base_/schedules/imagenet_bs256.py'](https://github.com/open-mmlab/mmpretrain/blob/main/configs/_base_/schedules/imagenet_bs256.py)：

```python
optim_wrapper = dict(
    # 使用 SGD 优化器来优化参数
    optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001))

# 学习率参数的调整策略
# 'MultiStepLR' 表示使用多步策略来调度学习率（LR）。
param_scheduler = dict(
    type='MultiStepLR', by_epoch=True, milestones=[30, 60, 90], gamma=0.1)

# 训练的配置，迭代 100 个 epoch，每一个训练 epoch 后都做验证集评估
# 'by_epoch=True' 默认使用 `EpochBaseLoop`,  'by_epoch=False' 默认使用 `IterBaseLoop`
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
# 使用默认的验证循环控制器
val_cfg = dict()
# 使用默认的测试循环控制器
test_cfg = dict()

# 通过默认策略自动缩放学习率，此策略适用于总批次大小 256
# 如果你使用不同的总批量大小，比如 512 并启用自动学习率缩放
# 我们将学习率扩大到 2 倍
auto_scale_lr = dict(base_batch_size=256)
```

### 运行设置

本部分主要包括保存权重策略、日志配置、训练参数、断点权重路径和工作目录等等。

以下是几乎所有算法都使用的运行配置['configs/_base_/default_runtime.py'](https://github.com/open-mmlab/mmpretrain/blob/main//configs/_base_/default_runtime.py)：

```python
# 默认所有注册器使用的域
default_scope = 'mmpretrain'

# 配置默认的 hook
default_hooks = dict(
    # 记录每次迭代的时间。
    timer=dict(type='IterTimerHook'),

    # 每 100 次迭代打印一次日志。
    logger=dict(type='LoggerHook', interval=100),

    # 启用默认参数调度 hook。
    param_scheduler=dict(type='ParamSchedulerHook'),

    # 每个 epoch 保存检查点。
    checkpoint=dict(type='CheckpointHook', interval=1),

    # 在分布式环境中设置采样器种子。
    sampler_seed=dict(type='DistSamplerSeedHook'),

    # 验证结果可视化，默认不启用，设置 True 时启用。
    visualization=dict(type='VisualizationHook', enable=False),
)

# 配置环境
env_cfg = dict(
   # 是否开启 cudnn benchmark
    cudnn_benchmark=False,

    # 设置多进程参数
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),

    # 设置分布式参数
    dist_cfg=dict(backend='nccl'),
)

# 设置可视化工具
vis_backends = [dict(type='LocalVisBackend')] # 使用磁盘(HDD)后端
visualizer = dict(
    type='UniversalVisualizer', vis_backends=vis_backends, name='visualizer')

# 设置日志级别
log_level = 'INFO'

# 从哪个检查点加载
load_from = None

# 是否从加载的检查点恢复训练
resume = False
```

## 继承并修改配置文件

为了精简代码、更快的修改配置文件以及便于理解，我们建议继承现有方法。

对于在同一算法文件夹下的所有配置文件，MMPretrain 推荐只存在 **一个** 对应的 _原始配置_ 文件。
所有其他的配置文件都应该继承 _原始配置_ 文件，这样就能保证配置文件的最大继承深度为 3。

例如，如果在 ResNet 的基础上做了一些修改，用户首先可以通过指定 `_base_ = './resnet50_8xb32_in1k.py'`（相对于你的配置文件的路径），来继承基础的 ResNet 结构、数据集以及其他训练配置信息，然后修改配置文件中的必要参数以完成继承。如想在基础 resnet50 的基础上使用 `CutMix` 训练增强，将训练轮数由 100 改为 300 和修改学习率衰减轮数，同时修改数据集路径，可以建立新的配置文件 `configs/resnet/resnet50_8xb32-300e_in1k.py`， 文件中写入以下内容：

```python
# 在 'configs/resnet/' 创建此文件
_base_ = './resnet50_8xb32_in1k.py'

# 模型在之前的基础上使用 CutMix 训练增强
model = dict(
    train_cfg=dict(
        augments=dict(type='CutMix', alpha=1.0)
    )
)

# 优化策略在之前基础上训练更多个 epoch
train_cfg = dict(max_epochs=300, val_interval=10)  # 训练 300 个 epoch，每 10 个 epoch 评估一次
param_scheduler = dict(step=[150, 200, 250])   # 学习率调整也有所变动

# 使用自己的数据集目录
train_dataloader = dict(
    dataset=dict(data_root='mydata/imagenet/train'),
)
val_dataloader = dict(
    batch_size=64,                  # 验证时没有反向传播，可以使用更大的 batchsize
    dataset=dict(data_root='mydata/imagenet/val'),
)
test_dataloader = dict(
    batch_size=64,                  # 测试时没有反向传播，可以使用更大的 batchsize
    dataset=dict(data_root='mydata/imagenet/val'),
)
```

### 使用配置文件里的中间变量

用一些中间变量，中间变量让配置文件更加清晰，也更容易修改。

例如数据集里的 `train_pipeline` / `test_pipeline` 是作为数据流水线的中间变量。我们首先要定义它们，然后将它们传递到 `train_dataloader` / `test_dataloader` 中。如果想修改训练或测试时输入图片的大小，就需要修改 `train_pipeline` / `test_pipeline` 这些中间变量。

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

### 忽略基础配置文件里的部分内容

有时，您需要设置 `_delete_=True` 去忽略基础配置文件里的一些域内容。可以查看 {external+mmengine:doc}`MMEngine 文档 <advanced_tutorials/config>` 进一步了解该设计。

以下是一个简单应用案例。 如果在上述 ResNet50 案例中 使用余弦调度 ,使用继承并直接修改会报 `get unexcepected keyword 'step'` 错，因为基础配置文件 `param_scheduler` 域信息的 `'step'` 字段被保留下来了，需要加入 `_delete_=True` 去忽略基础配置文件里的 `param_scheduler` 相关域内容：

```python
_base_ = '../../configs/resnet/resnet50_8xb32_in1k.py'

# 学习率调整策略
param_scheduler = dict(type='CosineAnnealingLR', by_epoch=True, _delete_=True)
```

### 引用基础配置文件里的变量

有时，您可以引用 `_base_` 配置信息的一些域内容，这样可以避免重复定义。可以查看 {external+mmengine:doc}`MMEngine 文档 <advanced_tutorials/config>` 进一步了解该设计。

以下是一个简单应用案例，在训练数据预处理流水线中使用 `auto augment` 数据增强，参考配置文件 [`configs/resnest/resnest50_32xb64_in1k.py`](https://github.com/open-mmlab/mmpretrain/blob/main/configs/resnest/resnest50_32xb64_in1k.py)。 在定义 `train_pipeline` 时，可以直接在 `_base_` 中加入定义 auto augment 数据增强的文件命名，再通过 `{{_base_.auto_increasing_policies}}` 引用变量：

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
    dict(type='PackInputs'),
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
