# 如何微调模型

在很多场景下，我们需要快速地将模型应用到新的数据集上，但从头训练模型通常很难快速收敛，这种不确定性会浪费额外的时间。
通常，已有的、在大数据集上训练好的模型会比随机初始化提供更为有效的先验信息，粗略来讲，在此基础上的学习我们称之为模型微调。
已经证明，在 ImageNet 数据集上预先训练的分类模型对于其他数据集和其他下游任务有很好的效果。

因此，该教程提供了如何将 [Model Zoo](../modelzoo_statistics.md) 中提供的预训练模型用于其他数据集，已获得更好的效果。

在新数据集上微调模型分为两步：

- 按照 [数据集准备](dataset_prepare.md) 添加对新数据集的支持。
- 按照本教程中讨论的内容修改配置文件

假设我们现在有一个在 ImageNet-2012 数据集上训练好的 ResNet-50 模型，并且希望在
CIFAR-10 数据集上进行模型微调，我们需要修改配置文件中的五个部分。

## 继承基础配置

首先，创建一个新的配置文件 `configs/tutorial/resnet50_finetune_cifar.py` 来保存我们的配置，当然，这个文件名可以自由设定。

为了重用不同基础配置之间的通用部分，我们支持从多个现有配置中继承配置，其中包括：

- 模型配置：要微调 ResNet-50 模型，可以继承 `_base_/models/resnet50.py` 来搭建模型的基本结构。
- 数据集配置：使用 CIFAR10 数据集，可以继承 `_base_/datasets/cifar10_bs16.py`。
- 训练策略配置：可以继承 batchsize 为 128 的 CIFAR10 数据集基本训练配置文件`_base_/schedules/cifar10_bs128.py`。
- 运行配置：为了保留运行相关设置，比如默认训练钩子、环境配置等，需要继承 `_base_/default_runtime.py`。

要继承以上这些配置文件，只需要把下面一段代码放在我们的配置文件开头。

```python
_base_ = [
    '../_base_/models/resnet50.py',
    '../_base_/datasets/cifar10_bs16.py',
    '../_base_/schedules/cifar10_bs128.py',
    '../_base_/default_runtime.py',
]
```

除此之外，你也可以不使用继承，直接编写完整的配置文件，例如
[`configs/lenet/lenet5_mnist.py`](https://github.com/open-mmlab/mmclassification/blob/master/configs/lenet/lenet5_mnist.py)。

## 修改模型

在进行模型微调时，我们通常希望在主干网络（backbone）加载预训练模型，再用我们的数据集训练一个新的分类头（head）。

为了在主干网络加载预训练模型，我们需要修改主干网络的初始化设置，使用
`Pretrained` 类型的初始化函数。另外，在初始化设置中，我们使用 `prefix='backbone'`
来告诉初始化函数需要加载的子模块的前缀，`backbone`即指加载模型中的主干网络。
方便起见，我们这里使用一个在线的权重文件链接，它
会在训练前自动下载对应的文件，你也可以提前下载这个模型，然后使用本地路径。

接下来，新的配置文件需要按照新数据集的类别数目来修改分类头的配置。只需要修改分
类头中的 `num_classes` 设置即可。

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
这里我们只需要设定我们想要修改的部分配置，其他配置将会自动从我们的父配置文件中获取。
```

另外，当新的小数据集和原本预训练的大数据中的数据分布较为类似的话，我们在进行微调时会希望
冻结主干网络前面几层的参数，只训练后面层以及分类头的参数，这么做有助于在后续训练中，
保持网络从预训练权重中获得的提取低阶特征的能力。在 MMClassification 中，
这一功能可以通过简单的一个 `frozen_stages` 参数来实现。比如我们需要冻结前两层网
络的参数，只需要在上面的配置中添加一行：

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
目前还不是所有的网络都支持 `frozen_stages` 参数，在使用之前，请先检查
[文档](https://mmclassification.readthedocs.io/zh_CN/1.x/api.html#module-mmpretrain.models.backbones)
以确认你所使用的主干网络是否支持。
```

## 修改数据集

当针对一个新的数据集进行微调时，我们通常都需要修改一些数据集相关的配置。比如这
里，我们就需要把 CIFAR-10 数据集中的图像大小从 32 缩放到 224 来配合 ImageNet 上
预训练模型的输入。这一需要可以通过修改数据集的预处理流水线（pipeline）并覆盖数据加载器（dataloader）来实现。

```python
# 数据流水线设置
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
# 数据加载器设置
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader
```

## 修改训练策略设置

用于微调任务的超参数与默认配置不同，通常只需要较小的学习率以及较快的衰减策略。

```python
# 用于批大小为 128 的优化器学习率
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))
# 学习率衰减策略
param_scheduler = dict(
    type='MultiStepLR', by_epoch=True, milestones=[15], gamma=0.1)
```

```{tip}
更多可修改的细节可以参考[如何编写配置文件](config.md).
```

## 开始训练

现在，我们完成了用于微调的配置文件，完整的文件如下：

```python
_base_ = [
    '../_base_/models/resnet50.py',
    '../_base_/datasets/cifar10_bs16.py',
    '../_base_/schedules/cifar10_bs128.py',
    '../_base_/default_runtime.py',
]

# 模型设置
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

# 数据集设置
# 数据流水线设置
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
# 数据加载器设置
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

# 训练策略设置
# 用于批大小为 128 的优化器学习率
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))
# 学习率衰减策略
param_scheduler = dict(
    type='MultiStepLR', by_epoch=True, milestones=[15], gamma=0.1)

```

接下来，我们使用一台 8 张 GPU 的电脑来训练我们的模型，指令如下：

```shell
bash tools/dist_train.sh configs/tutorial/resnet50_finetune_cifar.py 8
```

当然，我们也可以使用单张 GPU 来进行训练，使用如下命令：

```shell
python tools/train.py configs/tutorial/resnet50_finetune_cifar.py
```

但是如果我们使用单张 GPU 进行训练的话，需要在数据集设置部分作如下修改：

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

这是因为我们的训练策略是针对批次大小（batch size）为 128 设置的。在父配置文件中，
设置了单张 `batch_size=16`，如果使用 8 张 GPU，总的批次大小就是 128。而如果使
用单张 GPU，就必须手动修改 `batch_size=128` 来匹配训练策略。
