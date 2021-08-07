# 教程 1：如何微调模型

已经证明，在 ImageNet 数据集上预先训练的分类模型对于其他数据集和其他下游任务有很好的效果。

该教程提供了如何将 [Model Zoo](https://github.com/open-mmlab/mmclassification/blob/master/docs/model_zoo.md) 中提供的预训练模型用于其他数据集，已获得更好的效果。

在新数据集上微调模型分为两步：

- 按照 [教程 2：如何增加新数据集](new_dataset.md) 添加对新数据集的支持。
- 按照本教程中讨论的内容修改配置文件

以 CIFAR10 数据集的微调为例，用户需要修改配置文件中的五个部分。

## 继承基础配置

为了重用不同配置之间的通用部分，我们支持从多个现有配置中继承配置。要微调 ResNet-50 模型，新配置需要继承 `_base_/models/resnet50.py` 来搭建模型的基本结构。为了使用 CIFAR10 数据集，新的配置文件可以直接继承 `_base_/datasets/cifar10.py`。而为了保留运行相关设置，比如训练调整器，新的配置文件需要继承 `_base_/default_runtime.py`。

```python
_base_ = [
    '../_base_/models/resnet50.py',
    '../_base_/datasets/cifar10.py', '../_base_/default_runtime.py'
]
```

除此之外，用户也可以直接编写完整的配置文件，而不是使用继承，例如 `configs/mnist/lenet5.py`。

## 修改分类头

接下来，新的配置文件需要按照新数据集的类别数目来修改分类头的配置。只需要修改分类头中的 `num_classes` 设置，除了最终分类头之外的绝大部分预训练模型权重都会被重用。

```python
_base_ = ['./resnet50.py']
model = dict(
    pretrained=None,
    head=dict(
        type='LinearClsHead',
        num_classes=10,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))
```

## 修改数据集

用户可能还需要准备数据集并编写有关数据集的配置。我们目前支持 MNIST，CIFAR 和 ImageNet 数据集。为了在 CIFAR10 数据集上进行微调，考虑到其原始输入大小为 32，而在 ImageNet 上预训练模型的输入大小为 224，因此我们应将其大小调整为 224。

```python
_base_ = ['./cifar10.py']
img_norm_cfg = dict(
     mean=[125.307, 122.961, 113.8575],
     std=[51.5865, 50.847, 51.255],
     to_rgb=True)
train_pipeline = [
    dict(type='RandomCrop', size=32, padding=4),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Resize', size=224)
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
 ]
 test_pipeline = [
    dict(type='Resize', size=224)
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
 ]
```

## 修改训练调整设置

用于微调任务的超参数与默认配置不同，通常只需要较小的学习率和较少的训练时间。

```python
# 用于批大小为 128 的优化器学习率
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# 学习策略
lr_config = dict(
    policy='step',
    step=[15])
runner = dict(type='EpochBasedRunner', max_epochs=200)
log_config = dict(interval=100)
```

## 使用预训练模型

为了使用预先训练的模型，新的配置文件中需要使用 `load_from` 添加预训练模型权重文件的链接。而为了避免训练过程中自动下载的耗时，用户可以在训练之前下载模型权重文件，并配置本地路径。

```python
load_from = 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmclassification/models/tbd.pth'  # noqa
```
