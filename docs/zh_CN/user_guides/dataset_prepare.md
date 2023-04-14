# 准备数据集

## CustomDataset

[`CustomDataset`](mmpretrain.datasets.CustomDataset) 是一个通用的数据集类，供您使用自己的数据集。目前 `CustomDataset` 支持以下两种方式组织你的数据集文件：

### 子文件夹方式

在这种格式下，您只需要重新组织您的数据集文件夹并将所有样本放在一个文件夹中，而无需创建任何标注文件。

对于监督任务（使用 `with_label=true`），我们使用子文件夹的名称作为类别名称，如下例所示，`class_x` 和 `class_y` 将被识别为类别名称。

```text
data_prefix/
├── class_x
│   ├── xxx.png
│   ├── xxy.png
│   └── ...
│       └── xxz.png
└── class_y
    ├── 123.png
    ├── nsdf3.png
    ├── ...
    └── asd932_.png
```

对于无监督任务（使用 `with_label=false`），我们直接加载指定文件夹下的所有样本文件：

```
data_prefix/
├── folder_1
│   ├── xxx.png
│   ├── xxy.png
│   └── ...
├── 123.png
├── nsdf3.png
└── ...
```

假如你希望将之用于训练，那么配置文件中需要添加以下配置：

```python
train_dataloader = dict(
    ...
    # 训练数据集配置
    dataset=dict(
        type='CustomDataset',
        data_prefix='path/to/data_prefix',
        with_label=True,  # 对于无监督任务，使用 False
        pipeline=...
    )
)
```

```{note}
如果要使用此格式，请不要指定 `ann_file`，或指定 `ann_file=''`。

请注意，子文件夹格式需要对文件夹进行扫描，这可能会导致初始化速度变慢，尤其是对于大型数据集或慢速文件 IO。
```

### 标注文件方式

标注文件格式主要使用文本文件来保存类别信息，`data_prefix` 存放图片，`ann_file` 存放标注类别信息。

如下案例，dataset 目录如下：

在这种格式中，我们使用文本标注文件来存储图像文件路径和对应的类别索引。

对于监督任务（`with_label=true`），注释文件应在一行中包含一个样本的文件路径和类别索引，并用空格分隔，如下所示：

所有这些文件路径都可以是绝对路径，也可以是相对于 `data_prefix` 的相对路径。

```text
folder_1/xxx.png 0
folder_1/xxy.png 1
123.png 4
nsdf3.png 3
...
```

```{note}
类别的索引号从 0 开始。真实标签的值应在`[0, num_classes - 1]`范围内。

此外，请使用数据集设置中的 `classes` 字段来指定每个类别的名称
```

对于无监督任务（`with_label=false`），标注文件只需要在一行中包含一个样本的文件路径，如下：

```text
folder_1/xxx.png
folder_1/xxy.png
123.png
nsdf3.png
...
```

假设整个数据集文件夹如下：

```text
data_root
├── meta
│   ├── test.txt     # 测试数据集的标注文件
│   ├── train.txt    # 训练数据集的标注文件
│   └── val.txt      # 验证数据集的标注文件

├── train
│   ├── 123.png
│   ├── folder_1
│   │   ├── xxx.png
│   │   └── xxy.png
│   └── nsdf3.png
├── test
└── val
```

这是配置文件中的数据集设置的示例：

```python
# 训练数据设置
train_dataloader = dict(
    dataset=dict(
        type='CustomDataset',
        data_root='path/to/data_root',  # `ann_flie` 和 `data_prefix` 共同的文件路径前缀
        ann_file='meta/train.txt',      # 相对于 `data_root` 的标注文件路径
        data_prefix='train',            # `ann_file` 中文件路径的前缀，相对于 `data_root`
        classes=['A', 'B', 'C', 'D', ...],  # 每个类别的名称
        pipeline=...,    # 处理数据集样本的一系列变换操作
    )
    ...
)
```

```{note}
有关如何使用 `CustomDataset` 的完整示例，请参阅[如何使用自定义数据集进行预训练](../notes/pretrain_custom_dataset.md)
```

## ImageNet

ImageNet 有多个版本，但最常用的一个是 [ILSVRC 2012](http://www.image-net.org/challenges/LSVRC/2012/)。 可以通过以下步骤使用它。

1. 注册一个帐户并登录到[下载页面](http://www.image-net.org/download-images)。
2. 找到 ILSVRC2012 的下载链接，下载以下两个文件：
   - ILSVRC2012_img_train.tar (~138GB)
   - ILSVRC2012_img_val.tar (~6.3GB)
3. 解压已下载的图片。
4. 根据 [CustomDataset](#CustomDataset) 的格式约定重新组织图像文件

```{note}
在 MMPretrain 中，所有配置文件默认使用文本标注文件的数据集。因此，如果希望使用子文件夹格式，请在配置文件中设置 `ann_file=''`。
```

### 子文件夹格式

重新组织数据集如下：

```text
data/imagenet/
├── train/
│   ├── n01440764
│   │   ├── n01440764_10026.JPEG
│   │   ├── n01440764_10027.JPEG
│   │   ├── n01440764_10029.JPEG
│   │   ├── n01440764_10040.JPEG
│   │   ├── n01440764_10042.JPEG
│   │   ├── n01440764_10043.JPEG
│   │   └── n01440764_10048.JPEG
│   ├── ...
├── val/
│   ├── n01440764
│   │   ├── ILSVRC2012_val_00000293.JPEG
│   │   ├── ILSVRC2012_val_00002138.JPEG
│   │   ├── ILSVRC2012_val_00003014.JPEG
│   │   └── ...
│   ├── ...
```

然后，您可以使用具有以下配置的 [`ImageNet`](mmpretrain.datasets.ImageNet) 数据集：

```python
train_dataloader = dict(
    ...
    # 训练数据集配置
    dataset=dict(
        type='ImageNet',
        data_root='data/imagenet',
        data_prefix='train',
        ann_file='',
        pipeline=...,
    )
)

val_dataloader = dict(
    ...
    # 验证数据集配置
    dataset=dict(
        type='ImageNet',
        data_root='data/imagenet',
        data_prefix='val',
        ann_file='',
        pipeline=...,
    )
)

test_dataloader = val_dataloader
```

### 文本标注文件格式

您可以从[此链接](https://download.openmmlab.com/mmclassification/datasets/imagenet/meta/caffe_ilsvrc12.tar.gz)下载并解压元数据，然后组织文件夹如下：

```text
data/imagenet/
├── meta/
│   ├── train.txt
│   ├── test.txt
│   └── val.txt
├── train/
│   ├── n01440764
│   │   ├── n01440764_10026.JPEG
│   │   ├── n01440764_10027.JPEG
│   │   ├── n01440764_10029.JPEG
│   │   ├── n01440764_10040.JPEG
│   │   ├── n01440764_10042.JPEG
│   │   ├── n01440764_10043.JPEG
│   │   └── n01440764_10048.JPEG
│   ├── ...
├── val/
│   ├── ILSVRC2012_val_00000001.JPEG
│   ├── ILSVRC2012_val_00000002.JPEG
│   ├── ILSVRC2012_val_00000003.JPEG
│   ├── ILSVRC2012_val_00000004.JPEG
│   ├── ...
```

然后，您可以使用具有以下配置的 [`ImageNet`](mmpretrain.datasets.ImageNet) 数据集：

```python
train_dataloader = dict(
    ...
    # 训练数据集配置
    dataset=dict(
        type='ImageNet',
        data_root='imagenet_folder',
        data_prefix='train/',
        ann_file='meta/train.txt',
        pipeline=...,
    )
)

val_dataloader = dict(
    ...
    # 验证数据集配置
    dataset=dict(
        type='ImageNet',
        data_root='imagenet_folder',
        data_prefix='val/',
        ann_file='meta/val.txt',
        pipeline=...,
    )
)

test_dataloader = val_dataloader
```

## CIFAR

我们支持自动下载 [`CIFAR10`](mmpretrain.datasets.CIFAR10) 和 [`CIFAR100`](mmpretrain.datasets.CIFAR100) 数据集，您只需在 `data_root` 字段中指定下载文件夹即可。 并且通过指定 `test_mode=False` / `test_mode=True` 来使用训练数据集或测试数据集。

```python
train_dataloader = dict(
    ...
    # 训练数据集配置
    dataset=dict(
        type='CIFAR10',
        data_root='data/cifar10',
        test_mode=False,
        pipeline=...,
    )
)
val_dataloader = dict(
    ...
    # 验证数据集配置
    dataset=dict(
        type='CIFAR10',
        data_root='data/cifar10',
        test_mode=True,
        pipeline=...,
    )
)
test_dataloader = val_dataloader
```

## MNIST

我们支持自动下载 [MNIST](mmpretrain.datasets.MNIST) 和 [Fashion-MNIST](mmpretrain.datasets.FashionMNIST) 数据集，您只需指定 `data_root` 字段中的下载路径即可。 并且通过指定 `test_mode=False` / `test_mode=True` 来使用训练数据集或测试数据集。

```python
train_dataloader = dict(
    ...
    # 训练数据集配置
    dataset=dict(
        type='MNIST',
        data_root='data/mnist',
        test_mode=False,
        pipeline=...,
    )
)
val_dataloader = dict(
    ...
    # 验证数据集配置
    dataset=dict(
        type='MNIST',
        data_root='data/mnist',
        test_mode=True,
        pipeline=...,
    )
)
test_dataloader = val_dataloader
```

## OpenMMLab 2.0 标准数据集

为了统一不同任务的数据集接口，便于多任务的算法模型训练，OpenMMLab 制定了 **OpenMMLab 2.0 数据集格式规范**， 数据集标注文件需符合该规范，数据集基类基于该规范去读取与解析数据标注文件。如果用户提供的数据标注文件不符合规定格式，用户可以选择将其转化为规定格式，并使用 OpenMMLab 的算法库基于该数据标注文件进行算法训练和测试。

OpenMMLab 2.0 数据集格式规范规定，标注文件必须为 `json` 或 `yaml`，`yml` 或 `pickle`，`pkl` 格式；标注文件中存储的字典必须包含 `metainfo` 和 `data_list` 两个字段。其中 `metainfo` 是一个字典，里面包含数据集的元信息；`data_list` 是一个列表，列表中每个元素是一个字典，该字典定义了一个原始数据（raw data），每个原始数据包含一个或若干个训练/测试样本。

假设您要使用训练数据集，那么配置文件如下所示：

```

{
    'metainfo':
        {
            'classes': ('cat', 'dog'), # 'cat' 的类别序号为 0，'dog' 为 1。
            ...
        },
    'data_list':
        [
            {
                'img_path': "xxx/xxx_0.jpg",
                'gt_label': 0,
                ...
            },
            {
                'img_path': "xxx/xxx_1.jpg",
                'gt_label': 1,
                ...
            },
            ...
        ]
}
```

同时假设数据集存放路径如下：

```text
data
├── annotations
│   ├── train.json
│   └── ...
├── train
│   ├── xxx/xxx_0.jpg
│   ├── xxx/xxx_1.jpg
│   ├── ...
```

通过以下字典构建：

```python
dataset_cfg=dict(
    type='CustomDataset',
    ann_file='path/to/ann_file_path',
    data_prefix='path/to/images_folder',
    pipeline=transfrom_list)
```

## 其他数据集

MMPretrain 还支持更多其他的数据集，可以通过查阅[数据集文档](mmpretrain.datasets)获取它们的配置信息。

如果需要使用一些特殊格式的数据集，您需要实现您自己的数据集类，请参阅[添加新数据集](../advanced_guides/datasets.md)。

## 数据集包装

MMEngine 中支持以下数据包装器，您可以参考 {external+mmengine:doc}`MMEngine 教程 <advanced_tutorials/basedataset>` 了解如何使用它。

- {external:py:class}`~mmengine.dataset.ConcatDataset`
- {external:py:class}`~mmengine.dataset.RepeatDataset`
- {external:py:class}`~mmengine.dataset.ClassBalancedDataset`

除上述之外，MMPretrain 还支持了[KFoldDataset](mmpretrain.datasets.KFoldDataset)，需用通过使用 `tools/kfold-cross-valid.py` 来使用它。
