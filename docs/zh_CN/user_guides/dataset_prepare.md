# 准备数据集

目前 MMClassification 所支持的数据集有：

- [CustomDataset](#customdataset)
- [ImageNet](#imagenet)
- [CIFAR](#cifar)
- [MINIST](#mnist)
- [OpenMMLab 2.0 标准数据集](#openmmlab-20-标准数据集)
- [其他数据集](#其他数据集)
- [数据集包装](#数据集包装)

如果你使用的数据集不在以上所列公开数据集中，需要转换数据集格式来适配 **`CustomDataset`**。

## CustomDataset

[`CustomDataset`](mmpretrain.datasets.CustomDataset) 是一个通用的数据集类，供您使用自己的数据集。目前 `CustomDataset` 支持以下两种方式组织你的数据集文件：

### 子文件夹方式

文件夹格式通过文件来区别图片的类别，如下， class_1 和 class_2 就代表了区分了不同的类别。

```text
data_prefix/
├── class_1
│   ├── xxx.png
│   ├── xxy.png
│   └── ...
├── class_2
│   ├── 123.png
│   ├── 124.png
│   └── ...
```

假如你希望将之用于训练，那么配置文件中需要添加以下配置：

```python
train_dataloader = dict(
    ...
    # 训练数据集配置
    dataset=dict(
        type='CustomDataset',
        data_prefix='path/to/data_prefix',
        pipeline=...
    )
)
```

### 标注文件方式

标注文件格式主要使用文本文件来保存类别信息，`data_prefix` 存放图片，`ann_file` 存放标注类别信息。

如下案例，dataset 目录如下：

```text
data_root/
├── meta/
│   ├── ann_file
│   └── ...
├── data_prefix/
│   ├── folder_1
│   │   ├── xxx.png │   │   ├── xxy.png
│   │   └── ...
│   ├── 123.png
│   ├── nsdf3.png
│   └── ...
```

标注文件 `ann_file` 内为普通文本，分为两列，第一列为图片路径，第二列为**类别的序号**。如下：

```text
folder_1/xxx.png 0
folder_1/xxy.png 1
123.png 1
nsdf3.png 2
...
```

```{note}
类别序号的值应当在 [0, num_classes - 1] 范围的整数。
```

另外还需要数据集配置文件中指定 `classes` 字段，如：

```python
train_dataloader = dict(
    ...
    # 训练数据集配置
    dataset=dict(
        type='CustomDataset',
        ann_file='path/to/ann_file_path',
        data_prefix='path/to/images',
        classes=['A', 'B', 'C', 'D', ...]
        pipeline=...,
    )
)
```

```{note}
如果指定了 'ann_file'， 则通过 'ann_file' 得到标注信息；否则，按照子文件夹格式处理。
```

## ImageNet

ImageNet 有多个版本，但最常用的一个是 [ILSVRC 2012](http://www.image-net.org/challenges/LSVRC/2012/)。 可以通过以下步骤使用它。

1. 注册一个帐户并登录到[下载页面](http://www.image-net.org/download-images)。
2. 找到ILSVRC2012的下载链接，下载以下两个文件：
   - ILSVRC2012_img_train.tar (~138GB)
   - ILSVRC2012_img_val.tar (~6.3GB)
3. 解压已下载的图片。
4. 从此 [链接](https://download.openmmlab.com/mmclassification/datasets/imagenet/meta/caffe_ilsvrc12.tar.gz) 下载并解压标注文件。
5. 根据标注数据中的路径重新组织图像文件，应该是这样的:

```text
   imagenet/
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
        ann_file='meta/train.txt',
        data_prefix='train/',
        pipeline=...,
    )
)
val_dataloader = dict(
    ...
    # 验证数据集配置
    dataset=dict(
        type='ImageNet',
        data_root='imagenet_folder',
        ann_file='meta/val.txt',
        data_prefix='val/',
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

```json

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
                'img_label': 0,
                ...
            },
            {
                'img_path': "xxx/xxx_1.jpg",
                'img_label': 1,
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

MMCLassification 还是支持更多其他的数据集，可以通过查阅[数据集文档](mmpretrain.datasets)获取它们的配置信息。

## 数据集包装

MMEngine 中支持以下数据包装器，您可以参考 {external+mmengine:doc}`MMEngine 教程 <advanced_tutorials/basedataset>` 了解如何使用它。

- {external:py:class}`~mmengine.dataset.ConcatDataset`
- {external:py:class}`~mmengine.dataset.RepeatDataset`
- {external:py:class}`~mmengine.dataset.ClassBalancedDataset`

除上述之外，MMClassification 还支持了[KFoldDataset](mmpretrain.datasets.KFoldDataset)，需用通过使用 `tools/kfold-cross-valid.py` 来使用它。
