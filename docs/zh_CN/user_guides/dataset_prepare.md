# 准备数据集

目前 MMClassification 所支持的数据集有：

- [CustomDataset](https://mmclassification.readthedocs.io/zh_CN/1.x/api/datasets.html#custom-dataset)
- [ImageNet](https://mmclassification.readthedocs.io/zh_CN/1.x/api/datasets.html#imagenet)
  - ImageNet-1k
  - ImageNet-21k
- [CIFAR](https://mmclassification.readthedocs.io/zh_CN/1.x/api/datasets.html#cifar)
  - CIFAR10
  - CIFAR100
- [MINIST](https://mmclassification.readthedocs.io/zh_CN/1.x/api/datasets.html#mnist)
  - MINIST
  - FashionMNIST
- [CUB](https://mmclassification.readthedocs.io/zh_CN/1.x/api/datasets.html#cub)
- [VOC](https://mmclassification.readthedocs.io/zh_CN/1.x/api/datasets.html#voc)

如果你使用的数据集不在以上所列公开数据集中，需要转换数据集格式来适配 **`CustomDataset`**。

## 适配CustomDataset

`CustomDataset` 支持以下三种数据格式：

### 子文件夹格式

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

配置文件中需要指定 'data_prefix'，如下：

```python
dataset_cfg=dict(
    type='CustomDataset',
    data_prefix='path/to/data_prefix,
    pipeline=transfrom_list)
```

### 标注文件格式

标注文件格式主要使用文本文件来保存类别信息，`data_prefix` 存放图片，`ann_file` 存放标注类别信息。

如下案例，dataset 目录如下：

```text
data_root/
├── meta/
│   ├── ann_file
│   └── ...
├── data_prefix/
│   ├── folder_1
│   │   ├── xxx.png
│   │   ├── xxy.png
│   │   └── ...
│   ├── 123.png
│   ├── nsdf3.png
│   └── ...
```

标注文件 `ann_file` 内为普通文本，分为两列，第一列为图片路径，第二列为**类别的序号**。如下：

```text
folder_1/xxx.png 0  # 类别序号从 0 开始
folder_1/xxy.png 1
123.png 1
nsdf3.png 2
...
```

另外还需要数据集配置文件中指定 `classes` 字段，如：

```python
dataset_cfg=dict(
    type='CustomDataset',
    ann_file='path/to/ann_file_path',
    data_prefix='path/to/images',
    classes=['A', 'B', 'C', 'D'....]
    pipeline=transfrom_list)
```

```{note}
类别序号的值应当在 [0, num_classes - 1] 范围的整数。
```

```{note}
如果指定了 'ann_file'， 则通过 'ann_file' 得到标注信息；否则，按照子文件夹格式处理。
```

### OpenMMLab 2.0 规范

为了统一不同任务的数据集接口，便于多任务的算法模型训练，OpenMMLab 制定了 **OpenMMLab 2.0 数据集格式规范**， 数据集标注文件需符合该规范，数据集基类基于该规范去读取与解析数据标注文件。如果用户提供的数据标注文件不符合规定格式，用户可以选择将其转化为规定格式，并使用 OpenMMLab 的算法库基于该数据标注文件进行算法训练和测试。

OpenMMLab 2.0 数据集格式规范规定，标注文件必须为 `json` 或 `yaml`，`yml` 或 `pickle`，`pkl` 格式；标注文件中存储的字典必须包含 `metainfo` 和 `data_list` 两个字段。其中 `metainfo` 是一个字典，里面包含数据集的元信息；`data_list` 是一个列表，列表中每个元素是一个字典，该字典定义了一个原始数据（raw data），每个原始数据包含一个或若干个训练/测试样本。

以下是一个 JSON 标注文件的例子（该例子中每个原始数据只包含一个训练/测试样本）:

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

## 数据集类包装

MMEngine 中支持以下数据包装器，您可以参考 [MMEngine 教程](TODO:) 了解如何使用它。

- [ConcatDataset](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/tutorials/basedataset.md#concatdataset)
- [RepeatDataset](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/tutorials/basedataset.md#repeatdataset)
- [ClassBalanced](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/tutorials/basedataset.md#classbalanceddataset)

除上述之外，MMClassification 还支持了[KFoldDataset](https://mmclassification.readthedocs.io/zh_CN/1.x/api/datasets.html#kfoldfataset).
