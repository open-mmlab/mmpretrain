# 教程 3：如何添加新数据集

目前 MMClassification 所支持的数据集以及数据集基类包装有：

- [CustomDataset](https://mmclassification.readthedocs.io/zh_CN/latest/api/datasets.html#custom-dataset)
- [ImageNet](https://mmclassification.readthedocs.io/zh_CN/latest/api/datasets.html#imagenet)
  - ImageNet-1k
  - ImageNet-21k
- [CIFAR](https://mmclassification.readthedocs.io/zh_CN/latest/api/datasets.html#cifar)
  - CIFAR10
  - CIFAR100
- [MINIST](https://mmclassification.readthedocs.io/zh_CN/latest/api/datasets.html#mnist)
  - MINIST
  - FashionMNIST
- [CUB](https://mmclassification.readthedocs.io/zh_CN/latest/api/datasets.html#cub)
- [VOC](https://mmclassification.readthedocs.io/zh_CN/latest/api/datasets.html#voc)
- [DatasetWrapper](https://mmclassification.readthedocs.io/zh_CN/latest/api/datasets.html#dataset-wrappers)
  - ConcatDataset
  - RepeatDataset
  - ClassBalancedDataset
  - KFoldDataset

如果你使用的数据集不在以上所列公开数据集中，可以通过**转换数据集格式适配 `CustomDataset`** 或者**添加新数据集类**。

## 适配CustomDataset

`CustomDataset` 支持以下三种数据格式：

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
folder_1/xxx.png 0
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

```note
类别序号的值应当属于 [0, num_classes - 1] 范围。
```

### 子文件夹格式

文件夹格式通过文件来区别图片的类别，如下， folfer_1 和 folfer_2 就代表了区分了不同的类别。

```text
data_prefix/
├── folder_1     # It is recommended to use the category name as the folder name
│   ├── xxx.png
│   ├── xxy.png
│   └── ...
├── folder_2
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

```note
如果指定了 'ann_file'， 则通过 'ann_file' 得到标注信息，否则，按照子文件夹格式处理。
```

### 添加新数据集类

用户可以编写一个继承自 [`BasesDataset`](https://mmclassification.readthedocs.io/zh_CN/latest/_modules/mmcls/datasets/base_dataset.html#BaseDataset) 的新数据集类，并重载 `load_data_list(self)` 方法，类似 [CIFAR10](https://github.com/open-mmlab/mmclassification/blob/master/mmcls/datasets/cifar.py) 和 [ImageNet](https://github.com/open-mmlab/mmclassification/blob/master/mmcls/datasets/imagenet.py)。

通常，此方法返回一个包含所有样本的列表，其中的每个样本都是一个字典。字典中包含了必要的数据信息，例如 `img` 和 `gt_label`。

假设我们将要实现一个 `Filelist` 数据集，该数据集将使用文件列表进行训练和测试。注释列表的格式如下：

```text
000001.jpg 0
000002.jpg 1
...
```

我们可以在 `mmcls/datasets/filelist.py` 中创建一个新的数据集类以加载数据。

```python
import numpy as np

from mmcls.registry import DATASETS
from .base_dataset import BaseDataset


@DATASETS.register_module()
class Filelist(BaseDataset):

    def load_data_list(self):
        assert isinstance(self.ann_file, str)

        data_list = []
        with open(self.ann_file) as f:
            samples = [x.strip().split(' ') for x in f.readlines()]
            for filename, gt_label in samples:
                img_path = add_prefix(filename, self.img_prefix)
                info = {'img_path': img_path, 'gt_label': int(gt_label)}
                data_list.append(info)
        return data_list
```

将新的数据集类加入到 `mmcls/datasets/__init__.py` 中：

```python
from .base_dataset import BaseDataset
...
from .filelist import Filelist

__all__ = [
    'BaseDataset', ... ,'Filelist'
]
```

然后在配置文件中，为了使用 `Filelist`，用户可以按以下方式修改配置

```python
train = dict(
    type='Filelist',
    data_prefix='path/to/images',
    ann_file = 'image_list.txt',
    pipeline=transfrom_list
)
```

```note
如果数据样本时获取的字典中，只包含了 'img_path' 不包含 'img'， 则在 pipeline 中必须包含 'LoadImgFromFile'。
```

无论是 `CustomData` 还是自定义的数据集类，都继承了 [`BaseDataset`](https://github.com/open-mmlab/mmclassification/blob/master/mmcls/datasets/base_dataset.py), 其基本用法，**懒加载**以及**节省内存**的特性可以参考相关文档 [mmengine.basedataset](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/tutorials/basedataset.md)。

## 通过混合数据集来自定义数据集

MMClassification 还支持混合数据集以进行训练。目前支持合并和重复数据集。

### 合并数据集

我们使用 `ConcatDataset` 作为一个合并数据集的封装。举个例子，假设原始数据集是 `Dataset_A` 以及 `Dataset_B`，为了合并它们，我们需要如下的配置文件：

```python
dataset_A_train = dict(
        type='ConcatDataset',
        datasets=[
            dict(type='Dataset_A', ...， pipeline=train_pipeline),
            dict(type='Dataset_B', ..., pipeline=transfrom_list),
        ])
```

### 重复数据集

我们使用 `RepeatDataset` 作为一个重复数据集的封装。举个例子，假设原始数据集是 `Dataset_A`，为了重复它，我们需要如下的配置文件：

```python
dataset_A_train = dict(
        type='RepeatDataset',
        times=N,
        dataset=dict(  # 这里是 Dataset_A 的原始配置
            type='Dataset_A',
            ...,
            pipeline=transfrom_list
        )
    )
```

### 类别平衡数据集

我们使用 `ClassBalancedDataset` 作为根据类别频率对数据集进行重复采样的封装类。进行重复采样的数据集需要实现函数 `self.get_cat_ids(idx)` 以支持 `ClassBalancedDataset`。

举个例子，按照 `oversample_thr=1e-3` 对 `Dataset_A` 进行重复采样，需要如下的配置文件：

```python
dataset_A_train = dict(
        type='ClassBalancedDataset',
        oversample_thr=1e-3,
        dataset=dict(  # 这里d是 Dataset_A 的原始配置
            type='Dataset_A',
            ...,
            pipeline=transfrom_list
        )
    )
```

### K 折交叉验证数据集

K 折交叉验证通常应用在小规模的数据集中，我们使用 `KFoldDataset` 来支持这一特性. 可以查看 [API](https://mmclassification.readthedocs.io/zh_CN/latest/api/datasets.html#dataset-wrappers) 获取更多接口信息。

比如，以下是在 Imagnet-1k 数据集上使用 `KFoldDataset` 的配置文件样例：

```python
train_dataloader = dict(
    batch_size=64,
    num_workers=5,
    dataset=dict(
        type='KFoldDataset',
        dataset=dict(
            type="ImageNet",
            data_root='data/imagenet',
            ann_file='meta/train.txt',
            data_prefix='train',
            pipeline=train_pipeline),
        # 修改 `fold` 以使用不同的拆分。
        # 对于 5 折交叉验证，需要执行五个实验（fold=0，fold=1，fold=2，...）
        fold=0,
        num_splits=5,
        # seed = 1,  # 如果设置种子，在分割数据集之前打乱样本。
    ),
    sampler=dict(type='DefaultSampler', shuffle=True),
    persistent_workers=True,
)

val_dataloader = dict(
    batch_size=64,
    num_workers=5,
    dataset=dict(
        type='KFoldDataset',
        dataset=dict(
            type="ImageNet",
            data_root='data/imagenet',
            ann_file='meta/train.txt',
            data_prefix='train', # 对于 K-Fold 交叉验证，所有图像都应该放在同一个文件夹中。
            pipeline=test_pipeline),
        # train/val/test set 中的所有参数都需要相同，例如 fold、num_splits 和 seed
        fold=0,
        num_splits=5,
        # seed=1,
        # test_mode=True,
        # `test_mode` 将在 `apis/train.py` 和 `tools/test.py` 中自动设置为 True，您也可以显式指定它。
    )
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True,
)
test_dataloader = val_dataloader
```

`KFoldDataset` 需要配合 `tools/kfolder-cross-vaild.py` 同时使用，如以下常用命令:

开启一个5折交叉验证实验:

```python
python tools/kfold-cross-valid.py $CONFIG --num-splits 5
```

从中断的 5 折交叉验证实验中恢复:

```python
python tools/kfold-cross-valid.py $CONFIG --num-splits 5 --resume-from work_dirs/fold2/latest.pth
```

总结 5 折交叉验证:

```python
python tools/kfold-cross-valid.py $CONFIG --num-splits 5 --summary
```

```note
训练前需要将原本的 train 以及 val 数据集合并在一起（放在同一个目录下或者同一个标注文件中）。
```
