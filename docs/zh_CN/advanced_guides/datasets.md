# 教程 3：如何自定义数据集

我们支持许多常用的图像分类领域公开数据集，你可以在
[此页面](https://mmclassification.readthedocs.io/zh_CN/latest/api/datasets.html)中找到它们。

在本节中，我们将介绍如何[使用自己的数据集](#使用自己的数据集)以及如何[使用数据集包装](#使用数据集包装)。

## 使用自己的数据集

### 将数据集重新组织为已有格式

想要使用自己的数据集，最简单的方法就是将数据集转换为现有的数据集格式。

对于多分类任务，我们推荐使用 [`CustomDataset`](https://mmclassification.readthedocs.io/zh_CN/latest/api/datasets.html#mmcls.datasets.CustomDataset) 格式。

`CustomDataset` 支持两种类型的数据格式：

1. 提供一个标注文件，其中每一行表示一张样本图片。

   样本图片可以以任意的结构进行组织，比如：

   ```
   train/
   ├── folder_1
   │   ├── xxx.png
   │   ├── xxy.png
   │   └── ...
   ├── 123.png
   ├── nsdf3.png
   └── ...
   ```

   而标注文件则记录了所有样本图片的文件路径以及相应的类别序号。其中第一列表示图像
   相对于主目录（本例中为 `train` 目录）的路径，第二列表示类别序号：

   ```
   folder_1/xxx.png 0
   folder_1/xxy.png 1
   123.png 1
   nsdf3.png 2
   ...
   ```

   ```{note}
   类别序号的值应当属于 `[0, num_classes - 1]` 范围。
   ```

2. 将所有样本文件按如下结构进行组织：

   ```
   train/
   ├── cat
   │   ├── xxx.png
   │   ├── xxy.png
   │   └── ...
   │       └── xxz.png
   ├── bird
   │   ├── bird1.png
   │   ├── bird2.png
   │   └── ...
   └── dog
       ├── 123.png
       ├── nsdf3.png
       ├── ...
       └── asd932_.png
   ```

   这种情况下，你不需要提供标注文件，所有位于 `cat` 目录下的图片文件都会被视为 `cat` 类别的样本。

通常而言，我们会将整个数据集分为三个子数据集：`train`，`val` 和 `test`，分别用于训练、验证和测试。**每一个**子
数据集都需要被组织成如上的一种结构。

举个例子，完整的数据集结构如下所示（使用第一种组织结构）：

```
mmclassification
└── data
    └── my_dataset
        ├── meta
        │   ├── train.txt
        │   ├── val.txt
        │   └── test.txt
        ├── train
        ├── val
        └── test
```

之后在你的配置文件中，可以修改其中的 `data` 字段为如下格式：

```python
...
dataset_type = 'CustomDataset'
classes = ['cat', 'bird', 'dog']  # 数据集中各类别的名称

data = dict(
    train=dict(
        type=dataset_type,
        data_prefix='data/my_dataset/train',
        ann_file='data/my_dataset/meta/train.txt',
        classes=classes,
        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        data_prefix='data/my_dataset/val',
        ann_file='data/my_dataset/meta/val.txt',
        classes=classes,
        pipeline=test_pipeline
    ),
    test=dict(
        type=dataset_type,
        data_prefix='data/my_dataset/test',
        ann_file='data/my_dataset/meta/test.txt',
        classes=classes,
        pipeline=test_pipeline
    )
)
...
```

### 创建一个新的数据集类

用户可以编写一个继承自 `BasesDataset` 的新数据集类，并重载 `load_annotations(self)` 方法，
类似 [CIFAR10](https://github.com/open-mmlab/mmclassification/blob/master/mmcls/datasets/cifar.py)
和 [ImageNet](https://github.com/open-mmlab/mmclassification/blob/master/mmcls/datasets/imagenet.py)。

通常，此方法返回一个包含所有样本的列表，其中的每个样本都是一个字典。字典中包含了必要的数据信息，例如 `img` 和 `gt_label`。

假设我们将要实现一个 `Filelist` 数据集，该数据集将使用文件列表进行训练和测试。注释列表的格式如下：

```
000001.jpg 0
000002.jpg 1
```

我们可以在 `mmcls/datasets/filelist.py` 中创建一个新的数据集类以加载数据。

```python
import mmcv
import numpy as np

from .builder import DATASETS
from .base_dataset import BaseDataset


@DATASETS.register_module()
class Filelist(BaseDataset):

    def load_annotations(self):
        assert isinstance(self.ann_file, str)

        data_infos = []
        with open(self.ann_file) as f:
            samples = [x.strip().split(' ') for x in f.readlines()]
            for filename, gt_label in samples:
                info = {'img_prefix': self.data_prefix}
                info['img_info'] = {'filename': filename}
                info['gt_label'] = np.array(gt_label, dtype=np.int64)
                data_infos.append(info)
            return data_infos

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
    ann_file = 'image_list.txt',
    pipeline=train_pipeline
)
```

## 使用数据集包装

数据集包装是一种可以改变数据集类行为的类，比如将数据集中的样本进行重复，或是将不同类别的数据进行再平衡。

### 重复数据集

我们使用 `RepeatDataset` 作为一个重复数据集的封装。举个例子，假设原始数据集是 `Dataset_A`，为了重复它，我们需要如下的配置文件：

```python
data = dict(
    train=dict(
        type='RepeatDataset',
        times=N,
        dataset=dict(  # 这里是 Dataset_A 的原始配置
            type='Dataset_A',
            ...
            pipeline=train_pipeline
        )
    )
    ...
)
```

### 类别平衡数据集

我们使用 `ClassBalancedDataset` 作为根据类别频率对数据集进行重复采样的封装类。进行重复采样的数据集需要实现函数 `self.get_cat_ids(idx)` 以支持 `ClassBalancedDataset`。

举个例子，按照 `oversample_thr=1e-3` 对 `Dataset_A` 进行重复采样，需要如下的配置文件：

```python
data = dict(
    train = dict(
        type='ClassBalancedDataset',
        oversample_thr=1e-3,
        dataset=dict(  # 这里是 Dataset_A 的原始配置
            type='Dataset_A',
            ...
            pipeline=train_pipeline
        )
    )
    ...
)
```

更加具体的细节，请参考 [API 文档](https://mmclassification.readthedocs.io/zh_CN/latest/api/datasets.html#mmcls.datasets.ClassBalancedDataset)。
