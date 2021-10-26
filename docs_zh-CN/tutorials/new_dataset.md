# 教程 3：如何添加新数据集

## 通过重新组织数据来自定义数据集

### 将数据集重新组织为已有格式

最简单的方法是将数据集转换为现有的数据集格式 (ImageNet)。

为了训练，根据图片的类别，存放至不同子目录下。训练数据文件夹结构如下所示：

```
imagenet
├── ...
├── train
│   ├── n01440764
│   │   ├── n01440764_10026.JPEG
│   │   ├── n01440764_10027.JPEG
│   │   ├── ...
│   ├── ...
│   ├── n15075141
│   │   ├── n15075141_999.JPEG
│   │   ├── n15075141_9993.JPEG
│   │   ├── ...
```

为了验证，我们提供了一个注释列表。列表的每一行都包含一个文件名及其相应的真实标签。格式如下：

```
ILSVRC2012_val_00000001.JPEG 65
ILSVRC2012_val_00000002.JPEG 970
ILSVRC2012_val_00000003.JPEG 230
ILSVRC2012_val_00000004.JPEG 809
ILSVRC2012_val_00000005.JPEG 516
```

注：真实标签的值应该位于 `[0, 类别数目 - 1]` 之间

### 自定义数据集的示例

用户可以编写一个继承自 `BasesDataset` 的新数据集类，并重载 `load_annotations(self)` 方法，类似 [CIFAR10](https://github.com/open-mmlab/mmclassification/blob/master/mmcls/datasets/cifar.py) 和 [ImageNet](https://github.com/open-mmlab/mmclassification/blob/master/mmcls/datasets/imagenet.py)。


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

## 通过混合数据集来自定义数据集

MMClassification 还支持混合数据集以进行训练。目前支持合并和重复数据集。

### 重复数据集

我们使用 `RepeatDataset` 作为一个重复数据集的封装。举个例子，假设原始数据集是 `Dataset_A`，为了重复它，我们需要如下的配置文件：

```python
dataset_A_train = dict(
        type='RepeatDataset',
        times=N,
        dataset=dict(  # 这里是 Dataset_A 的原始配置
            type='Dataset_A',
            ...
            pipeline=train_pipeline
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
        dataset=dict(  # 这里是 Dataset_A 的原始配置
            type='Dataset_A',
            ...
            pipeline=train_pipeline
        )
    )
```

更加具体的细节，请参考 [源代码](https://github.com/open-mmlab/mmclassification/tree/master/mmcls/datasets/dataset_wrappers.py)。
