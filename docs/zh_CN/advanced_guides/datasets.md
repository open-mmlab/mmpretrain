# 添加新数据集

用户可以编写一个继承自 [BasesDataset](https://mmpretrain.readthedocs.io/zh_CN/latest/_modules/mmpretrain/datasets/base_dataset.html#BaseDataset) 的新数据集类，并重载 `load_data_list(self)` 方法，类似 [CIFAR10](https://github.com/open-mmlab/mmpretrain/blob/main/mmpretrain/datasets/cifar.py) 和 [ImageNet](https://github.com/open-mmlab/mmpretrain/blob/main/mmpretrain/datasets/imagenet.py)。

通常，此方法返回一个包含所有样本的列表，其中的每个样本都是一个字典。字典中包含了必要的数据信息，例如 `img` 和 `gt_label`。

假设我们将要实现一个 `Filelist` 数据集，该数据集将使用文件列表进行训练和测试。注释列表的格式如下：

```text
000001.jpg 0
000002.jpg 1
...
```

## 1. 创建数据集类

我们可以在 `mmpretrain/datasets/filelist.py` 中创建一个新的数据集类以加载数据。

```python
from mmpretrain.registry import DATASETS
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

## 2. 添加到库

将新的数据集类加入到 `mmpretrain/datasets/__init__.py` 中：

```python
from .base_dataset import BaseDataset
...
from .filelist import Filelist

__all__ = [
    'BaseDataset', ... ,'Filelist'
]
```

### 3. 修改相关配置文件

然后在配置文件中，为了使用 `Filelist`，用户可以按以下方式修改配置

```python
train_dataloader = dict(
    ...
    dataset=dict(
        type='Filelist',
        ann_file='image_list.txt',
        pipeline=train_pipeline,
    )
)
```

所有继承 [`BaseDataset`](https://github.com/open-mmlab/mmpretrain/blob/main/mmpretrain/datasets/base_dataset.py) 的数据集类都具有**懒加载**以及**节省内存**的特性，可以参考相关文档 {external+mmengine:doc}`BaseDataset <advanced_tutorials/basedataset>`。

```{note}
如果数据样本时获取的字典中，只包含了 'img_path' 不包含 'img'， 则在 pipeline 中必须包含 'LoadImgFromFile'。
```
