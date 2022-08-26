# Adding New Dataset

## Add New Dataset Class

You can write a new Dataset class inherited from `BaseDataset`, and overwrite `load_data_list(self)`,
like [CIFAR10](https://github.com/open-mmlab/mmclassification/blob/master/mmcls/datasets/cifar.py) and [ImageNet](https://github.com/open-mmlab/mmclassification/blob/master/mmcls/datasets/imagenet.py).
Typically, this function returns a list, where each sample is a dict, containing necessary data information, e.g., `img` and `gt_label`.

Assume we are going to implement a `Filelist` dataset, which takes filelists for both training and testing. The format of annotation list is as follows:

```text
000001.jpg 0
000002.jpg 1
```

### Create Dataset Class

We can create a new dataset in `mmcls/datasets/filelist.py` to load the data.

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

### Import Dataset Class (MMCls as framework)

And add this dataset class in `mmcls/datasets/__init__.py`

```python
from .base_dataset import BaseDataset
...
from .filelist import Filelist

__all__ = [
    'BaseDataset', ... ,'Filelist'
]
```

Then in the config, to use `Filelist` you can modify the config as the following

```python
train = dict(
    type='Filelist',
    ann_file = 'image_list.txt',
    pipeline=transfrom_list
)
```

### Import Dataset Class (MMCls as 3rd-party package)

If you want to use the MMClassifiaction as a 3rd-party package, you can import your class in your config towards this [guide](TODO:).

```python
custom_imports = dict(imports=['my_package.my_module'], allow_failed_imports=False)

train = dict(
    type='Filelist',
    ann_file = 'image_list.txt',
    pipeline=transfrom_list
)
```

Both `CustomData` and custom dataset classes inherit from [`BaseDataset`](https://github.com/open-mmlab/mmclassification/blob/1.x/mmcls/datasets/base_dataset.py), which basic usage, **lazy loading** and **memory saving** features can refer to related documents [mmengine.basedataset](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/tutorials/basedataset.md).

```note
If the dictionary of the data sample contains 'img_path' but not 'img', then 'LoadImgFromFile' transform must be added in the pipeline.
```
