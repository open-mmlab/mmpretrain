# Tutorial 3: Adding New Dataset

## Customize datasets by reorganizing data

### Reorganize dataset to existing format

The simplest way is to convert your dataset to existing dataset formats (ImageNet).

For training, it differentiates classes by folders. The directory of training data is as follows:

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

For validation, we provide a annotation list. Each line of the list contrains a filename and its corresponding ground-truth labels. The format is as follows:

```
ILSVRC2012_val_00000001.JPEG 65
ILSVRC2012_val_00000002.JPEG 970
ILSVRC2012_val_00000003.JPEG 230
ILSVRC2012_val_00000004.JPEG 809
ILSVRC2012_val_00000005.JPEG 516
```

Note: The value of ground-truth labels should fall in range `[0, num_classes - 1]`.

### An example of customized dataset

You can write a new Dataset class inherited from `BaseDataset`, and overwrite `load_annotations(self)`,
like [CIFAR10](https://github.com/open-mmlab/mmclassification/blob/master/mmcls/datasets/cifar.py) and [ImageNet](https://github.com/open-mmlab/mmclassification/blob/master/mmcls/datasets/imagenet.py).
Typically, this function returns a list, where each sample is a dict, containing necessary data information, e.g., `img` and `gt_label`.

Assume we are going to implement a `Filelist` dataset, which takes filelists for both training and testing. The format of annotation list is as follows:

```
000001.jpg 0
000002.jpg 1
```

We can create a new dataset in `mmcls/datasets/filelist.py` to load the data.

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
    pipeline=train_pipeline
)
```

## Customize datasets by mixing dataset

MMClassification also supports to mix dataset for training.
Currently it supports to concat and repeat datasets.

### Repeat dataset

We use `RepeatDataset` as wrapper to repeat the dataset. For example, suppose the original dataset is `Dataset_A`, to repeat it, the config looks like the following

```python
dataset_A_train = dict(
        type='RepeatDataset',
        times=N,
        dataset=dict(  # This is the original config of Dataset_A
            type='Dataset_A',
            ...
            pipeline=train_pipeline
        )
    )
```

### Class balanced dataset

We use `ClassBalancedDataset` as wrapper to repeat the dataset based on category
frequency. The dataset to repeat needs to instantiate function `self.get_cat_ids(idx)`
to support `ClassBalancedDataset`.
For example, to repeat `Dataset_A` with `oversample_thr=1e-3`, the config looks like the following

```python
dataset_A_train = dict(
        type='ClassBalancedDataset',
        oversample_thr=1e-3,
        dataset=dict(  # This is the original config of Dataset_A
            type='Dataset_A',
            ...
            pipeline=train_pipeline
        )
    )
```

You may refer to [source code](https://github.com/open-mmlab/mmclassification/tree/master/mmcls/datasets/dataset_wrappers.py) for details.
