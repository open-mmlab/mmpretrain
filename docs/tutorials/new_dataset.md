# Tutorial 2: Adding New Dataset

## Customize datasets by reorganizing data

### Reorganize dataset to existing format

The simplest way is to convert your dataset to existing dataset formats (i.e. ImageFolderDataset) and use it in your config file.

The ImageFolderDataset can differentiate classes by folders. The directory of an example training data is as follows:

```
images
├── ...
├── train
│   ├── class_0
│   │   ├── class_0_10026.JPEG
│   │   ├── class_0_10027.JPEG
│   │   ├── ...
│   ├── ...
│   ├── class_N
│   │   ├── class_N_999.JPEG
│   │   ├── class_N_9993.JPEG
│   │   ├── ...
```

> If the name of a folder is not specified in the `classes` argument of the config file, that folder will be skipped.
> This allows to train with a subset of the classes in an existing dataset.

Additionally, you can provide an annotation list. Each line of the list contrains a filename and its corresponding ground-truth labels. The format is as follows:

```
class_0_10300.JPEG 0
class_2_10123.JPEG 2
class_1_1273.JPEG 1
class_N_91273.JPEG N
```

> The files are expected to be located at the path resulting from joining `data_prefix` and each filename. The filenames can contain parent folders.

> The value of ground-truth labels should fall in range `[0, num_classes - 1]`.

Once you have your data reorganized in one of these formats, you can use it in your config file as follows:

```python
# No annotation list
dataset_A_train = dict(
    type='ImageFolderDataset',
    data_prefix='images/train',
    classes=('class_0', 'class_1', 'class_N'),
    pipeline=train_pipeline
)
```

```python
# With annotation list
dataset_A_val = dict(
    type='ImageFolderDataset',
    data_prefix='images/val',
    classes=('class_0', 'class_1', 'class_N'),
    ann_file='image_list.txt',
    pipeline=val_pipeline
)
```

## Customize by creating a custom Dataset class.

You can write a new Dataset class inherited from `BaseDataset`, and overwrite `load_annotations(self)`,
like [CIFAR10](https://github.com/open-mmlab/mmclassification/blob/master/mmcls/datasets/cifar.py) and [ImageNet](https://github.com/open-mmlab/mmclassification/blob/master/mmcls/datasets/imagenet.py).
Typically, this function returns a list, where each sample is a dict, containing necessary data informations, e.g., `img` and `gt_label`.

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
class MyDataset(BaseDataset):

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

Then in the config, to use `Filelist` you can modify the config as the following

```python
dataset_A_train = dict(
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
You may refer to [source code](../../mmcls/datasets/dataset_wrappers.py) for details.
