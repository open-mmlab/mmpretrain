# Tutorial 3: Customize Dataset

We support many common public datasets for image classification task, you can find them in
[this page](https://mmclassification.readthedocs.io/en/latest/api/datasets.html).

In this section, we demonstrate how to [use your own dataset](#use-your-own-dataset)
and [use dataset wrapper](#use-dataset-wrapper).

## Use your own dataset

### Reorganize dataset to existing format

The simplest way to use your own dataset is to convert it to existing dataset formats.

For multi-class classification task, we recommend to use the format of
[`CustomDataset`](https://mmclassification.readthedocs.io/en/latest/api/datasets.html#mmcls.datasets.CustomDataset).

The `CustomDataset` supports two kinds of format:

1. An annotation file is provided, and each line indicates a sample image.

   The sample images can be organized in any structure, like:

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

   And an annotation file records all paths of samples and corresponding
   category index. The first column is the image path relative to the folder
   (in this example, `train`) and the second column is the index of category:

   ```
   folder_1/xxx.png 0
   folder_1/xxy.png 1
   123.png 1
   nsdf3.png 2
   ...
   ```

   ```{note}
   The value of the category indices should fall in range `[0, num_classes - 1]`.
   ```

2. The sample images are arranged in the special structure:

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

   In this case, you don't need provide annotation file, and all images in the directory `cat` will be
   recognized as samples of `cat`.

Usually, we will split the whole dataset to three sub datasets: `train`, `val`
and `test` for training, validation and test. And **every** sub dataset should
be organized as one of the above structures.

For example, the whole dataset is as below (using the first structure):

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

And in your config file, you can modify the `data` field as below:

```python
...
dataset_type = 'CustomDataset'
classes = ['cat', 'bird', 'dog']  # The category names of your dataset

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

### Create a new dataset class

You can write a new dataset class inherited from `BaseDataset`, and overwrite `load_annotations(self)`,
like [CIFAR10](https://github.com/open-mmlab/mmclassification/blob/master/mmcls/datasets/cifar.py) and
[CustomDataset](https://github.com/open-mmlab/mmclassification/blob/master/mmcls/datasets/custom.py).

Typically, this function returns a list, where each sample is a dict, containing necessary data information,
e.g., `img` and `gt_label`.

Assume we are going to implement a `Filelist` dataset, which takes filelists for both training and testing.
The format of annotation list is as follows:

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
    ann_file='image_list.txt',
    pipeline=train_pipeline
)
```

## Use dataset wrapper

The dataset wrapper is a kind of class to change the behavior of dataset class, such as repeat the dataset or
re-balance the samples of different categories.

### Repeat dataset

We use `RepeatDataset` as wrapper to repeat the dataset. For example, suppose the original dataset is
`Dataset_A`, to repeat it, the config looks like the following

```python
data = dict(
    train = dict(
        type='RepeatDataset',
        times=N,
        dataset=dict(  # This is the original config of Dataset_A
            type='Dataset_A',
            ...
            pipeline=train_pipeline
        )
    )
    ...
)
```

### Class balanced dataset

We use `ClassBalancedDataset` as wrapper to repeat the dataset based on category frequency. The dataset to
repeat needs to implement method `get_cat_ids(idx)` to support `ClassBalancedDataset`. For example, to repeat
`Dataset_A` with `oversample_thr=1e-3`, the config looks like the following

```python
data = dict(
    train = dict(
        type='ClassBalancedDataset',
        oversample_thr=1e-3,
        dataset=dict(  # This is the original config of Dataset_A
            type='Dataset_A',
            ...
            pipeline=train_pipeline
        )
    )
    ...
)
```

You may refer to [API reference](https://mmclassification.readthedocs.io/en/latest/api/datasets.html#mmcls.datasets.ClassBalancedDataset) for details.
