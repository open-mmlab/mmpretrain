# Adding New Dataset

You can write a new dataset class inherited from `BaseDataset`, and overwrite `load_data_list(self)`,
like [CIFAR10](https://github.com/open-mmlab/mmpretrain/blob/main/mmpretrain/datasets/cifar.py) and [ImageNet](https://github.com/open-mmlab/mmpretrain/blob/main/mmpretrain/datasets/imagenet.py).
Typically, this function returns a list, where each sample is a dict, containing necessary data information, e.g., `img` and `gt_label`.

Assume we are going to implement a `Filelist` dataset, which takes filelists for both training and testing. The format of annotation list is as follows:

```text
000001.jpg 0
000002.jpg 1
```

## 1. Create Dataset Class

We can create a new dataset in `mmpretrain/datasets/filelist.py` to load the data.

```python
from mmpretrain.registry import DATASETS
from .base_dataset import BaseDataset


@DATASETS.register_module()
class Filelist(BaseDataset):

    def load_data_list(self):
        assert isinstance(self.ann_file, str),

        data_list = []
        with open(self.ann_file) as f:
            samples = [x.strip().split(' ') for x in f.readlines()]
            for filename, gt_label in samples:
                img_path = add_prefix(filename, self.img_prefix)
                info = {'img_path': img_path, 'gt_label': int(gt_label)}
                data_list.append(info)
        return data_list
```

## 2. Add to the package

And add this dataset class in `mmpretrain/datasets/__init__.py`

```python
from .base_dataset import BaseDataset
...
from .filelist import Filelist

__all__ = [
    'BaseDataset', ... ,'Filelist'
]
```

## 3. Modify Related Config

Then in the config, to use `Filelist` you can modify the config as the following

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

All the dataset classes inherit from [`BaseDataset`](https://github.com/open-mmlab/mmpretrain/blob/main/mmpretrain/datasets/base_dataset.py) have **lazy loading** and **memory saving** features, you can refer to related documents of {external+mmengine:doc}`BaseDataset <advanced_tutorials/basedataset>`.

```{note}
If the dictionary of the data sample contains 'img_path' but not 'img', then 'LoadImgFromFile' transform must be added in the pipeline.
```
