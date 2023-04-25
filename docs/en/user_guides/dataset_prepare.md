# Prepare Dataset

## CustomDataset

[`CustomDataset`](mmpretrain.datasets.CustomDataset) is a general dataset class for you to use your own datasets. To use `CustomDataset`, you need to organize your dataset files according to the following two formats:

### Subfolder Format

In this format, you only need to re-organize your dataset folder and place all samples in one folder without
creating any annotation files.

For supervised tasks (with `with_label=True`), we use the name of sub-folders as the categories names, as
shown in the below example, `class_x` and `class_y` will be recognized as the categories names.

```text
data_prefix/
├── class_x
│   ├── xxx.png
│   ├── xxy.png
│   └── ...
│       └── xxz.png
└── class_y
    ├── 123.png
    ├── nsdf3.png
    ├── ...
    └── asd932_.png
```

For unsupervised tasks (with `with_label=False`), we directly load all sample files under the specified folder:

```text
data_prefix/
├── folder_1
│   ├── xxx.png
│   ├── xxy.png
│   └── ...
├── 123.png
├── nsdf3.png
└── ...
```

Assume you want to use it as the training dataset, and the below is the configurations in your config file.

```python
train_dataloader = dict(
    ...
    # Training dataset configurations
    dataset=dict(
        type='CustomDataset',
        data_prefix='path/to/data_prefix',
        with_label=True,   # or False for unsupervised tasks
        pipeline=...
    )
)
```

```{note}
If you want to use this format, do not specify `ann_file`, or specify `ann_file=''`.

And please note that the subfolder format requires a folder scanning which may cause a slower initialization,
especially for large datasets or slow file IO.
```

### Text Annotation File Format

In this format, we use a text annotation file to store image file paths and the corespondding category
indices.

For supervised tasks (with `with_label=True`), the annotation file should include the file path and the
category index of one sample in one line and split them by a space, as below:

All these file paths can be absolute paths, or paths relative to the `data_prefix`.

```text
folder_1/xxx.png 0
folder_1/xxy.png 1
123.png 4
nsdf3.png 3
...
```

```{note}
The index numbers of categories start from 0. And the value of ground-truth labels should fall in range `[0, num_classes - 1]`.

In addition, please use the `classes` field in the dataset settings to specify the name of every category.
```

For unsupervised tasks (with `with_label=False`), the annotation file only need to include the file path of
one sample in one line, as below:

```text
folder_1/xxx.png
folder_1/xxy.png
123.png
nsdf3.png
...
```

Assume the entire dataset folder is as below:

```text
data_root
├── meta
│   ├── test.txt     # The annotation file for the test dataset
│   ├── train.txt    # The annotation file for the training dataset
│   └── val.txt      # The annotation file for the validation dataset.
├── train
│   ├── 123.png
│   ├── folder_1
│   │   ├── xxx.png
│   │   └── xxy.png
│   └── nsdf3.png
├── test
└── val
```

Here is an example dataset settings in config files:

```python
# Training dataloader configurations
train_dataloader = dict(
    dataset=dict(
        type='CustomDataset',
        data_root='path/to/data_root',  # The common prefix of both `ann_flie` and `data_prefix`.
        ann_file='meta/train.txt',      # The path of annotation file relative to the data_root.
        data_prefix='train',            # The prefix of file paths in the `ann_file`, relative to the data_root.
        with_label=True,                # or False for unsupervised tasks
        classes=['A', 'B', 'C', 'D', ...],  # The name of every category.
        pipeline=...,    # The transformations to process the dataset samples.
    )
    ...
)
```

```{note}
For a complete example about how to use the `CustomDataset`, please see [How to Pretrain with Custom Dataset](../notes/pretrain_custom_dataset.md)
```

## ImageNet

ImageNet has multiple versions, but the most commonly used one is [ILSVRC 2012](http://www.image-net.org/challenges/LSVRC/2012/). It can be accessed with the following steps.

1. Register an account and login to the [download page](http://www.image-net.org/download-images).
2. Find download links for ILSVRC2012 and download the following two files
   - ILSVRC2012_img_train.tar (~138GB)
   - ILSVRC2012_img_val.tar (~6.3GB)
3. Untar the downloaded files
4. Re-organize the image files according the format convention of [CustomDataset](#CustomDataset)

```{note}
In MMPretrain, we use the text annotation file format ImageNet in all provided config files. Therefore, to use
the subfolder format, you please set `ann_file=''` in these config files.
```

### Subfolder Format

Re-organize the dataset as below:

```text
data/imagenet/
├── train/
│   ├── n01440764
│   │   ├── n01440764_10026.JPEG
│   │   ├── n01440764_10027.JPEG
│   │   ├── n01440764_10029.JPEG
│   │   ├── n01440764_10040.JPEG
│   │   ├── n01440764_10042.JPEG
│   │   ├── n01440764_10043.JPEG
│   │   └── n01440764_10048.JPEG
│   ├── ...
├── val/
│   ├── n01440764
│   │   ├── ILSVRC2012_val_00000293.JPEG
│   │   ├── ILSVRC2012_val_00002138.JPEG
│   │   ├── ILSVRC2012_val_00003014.JPEG
│   │   └── ...
│   ├── ...
```

And then, you can use the [`ImageNet`](mmpretrain.datasets.ImageNet) dataset with the below configurations:

```python
train_dataloader = dict(
    ...
    # Training dataset configurations
    dataset=dict(
        type='ImageNet',
        data_root='data/imagenet',
        data_prefix='train',
        ann_file='',
        pipeline=...,
    )
)

val_dataloader = dict(
    ...
    # Validation dataset configurations
    dataset=dict(
        type='ImageNet',
        data_root='data/imagenet',
        data_prefix='val',
        ann_file='',
        pipeline=...,
    )
)

test_dataloader = val_dataloader
```

### Text Annotation File Format

You can download and untar the meta data from this [link](https://download.openmmlab.com/mmclassification/datasets/imagenet/meta/caffe_ilsvrc12.tar.gz). And re-organize the dataset as below:

```text
data/imagenet/
├── meta/
│   ├── train.txt
│   ├── test.txt
│   └── val.txt
├── train/
│   ├── n01440764
│   │   ├── n01440764_10026.JPEG
│   │   ├── n01440764_10027.JPEG
│   │   ├── n01440764_10029.JPEG
│   │   ├── n01440764_10040.JPEG
│   │   ├── n01440764_10042.JPEG
│   │   ├── n01440764_10043.JPEG
│   │   └── n01440764_10048.JPEG
│   ├── ...
├── val/
│   ├── ILSVRC2012_val_00000001.JPEG
│   ├── ILSVRC2012_val_00000002.JPEG
│   ├── ILSVRC2012_val_00000003.JPEG
│   ├── ILSVRC2012_val_00000004.JPEG
│   ├── ...
```

And then, you can use the [`ImageNet`](mmpretrain.datasets.ImageNet) dataset with the below configurations:

```python
train_dataloader = dict(
    ...
    # Training dataset configurations
    dataset=dict(
        type='ImageNet',
        data_root='imagenet_folder',
        data_prefix='train/',
        ann_file='meta/train.txt',
        pipeline=...,
    )
)

val_dataloader = dict(
    ...
    # Validation dataset configurations
    dataset=dict(
        type='ImageNet',
        data_root='imagenet_folder',
        data_prefix='val/',
        ann_file='meta/val.txt',
        pipeline=...,
    )
)

test_dataloader = val_dataloader
```

## CIFAR

We support downloading the [`CIFAR10`](mmpretrain.datasets.CIFAR10) and [`CIFAR100`](mmpretrain.datasets.CIFAR100) datasets automatically, and you just need to specify the
download folder in the `data_root` field. And please specify `test_mode=False` / `test_mode=True`
to use training datasets or test datasets.

```python
train_dataloader = dict(
    ...
    # Training dataset configurations
    dataset=dict(
        type='CIFAR10',
        data_root='data/cifar10',
        test_mode=False,
        pipeline=...,
    )
)

val_dataloader = dict(
    ...
    # Validation dataset configurations
    dataset=dict(
        type='CIFAR10',
        data_root='data/cifar10',
        test_mode=True,
        pipeline=...,
    )
)

test_dataloader = val_dataloader
```

## MNIST

We support downloading the [MNIST](mmpretrain.datasets.MNIST) and [Fashion-MNIST](mmpretrain.datasets.FashionMNIST) datasets automatically, and you just need to specify the
download folder in the `data_root` field. And please specify `test_mode=False` / `test_mode=True`
to use training datasets or test datasets.

```python
train_dataloader = dict(
    ...
    # Training dataset configurations
    dataset=dict(
        type='MNIST',
        data_root='data/mnist',
        test_mode=False,
        pipeline=...,
    )
)

val_dataloader = dict(
    ...
    # Validation dataset configurations
    dataset=dict(
        type='MNIST',
        data_root='data/mnist',
        test_mode=True,
        pipeline=...,
    )
)

test_dataloader = val_dataloader
```

## OpenMMLab 2.0 Standard Dataset

In order to facilitate the training of multi-task algorithm models, we unify the dataset interfaces of different tasks. OpenMMLab has formulated the **OpenMMLab 2.0 Dataset Format Specification**. When starting a trainning task, the users can choose to convert their dataset annotation into the specified format, and use the algorithm library of OpenMMLab to perform algorithm training and testing based on the data annotation file.

The OpenMMLab 2.0 Dataset Format Specification stipulates that the annotation file must be in `json` or `yaml`, `yml`, `pickle` or `pkl` format; the dictionary stored in the annotation file must contain `metainfo` and `data_list` fields, The value of `metainfo` is a dictionary, which contains the meta information of the dataset; and the value of `data_list` is a list, each element in the list is a dictionary, the dictionary defines a raw data, each raw data contains a or several training/testing samples.

The following is an example of a JSON annotation file (in this example each raw data contains only one train/test sample):

```
{
    'metainfo':
        {
            'classes': ('cat', 'dog'), # the category index of 'cat' is 0 and 'dog' is 1.
            ...
        },
    'data_list':
        [
            {
                'img_path': "xxx/xxx_0.jpg",
                'gt_label': 0,
                ...
            },
            {
                'img_path': "xxx/xxx_1.jpg",
                'gt_label': 1,
                ...
            },
            ...
        ]
}
```

Assume you want to use the training dataset and the dataset is stored as the below structure:

```text
data
├── annotations
│   ├── train.json
├── train
│   ├── xxx/xxx_0.jpg
│   ├── xxx/xxx_1.jpg
│   ├── ...
```

Build from the following dictionaries:

```python
train_dataloader = dict(
    ...
    dataset=dict(
        type='BaseDataset',
        data_root='data',
        ann_file='annotations/train.json',
        data_prefix='train/',
        pipeline=...,
    )
)
```

## Other Datasets

To find more datasets supported by MMPretrain, and get more configurations of the above datasets, please see the [dataset documentation](mmpretrain.datasets).

To implement your own dataset class for some special formats, please see the [Adding New Dataset](../advanced_guides/datasets.md).

## Dataset Wrappers

The following datawrappers are supported in MMEngine, you can refer to {external+mmengine:doc}`MMEngine tutorial <advanced_tutorials/basedataset>` to learn how to use it.

- {external:py:class}`~mmengine.dataset.ConcatDataset`
- {external:py:class}`~mmengine.dataset.RepeatDataset`
- {external:py:class}`~mmengine.dataset.ClassBalancedDataset`

The MMPretrain also support [KFoldDataset](mmpretrain.datasets.KFoldDataset), please use it with `tools/kfold-cross-valid.py`.
