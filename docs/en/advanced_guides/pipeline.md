# Customize Data Pipeline

## Design of Data pipelines

In the [new dataset tutorial](./datasets.md), we know that the dataset class use the `load_data_list` method
to initialize the entire dataset, and we save the information of every sample to a dict.

Usually, to save memory usage, we only load image paths and labels in the `load_data_list`, and load full
image content when we use them. Moreover, we may want to do some random data augmentation during picking
samples when training. Almost all data loading, pre-processing, and formatting operations can be configured in
MMPretrain by the **data pipeline**.

The data pipeline means how to process the sample dict when indexing a sample from the dataset. And it
consists of a sequence of data transforms. Each data transform takes a dict as input, processes it, and outputs a
dict for the next data transform.

Here is a data pipeline example for ResNet-50 training on ImageNet.

```python
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=224),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]
```

All available data transforms in MMPretrain can be found in the [data transforms docs](mmpretrain.datasets.transforms).

## Modify the training/test pipeline

The data pipeline in MMPretrain is pretty flexible. You can control almost every step of the data
preprocessing from the config file, but on the other hand, you may be confused facing so many options.

Here is a common practice and guidance for image classification tasks.

### Loading

At the beginning of a data pipeline, we usually need to load image data from the file path.
[`LoadImageFromFile`](mmcv.transforms.LoadImageFromFile) is commonly used to do this task.

```python
train_pipeline = [
    dict(type='LoadImageFromFile'),
    ...
]
```

If you want to load data from files with special formats or special locations, you can [implement a new loading
transform](#add-new-data-transforms) and add it at the beginning of the data pipeline.

### Augmentation and other processing

During training, we usually need to do data augmentation to avoid overfitting. During the test, we also need to do
some data processing like resizing and cropping. These data transforms will be placed after the loading process.

Here is a simple data augmentation recipe example. It will randomly resize and crop the input image to the
specified scale, and randomly flip the image horizontally with probability.

```python
train_pipeline = [
    ...
    dict(type='RandomResizedCrop', scale=224),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    ...
]
```

Here is a heavy data augmentation recipe example used in [Swin-Transformer](../papers/swin_transformer.md)
training. To align with the official implementation, it specified `pillow` as the resize backend and `bicubic`
as the resize algorithm. Moreover, it added [`RandAugment`](mmpretrain.datasets.transforms.RandAugment) and
[`RandomErasing`](mmpretrain.datasets.transforms.RandomErasing) as extra data augmentation method.

This configuration specified every detail of the data augmentation, and you can simply copy it to your own
config file to apply the data augmentations of the Swin-Transformer.

```python
bgr_mean = [103.53, 116.28, 123.675]
bgr_std = [57.375, 57.12, 58.395]

train_pipeline = [
    ...
    dict(type='RandomResizedCrop', scale=224, backend='pillow', interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='RandAugment',
        policies='timm_increasing',
        num_policies=2,
        total_level=10,
        magnitude_level=9,
        magnitude_std=0.5,
        hparams=dict(
            pad_val=[round(x) for x in bgr_mean], interpolation='bicubic')),
    dict(
        type='RandomErasing',
        erase_prob=0.25,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=1 / 3,
        fill_color=bgr_mean,
        fill_std=bgr_std),
    ...
]
```

```{note}
Usually, the data augmentation part in the data pipeline handles only image-wise transforms, but not transforms
like image normalization or mixup/cutmix. It's because we can do image normalization and mixup/cutmix on batch data
to accelerate. To configure image normalization and mixup/cutmix, please use the [data preprocessor](mmpretrain.models.utils.data_preprocessor).
```

### Formatting

The formatting is to collect training data from the data information dict and convert these data to
model-friendly format.

In most cases, you can simply use [`PackInputs`](mmpretrain.datasets.transforms.PackInputs), and it will
convert the image in NumPy array format to PyTorch tensor, and pack the ground truth categories information and
other meta information as a [`DataSample`](mmpretrain.structures.DataSample).

```python
train_pipeline = [
    ...
    dict(type='PackInputs'),
]
```

## Add new data transforms

1. Write a new data transform in any file, e.g., `my_transform.py`, and place it in
   the folder `mmpretrain/datasets/transforms/`. The data transform class needs to inherit
   the [`mmcv.transforms.BaseTransform`](mmcv.transforms.BaseTransform) class and override
   the `transform` method which takes a dict as input and returns a dict.

   ```python
   from mmcv.transforms import BaseTransform
   from mmpretrain.registry import TRANSFORMS

   @TRANSFORMS.register_module()
   class MyTransform(BaseTransform):

       def transform(self, results):
           # Modify the data information dict `results`.
           return results
   ```

2. Import the new class in the `mmpretrain/datasets/transforms/__init__.py`.

   ```python
   ...
   from .my_transform import MyTransform

   __all__ = [
       ..., 'MyTransform'
   ]
   ```

3. Use it in config files.

   ```python
   train_pipeline = [
       ...
       dict(type='MyTransform'),
       ...
   ]
   ```

## Pipeline visualization

After designing data pipelines, you can use the [visualization tools](../useful_tools/dataset_visualization.md) to view the performance.
