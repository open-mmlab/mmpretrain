# 自定义数据处理流程

## 数据流水线的设计

在[添加新数据集教程](./datasets.md)中，我们知道数据集类使用 `load_data_list` 方法初始化整个数据集，并将每个样本的信息保存到字典中。

通常，为了节省内存使用，我们只在 `load_data_list` 中加载图像路径和标签，并在使用它们时加载完整的图像内容。此外，我们可能希望在训练时挑选样本时进行一些随机数据增强。MMClassification 几乎所有的数据加载、预处理和格式化操作都可以在数据流水线中进行配置。

数据流水线是指从数据集中索引样本后如何对样本字典进行处理。它由一系列数据转换组成。每个数据转换以一个字典作为输入，对其进行处理，并为下一个数据转换输出一个字典。

这里使用 ResNet-50 在 ImageNet 数据集上的数据流水线作为示例。

```python
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=224),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackClsInputs'),
]
```

您可以在[数据变换文档](mmcls.datasets.transforms)中找到 MMClassification 中所有可用的数据变换。

## 修改训练/测试流程

MMClassification 中的数据流水线非常灵活。您可以在配置文件中控制数据预处理的几乎每一个步骤，但另一方面，您可能会因面对如此多的选项而感到困惑。

以下是图像分类任务的常见实践和指南：

### 数据加载

在数据流水线的开头，我们通常需要从文件路径中加载图像数据。[`LoadImageFromFile`](mmcv.transforms.LoadImageFromFile)是常用于执行此任务的数据变换。

```python
train_pipeline = [
    dict(type='LoadImageFromFile'),
    ...
]
```

如果您想从具有特殊格式或特殊位置的文件中加载数据，可以参考[添加新的数据转换方法](#添加新的数据转换方法)来实现一个新的数据加载方法，并将其添加到数据流水线的开头。

### 增强和其他处理

在训练期间，通常需要进行数据增强以避免过拟合。在测试期间，我们还需要进行一些数据处理，如图像的缩放和裁剪。这些数据变换将放置在数据加载之后。

这里是一个简单的数据增强示例，它将随机调整输入图像的大小和裁剪到指定的尺度，并有概率对图像进行水平翻转。

```python
train_pipeline = [
    ...
    dict(type='RandomResizedCrop', scale=224),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    ...
]
```

这是一个在训练 [Swin-Transformer](../papers/swin_transformer.md) 中使用大量数据增强的示例。为了与官方实现保持一致，它指定了 `pillow` 作为 `resize` 的后端和 `bicubic` 作为 `resize` 的算法。
此外，它添加了 [`RandAugment`](mmcls.datasets.transforms.RandAugment) 和 [`RandomErasing`](mmcls.datasets.transforms.RandomErasing)  作为额外的数据增强方法。

这个配置指定了数据增强的每个细节，您可以将其简单地复制到自己的配置文件中，以应用 Swin-Transformer 中的数据增强流程。

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
通常，数据流水线中的数据增强部分仅处理在图像层面的变化，这与图像归一化或者 mixup/cutmix 方法的转化不同。这是因为我们可以在
批处理数据时对图像归一化和 mixup/cutmix 操作来加速数据处理。配置图像归一化和 mixup/cutmix，请使用[data preprocessor]
(mmcls.models.utils.data_preprocessor)。
```

### 格式化

格式化是从数据信息字典中收集训练数据，并将这些数据转换为适合模型训练的格式。

在大多数情况下，您可以简单地使用  [`PackClsInputs`](mmcls.datasets.transforms.PackClsInputs)，
它会将 `NumPy` 数组格式的图像转换为 `PyTorch` 的张量数据，并将真实类别信息和其他元信息打包为 [`ClsDataSample`](mmcls.structures.ClsDataSample) 格式。

```python
train_pipeline = [
    ...
    dict(type='PackClsInputs'),
]
```

## 添加新的数据转换方法

1. 在文件中编写一个新的数据转换方法，例如 `my_transform.py` 文件，并将其放置在文件夹 `mmcls/datasets/transforms/` 下。
   数据变换类需要继承 [`mmcv.transforms.BaseTransform`](mmcv.transforms.BaseTransform) 类并重写 `transform` 方法，
   该方法以字典作为输入并返回一个字典。

   ```python
   from mmcv.transforms import BaseTransform
   from mmcls.datasets import TRANSFORMS

   @TRANSFORMS.register_module()
   class MyTransform(BaseTransform):

       def transform(self, results):
           # Modify the data information dict `results`.
           return results
   ```

2. 将其导入到 `mmcls/datasets/transforms/__init__.py` 中。

   ```python
   ...
   from .my_transform import MyTransform

   __all__ = [
       ..., 'MyTransform'
   ]
   ```

3. 在配置文件中使用

   ```python
   train_pipeline = [
       ...
       dict(type='MyTransform'),
       ...
   ]
   ```

## 数据流水线可视化

在设计好数据流水线之后，您可以使用[可视化工具](../useful_tools/dataset_visualization.md)来查看数据流水线的执行结果。
