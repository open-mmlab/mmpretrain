# 教程 4：如何设计数据处理流程

## 设计数据流水线

按照典型的用法，我们通过 `Dataset` 和 `DataLoader` 来使用多个 worker 进行数据加
载。对 `Dataset` 的索引操作将返回一个与模型的 `forward` 方法的参数相对应的字典。

数据流水线和数据集在这里是解耦的。通常，数据集定义如何处理标注文件，而数据流水
线定义所有准备数据字典的步骤。流水线由一系列操作组成。每个操作都将一个字典作为
输入，并输出一个字典。

这些操作分为数据加载，预处理和格式化。

这里使用 ResNet-50 在 ImageNet 数据集上的数据流水线作为示例。

```python
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=256),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
```

对于每个操作，我们列出了添加、更新、删除的相关字典字段。在流水线的最后，我们使
用 `Collect` 仅保留进行模型 `forward` 方法所需的项。

### 数据加载

`LoadImageFromFile` - 从文件中加载图像

- 添加：img, img_shape, ori_shape

默认情况下，`LoadImageFromFile` 将会直接从硬盘加载图像，但对于一些效率较高、规
模较小的模型，这可能会导致 IO 瓶颈。MMCV 支持多种数据加载后端来加速这一过程。例
如，如果训练设备上配置了 [memcached](https://memcached.org/)，那么我们按照如下
方式修改配置文件。

```
memcached_root = '/mnt/xxx/memcached_client/'
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args=dict(
            backend='memcached',
            server_list_cfg=osp.join(memcached_root, 'server_list.conf'),
            client_cfg=osp.join(memcached_root, 'client.conf'))),
]
```

更多支持的数据加载后端，可以参见 [mmcv.fileio.FileClient](https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py)。

### 预处理

`Resize` - 缩放图像尺寸

- 添加：scale, scale_idx, pad_shape, scale_factor, keep_ratio
- 更新：img, img_shape

`RandomFlip` - 随机翻转图像

- 添加：flip, flip_direction
- 更新：img

`RandomCrop` - 随机裁剪图像

- 更新：img, pad_shape

`Normalize` - 图像数据归一化

- 添加：img_norm_cfg
- 更新：img

### 格式化

`ToTensor` - 转换（标签）数据至 `torch.Tensor`

- 更新：根据参数 `keys` 指定

`ImageToTensor` - 转换图像数据至 `torch.Tensor`

- 更新：根据参数 `keys` 指定

`Collect` - 保留指定键值

- 删除：除了参数 `keys` 指定以外的所有键值对

## 扩展及使用自定义流水线

1. 编写一个新的数据处理操作，并放置在 `mmcls/datasets/pipelines/` 目录下的任何
   一个文件中，例如 `my_pipeline.py`。这个类需要重载 `__call__` 方法，接受一个
   字典作为输入，并返回一个字典。

    ```python
    from mmcls.datasets import PIPELINES

    @PIPELINES.register_module()
    class MyTransform(object):

        def __call__(self, results):
            # 对 results['img'] 进行变换操作
            return results
    ```

2. 在 `mmcls/datasets/pipelines/__init__.py` 中导入这个新的类。

    ```python
    ...
    from .my_pipeline import MyTransform

    __all__ = [
        ..., 'MyTransform'
    ]
    ```

3. 在数据流水线的配置中添加这一操作。

    ```python
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='RandomResizedCrop', size=224),
        dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
        dict(type='MyTransform'),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='ImageToTensor', keys=['img']),
        dict(type='ToTensor', keys=['gt_label']),
        dict(type='Collect', keys=['img', 'gt_label'])
    ]
    ```

## 流水线可视化

设计好数据流水线后，可以使用[可视化工具](../tools/visualization.md)查看具体的效果。
