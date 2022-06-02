# Tutorial 4: Custom Data Pipelines

## Design of Data pipelines

Following typical conventions, we use `Dataset` and `DataLoader` for data loading
with multiple workers. Indexing `Dataset` returns a dict of data items corresponding to
the arguments of models forward method.

The data preparation pipeline and the dataset is decomposed. Usually a dataset
defines how to process the annotations and a data pipeline defines all the steps to prepare a data dict.
A pipeline consists of a sequence of operations. Each operation takes a dict as input and also output a dict for the next transform.

The operations are categorized into data loading, pre-processing and formatting.

Here is an pipeline example for ResNet-50 training on ImageNet.

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

For each operation, we list the related dict fields that are added/updated/removed.
At the end of the pipeline, we use `Collect` to only retain the necessary items for forward computation.

### Data loading

`LoadImageFromFile`

- add: img, img_shape, ori_shape

By default, `LoadImageFromFile` loads images from disk but it may lead to IO bottleneck for efficient small models.
Various backends are supported by mmcv to accelerate this process. For example, if the training machines have setup
[memcached](https://memcached.org/), we can revise the config as follows.

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

More supported backends can be found in [mmcv.fileio.FileClient](https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py).

### Pre-processing

`Resize`

- add: scale, scale_idx, pad_shape, scale_factor, keep_ratio
- update: img, img_shape

`RandomFlip`

- add: flip, flip_direction
- update: img

`RandomCrop`

- update: img, pad_shape

`Normalize`

- add: img_norm_cfg
- update: img

### Formatting

`ToTensor`

- update: specified by `keys`.

`ImageToTensor`

- update: specified by `keys`.

`Collect`

- remove: all other keys except for those specified by `keys`

For more information about other data transformation classes, please refer to [Data Transformations](../api/transforms.rst)

## Extend and use custom pipelines

1. Write a new pipeline in any file, e.g., `my_pipeline.py`, and place it in
   the folder `mmcls/datasets/pipelines/`. The pipeline class needs to override
   the `__call__` method which takes a dict as input and returns a dict.

   ```python
   from mmcls.datasets import PIPELINES

   @PIPELINES.register_module()
   class MyTransform(object):

       def __call__(self, results):
           # apply transforms on results['img']
           return results
   ```

2. Import the new class in `mmcls/datasets/pipelines/__init__.py`.

   ```python
   ...
   from .my_pipeline import MyTransform

   __all__ = [
       ..., 'MyTransform'
   ]
   ```

3. Use it in config files.

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

## Pipeline visualization

After designing data pipelines, you can use the [visualization tools](../tools/visualization.md) to view the performance.
