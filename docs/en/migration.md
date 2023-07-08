# Migration

We introduce some modifications in MMPretrain 1.x, and some of them are BC-breacking. To migrate your projects from **MMClassification 0.x** or **MMSelfSup 0.x** smoothly, please read this tutorial.

- [Migration](#migration)
  - [New dependencies](#new-dependencies)
- [General change of config](#general-change-of-config)
  - [Schedule settings](#schedule-settings)
  - [Runtime settings](#runtime-settings)
  - [Other changes](#other-changes)
- [Migration from MMClassification 0.x](#migration-from-mmclassification-0x)
  - [Config files](#config-files)
    - [Model settings](#model-settings)
    - [Data settings](#data-settings)
  - [Packages](#packages)
    - [`mmpretrain.apis`](#mmpretrainapis)
    - [`mmpretrain.core`](#mmpretraincore)
    - [`mmpretrain.datasets`](#mmpretraindatasets)
    - [`mmpretrain.models`](#mmpretrainmodels)
    - [`mmpretrain.utils`](#mmpretrainutils)
- [Migration from MMSelfSup 0.x](#migration-from-mmselfsup-0x)
  - [Config](#config)
    - [Dataset settings](#dataset-settings)
    - [Model settings](#model-settings-1)
  - [Package](#package)

## New dependencies

```{warning}
MMPretrain 1.x has new package dependencies, and a new environment should be created for MMPretrain 1.x even if you already have a well-rounded MMClassification 0.x or MMSelfSup 0.x environment. Please refer to the [installation tutorial](./get_started.md) for the required package installation or install the packages manually.
```

1. [MMEngine](https://github.com/open-mmlab/mmengine): MMEngine is the core the OpenMMLab 2.0 architecture,
   and we have split many compentents unrelated to computer vision from MMCV to MMEngine.
2. [MMCV](https://github.com/open-mmlab/mmcv): The computer vision package of OpenMMLab. This is not a new
   dependency, but it should be upgraded to version `2.0.0rc1` or above.
3. [rich](https://github.com/Textualize/rich): A terminal formatting package, and we use it to enhance some
   outputs in the terminal.
4. [einops](https://github.com/arogozhnikov/einops): Operators for Einstein notations.

# General change of config

In this section, we introduce the general difference between old version(**MMClassification 0.x** or **MMSelfSup 0.x**) and **MMPretrain 1.x**.

## Schedule settings

| MMCls or MMSelfSup 0.x | MMPretrain 1.x  | Remark                                                                                                                          |
| ---------------------- | --------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| optimizer_config       | /               | It has been **removed**.                                                                                                        |
| /                      | optim_wrapper   | The `optim_wrapper` provides a common interface for updating parameters.                                                        |
| lr_config              | param_scheduler | The `param_scheduler` is a list to set learning rate or other parameters, which is more flexible.                               |
| runner                 | train_cfg       | The loop setting (`EpochBasedTrainLoop`, `IterBasedTrainLoop`) in `train_cfg` controls the work flow of the algorithm training. |

Changes in **`optimizer`** and **`optimizer_config`**:

- Now we use `optim_wrapper` field to specify all configurations related to optimization process. The
  `optimizer` has become a subfield of `optim_wrapper`.
- The `paramwise_cfg` field is also a subfield of `optim_wrapper`, instead of `optimizer`.
- The `optimizer_config` field has been removed, and all configurations has been moved to `optim_wrapper`.
- The `grad_clip` field has been renamed to `clip_grad`.

<table class="docutils">
<tr>
<td>Original</td>
<td>

```python
optimizer = dict(
    type='AdamW',
    lr=0.0015,
    weight_decay=0.3,
    paramwise_cfg = dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
    ))

optimizer_config = dict(grad_clip=dict(max_norm=1.0))
```

</td>
<tr>
<td>New</td>
<td>

```python
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.0015, weight_decay=0.3),
    paramwise_cfg = dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
    ),
    clip_grad=dict(max_norm=1.0),
)
```

</td>
</tr>
</table>

Changes in **`lr_config`**:

- The `lr_config` field has been removed and replaced by the new `param_scheduler`.
- The `warmup` related arguments have also been removed since we use a combination of schedulers to implement this
  functionality.

The new scheduler combination mechanism is highly flexible and enables the design of various learning rate/momentum curves.
For more details, see the {external+mmengine:doc}`parameter schedulers tutorial <tutorials/param_scheduler>`.

<table class="docutils">
<tr>
<td>Original</td>
<td>

```python
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_iters=5,
    warmup_ratio=0.01,
    warmup_by_epoch=True)
```

</td>
<tr>
<td>New</td>
<td>

```python
param_scheduler = [
    # warmup
    dict(
        type='LinearLR',
        start_factor=0.01,
        by_epoch=True,
        end=5,
        # Update the learning rate after every iters.
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(type='CosineAnnealingLR', by_epoch=True, begin=5),
]
```

</td>
</tr>
</table>

Changes in **`runner`**:

Most of the configurations that were originally in the `runner` field have been moved to `train_cfg`, `val_cfg`, and `test_cfg`.
These fields are used to configure the loop for training, validation, and testing.

<table class="docutils">
<tr>
<td>Original</td>
<td>

```python
runner = dict(type='EpochBasedRunner', max_epochs=100)
```

</td>
<tr>
<td>New</td>
<td>

```python
# The `val_interval` is the original `evaluation.interval`.
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
val_cfg = dict()   # Use the default validation loop.
test_cfg = dict()  # Use the default test loop.
```

</td>
</tr>
</table>

In OpenMMLab 2.0, we introduced `Loop` to control the behaviors in training, validation and testing. As a result, the functionalities of `Runner` have also been changed.
More details can be found in the {external+mmengine:doc}`MMEngine tutorials <design/runner>`.

## Runtime settings

Changes in **`checkpoint_config`** and **`log_config`**:

The `checkpoint_config` has been moved to `default_hooks.checkpoint`, and `log_config` has been moved to
`default_hooks.logger`. Additionally, many hook settings that were previously included in the script code have
been moved to the `default_hooks` field in the runtime configuration.

```python
default_hooks = dict(
    # record the time of every iterations.
    timer=dict(type='IterTimerHook'),

    # print log every 100 iterations.
    logger=dict(type='LoggerHook', interval=100),

    # enable the parameter scheduler.
    param_scheduler=dict(type='ParamSchedulerHook'),

    # save checkpoint per epoch, and automatically save the best checkpoint.
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='auto'),

    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type='DistSamplerSeedHook'),

    # validation results visualization, set True to enable it.
    visualization=dict(type='VisualizationHook', enable=False),
)
```

In OpenMMLab 2.0, we have split the original logger into logger and visualizer. The logger is used to record
information, while the visualizer is used to display the logger in different backends such as terminal,
TensorBoard, and Wandb.

<table class="docutils">
<tr>
<td>Original</td>
<td>

```python
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])
```

</td>
<tr>
<td>New</td>
<td>

```python
default_hooks = dict(
    ...
    logger=dict(type='LoggerHook', interval=100),
)

visualizer = dict(
    type='UniversalVisualizer',
    vis_backends=[dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')],
)
```

</td>
</tr>
</table>

Changes in **`load_from`** and **`resume_from`**:

- The `resume_from` is removed. And we use `resume` and `load_from` to replace it.
  - If `resume=True` and `load_from` is not None, resume training from the checkpoint in `load_from`.
  - If `resume=True` and `load_from` is None, try to resume from the latest checkpoint in the work directory.
  - If `resume=False` and `load_from` is not None, only load the checkpoint, not resume training.
  - If `resume=False` and `load_from` is None, do not load nor resume.

the `resume_from` field has been removed, and we use `resume` and `load_from` instead.

- If `resume=True` and `load_from` is not None, training is resumed from the checkpoint in `load_from`.
- If `resume=True` and `load_from` is None, the latest checkpoint in the work directory is used for resuming.
- If `resume=False` and `load_from` is not None, only the checkpoint is loaded without resuming training.
- If `resume=False` and `load_from` is None, neither checkpoint is loaded nor is training resumed.

Changes in **`dist_params`**: The `dist_params` field has become a subfield of `env_cfg` now.
Additionally, some new configurations have been added to `env_cfg`.

```python
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,

    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),

    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)
```

Changes in **`workflow`**: `workflow` related functionalities are removed.

New field **`visualizer`**: The visualizer is a new design in OpenMMLab 2.0 architecture. The runner uses an
instance of the visualizer to handle result and log visualization, as well as to save to different backends.
For more information, please refer to the {external+mmengine:doc}`MMEngine tutorial <advanced_tutorials/visualization>`.

```python
visualizer = dict(
    type='UniversalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        # Uncomment the below line to save the log and visualization results to TensorBoard.
        # dict(type='TensorboardVisBackend')
    ]
)
```

New field **`default_scope`**: The start point to search module for all registries. The `default_scope` in MMPretrain is `mmpretrain`. See {external+mmengine:doc}`the registry tutorial <advanced_tutorials/registry>` for more details.

## Other changes

We moved the definition of all registries in different packages to the `mmpretrain.registry` package.

# Migration from MMClassification 0.x

## Config files

In MMPretrain 1.x, we refactored the structure of configuration files, and the original files are not usable.

In this section, we will introduce all changes of the configuration files. And we assume you already have
ideas of the [config files](./user_guides/config.md).

### Model settings

No changes in `model.backbone`, `model.neck` and `model.head` fields.

Changes in **`model.train_cfg`**:

- `BatchMixup` is renamed to [`Mixup`](mmpretrain.models.utils.batch_augments.Mixup).
- `BatchCutMix` is renamed to [`CutMix`](mmpretrain.models.utils.batch_augments.CutMix).
- `BatchResizeMix` is renamed to [`ResizeMix`](mmpretrain.models.utils.batch_augments.ResizeMix).
- The `prob` argument is removed from all augments settings, and you can use the `probs` field in `train_cfg` to
  specify probabilities of every augemnts. If no `probs` field, randomly choose one by the same probability.

<table class="docutils">
<tr>
<td>Original</td>
<td>

```python
model = dict(
    ...
    train_cfg=dict(augments=[
        dict(type='BatchMixup', alpha=0.8, num_classes=1000, prob=0.5),
        dict(type='BatchCutMix', alpha=1.0, num_classes=1000, prob=0.5)
    ]
)
```

</td>
<tr>
<td>New</td>
<td>

```python
model = dict(
    ...
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8), dict(type='CutMix', alpha=1.0)]
)
```

</td>
</tr>
</table>

### Data settings

Changes in **`data`**:

- The original `data` field is splited to `train_dataloader`, `val_dataloader` and
  `test_dataloader`. This allows us to configure them in fine-grained. For example,
  you can specify different sampler and batch size during training and test.
- The `samples_per_gpu` is renamed to `batch_size`.
- The `workers_per_gpu` is renamed to `num_workers`.

<table class="docutils">
<tr>
<td>Original</td>
<td>

```python
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    train=dict(...),
    val=dict(...),
    test=dict(...),
)
```

</td>
<tr>
<td>New</td>
<td>

```python
train_dataloader = dict(
    batch_size=32,
    num_workers=2,
    dataset=dict(...),
    sampler=dict(type='DefaultSampler', shuffle=True)  # necessary
)

val_dataloader = dict(
    batch_size=32,
    num_workers=2,
    dataset=dict(...),
    sampler=dict(type='DefaultSampler', shuffle=False)  # necessary
)

test_dataloader = val_dataloader
```

</td>
</tr>
</table>

Changes in **`pipeline`**:

- The original formatting transforms **`ToTensor`**, **`ImageToTensor`** and **`Collect`** are combined as [`PackInputs`](mmpretrain.datasets.transforms.PackInputs).
- We don't recommend to do **`Normalize`** in the dataset pipeline. Please remove it from pipelines and set it in the `data_preprocessor` field.
- The argument `flip_prob` in [**`RandomFlip`**](mmcv.transforms.RandomFlip) is renamed to `prob`.
- The argument `size` in [**`RandomCrop`**](mmpretrain.datasets.transforms.RandomCrop) is renamed to `crop_size`.
- The argument `size` in [**`RandomResizedCrop`**](mmpretrain.datasets.transforms.RandomResizedCrop) is renamed to `scale`.
- The argument `size` in [**`Resize`**](mmcv.transforms.Resize) is renamed to `scale`. And `Resize` won't support size like `(256, -1)`, please use [`ResizeEdge`](mmpretrain.datasets.transforms.ResizeEdge) to replace it.
- The argument `policies` in [**`AutoAugment`**](mmpretrain.datasets.transforms.AutoAugment) and [**`RandAugment`**](mmpretrain.datasets.transforms.RandAugment) supports using string to specify preset policies. `AutoAugment` supports "imagenet" and `RandAugment` supports "timm_increasing".
- **`RandomResizedCrop`** and **`CenterCrop`** won't supports `efficientnet_style`, and please use [`EfficientNetRandomCrop`](mmpretrain.datasets.transforms.EfficientNetRandomCrop) and [`EfficientNetCenterCrop`](mmpretrain.datasets.transforms.EfficientNetCenterCrop) to replace them.

```{note}
We move some work of data transforms to the data preprocessor, like normalization, see [the documentation](mmpretrain.models.utils.data_preprocessor) for
more details.
```

<table class="docutils">
<tr>
<td>Original</td>
<td>

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
```

</td>
<tr>
<td>New</td>
<td>

```python
data_preprocessor = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=224),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]
```

</td>
</tr>
</table>

Changes in **`evaluation`**:

- The **`evaluation`** field is splited to `val_evaluator` and `test_evaluator`. And it won't supports `interval` and `save_best` arguments.
  The `interval` is moved to `train_cfg.val_interval`, see [the schedule settings](./user_guides/config.md#schedule-settings) and the `save_best`
  is moved to `default_hooks.checkpoint.save_best`, see [the runtime settings](./user_guides/config.md#runtime-settings).
- The 'accuracy' metric is renamed to [`Accuracy`](mmpretrain.evaluation.Accuracy).
- The 'precision', 'recall', 'f1-score' and 'support' are combined as [`SingleLabelMetric`](mmpretrain.evaluation.SingleLabelMetric), and use `items` argument to specify to calculate which metric.
- The 'mAP' is renamed to [`AveragePrecision`](mmpretrain.evaluation.AveragePrecision).
- The 'CP', 'CR', 'CF1', 'OP', 'OR', 'OF1' are combined as [`MultiLabelMetric`](mmpretrain.evaluation.MultiLabelMetric), and use `items` and `average` arguments to specify to calculate which metric.

<table class="docutils">
<tr>
<td>Original</td>
<td>

```python
evaluation = dict(
    interval=1,
    metric='accuracy',
    metric_options=dict(topk=(1, 5))
)
```

</td>
<tr>
<td>New</td>
<td>

```python
val_evaluator = dict(type='Accuracy', topk=(1, 5))
test_evaluator = val_evaluator
```

</td>
</tr>
<tr>
<td>Original</td>
<td>

```python
evaluation = dict(
    interval=1,
    metric=['mAP', 'CP', 'OP', 'CR', 'OR', 'CF1', 'OF1'],
    metric_options=dict(thr=0.5),
)
```

</td>
<tr>
<td>New</td>
<td>

```python
val_evaluator = [
    dict(type='AveragePrecision'),
    dict(type='MultiLabelMetric',
        items=['precision', 'recall', 'f1-score'],
        average='both',
        thr=0.5),
]
test_evaluator = val_evaluator
```

</td>
</tr>
</table>

## Packages

### `mmpretrain.apis`

The documentation can be found [here](mmpretrain.apis).

|       Function       | Changes                                                                                                                                          |
| :------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------- |
|     `init_model`     | No changes                                                                                                                                       |
|  `inference_model`   | No changes. But we recommend to use [`mmpretrain.ImageClassificationInferencer`](mmpretrain.apis.ImageClassificationInferencer) instead.         |
|    `train_model`     | Removed, use `runner.train` to train.                                                                                                            |
|   `multi_gpu_test`   | Removed, use `runner.test` to test.                                                                                                              |
|  `single_gpu_test`   | Removed, use `runner.test` to test.                                                                                                              |
| `show_result_pyplot` | Removed, use [`mmpretrain.ImageClassificationInferencer`](mmpretrain.apis.ImageClassificationInferencer) to inference model and show the result. |
|  `set_random_seed`   | Removed, use `mmengine.runner.set_random_seed`.                                                                                                  |
|  `init_random_seed`  | Removed, use `mmengine.dist.sync_random_seed`.                                                                                                   |

### `mmpretrain.core`

The `mmpretrain.core` package is renamed to [`mmpretrain.engine`](mmpretrain.engine).

|   Sub package   | Changes                                                                                                                           |
| :-------------: | :-------------------------------------------------------------------------------------------------------------------------------- |
|  `evaluation`   | Removed, use the metrics in [`mmpretrain.evaluation`](mmpretrain.evaluation).                                                     |
|     `hook`      | Moved to [`mmpretrain.engine.hooks`](mmpretrain.engine.hooks)                                                                     |
|  `optimizers`   | Moved to [`mmpretrain.engine.optimizers`](mmpretrain.engine.optimizers)                                                           |
|     `utils`     | Removed, the distributed environment related functions can be found in the [`mmengine.dist`](api/dist) package.                   |
| `visualization` | Removed, the related functionalities are implemented in [`mmengine.visualization.Visualizer`](mmengine.visualization.Visualizer). |

The `MMClsWandbHook` in `hooks` package is waiting for implementation.

The `CosineAnnealingCooldownLrUpdaterHook` in `hooks` package is removed, and we support this functionality by
the combination of parameter schedulers, see [the tutorial](./advanced_guides/schedule.md).

### `mmpretrain.datasets`

The documentation can be found [here](mmpretrain.datasets).

|                                       Dataset class                                       | Changes                                                                                                         |
| :---------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------- |
|                   [`CustomDataset`](mmpretrain.datasets.CustomDataset)                    | Add `data_root` argument as the common prefix of `data_prefix` and `ann_file` and support to load unlabeled data. |
|                        [`ImageNet`](mmpretrain.datasets.ImageNet)                         | Same as `CustomDataset`.                                                                                        |
|                     [`ImageNet21k`](mmpretrain.datasets.ImageNet21k)                      | Same as `CustomDataset`.                                                                                        |
|   [`CIFAR10`](mmpretrain.datasets.CIFAR10) & [`CIFAR100`](mmpretrain.datasets.CIFAR100)   | The `test_mode` argument is a required argument now.                                                            |
| [`MNIST`](mmpretrain.datasets.MNIST) & [`FashionMNIST`](mmpretrain.datasets.FashionMNIST) | The `test_mode` argument is a required argument now.                                                            |
|                             [`VOC`](mmpretrain.datasets.VOC)                              | Requires `data_root`, `image_set_path` and `test_mode` now.                                                     |
|                             [`CUB`](mmpretrain.datasets.CUB)                              | Requires `data_root` and `test_mode` now.                                                                       |

The `mmpretrain.datasets.pipelines` is renamed to `mmpretrain.datasets.transforms`.

|         Transform class         | Changes                                                                                                                                                                   |
| :-----------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
|       `LoadImageFromFile`       | Removed, use [`mmcv.transforms.LoadImageFromFile`](mmcv.transforms.LoadImageFromFile).                                                                                    |
|          `RandomFlip`           | Removed, use [`mmcv.transforms.RandomFlip`](mmcv.transforms.RandomFlip). The argument `flip_prob` is renamed to `prob`.                                                   |
|          `RandomCrop`           | The argument `size` is renamed to `crop_size`.                                                                                                                            |
|       `RandomResizedCrop`       | The argument `size` is renamed to `scale`. The argument `scale` is renamed to `crop_ratio_range`. Won't support `efficientnet_style`, use [`EfficientNetRandomCrop`](mmpretrain.datasets.transforms.EfficientNetRandomCrop). |
|          `CenterCrop`           | Removed, use [`mmcv.transforms.CenterCrop`](mmcv.transforms.CenterCrop). Won't support `efficientnet_style`, use [`EfficientNetCenterCrop`](mmpretrain.datasets.transforms.EfficientNetCenterCrop). |
|            `Resize`             | Removed, use [`mmcv.transforms.Resize`](mmcv.transforms.Resize). The argument `size` is renamed to `scale`. Won't support size like `(256, -1)`, use [`ResizeEdge`](mmpretrain.datasets.transforms.ResizeEdge). |
| `AutoAugment` & `RandomAugment` | The argument `policies` supports using string to specify preset policies.                                                                                                 |
|            `Compose`            | Removed, use [`mmcv.transforms.Compose`](mmcv.transforms.Compose).                                                                                                        |

### `mmpretrain.models`

The documentation can be found [here](mmpretrain.models). The interface of all **backbones**, **necks** and **losses** didn't change.

Changes in [`ImageClassifier`](mmpretrain.models.classifiers.ImageClassifier):

| Method of classifiers | Changes                                                                                                                                                                 |
| :-------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|    `extract_feat`     | No changes                                                                                                                                                              |
|       `forward`       | Now only accepts three arguments: `inputs`, `data_samples` and `mode`. See [the documentation](mmpretrain.models.classifiers.ImageClassifier.forward) for more details. |
|    `forward_train`    | Replaced by `loss`.                                                                                                                                                     |
|     `simple_test`     | Replaced by `predict`.                                                                                                                                                  |
|     `train_step`      | The `optimizer` argument is replaced by `optim_wrapper` and it accepts [`OptimWrapper`](mmengine.optim.OptimWrapper).                                                   |
|      `val_step`       | The original `val_step` is the same as `train_step`, now it calls `predict`.                                                                                            |
|      `test_step`      | New method, and it's the same as `val_step`.                                                                                                                            |

Changes in [heads](mmpretrain.models.heads):

| Method of heads | Changes                                                                                                                                                |
| :-------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------- |
|  `pre_logits`   | No changes                                                                                                                                             |
| `forward_train` | Replaced by `loss`.                                                                                                                                    |
|  `simple_test`  | Replaced by `predict`.                                                                                                                                 |
|     `loss`      | It accepts `data_samples` instead of `gt_labels` to calculate loss. The `data_samples` is a list of [ClsDataSample](mmpretrain.structures.DataSample). |
|    `forward`    | New method, and it returns the output of the classification head without any post-processs like softmax or sigmoid.                                    |

### `mmpretrain.utils`

|           Function           | Changes                                                                                                         |
| :--------------------------: | :-------------------------------------------------------------------------------------------------------------- |
|        `collect_env`         | No changes                                                                                                      |
|      `get_root_logger`       | Removed, use [`mmengine.logging.MMLogger.get_current_instance`](mmengine.logging.MMLogger.get_current_instance) |
|       `load_json_log`        | The output format changed.                                                                                      |
|   `setup_multi_processes`    | Removed, use [`mmengine.utils.dl_utils.set_multi_processing`](mmengine.utils.dl_utils.set_multi_processing).    |
| `wrap_non_distributed_model` | Removed, we auto wrap the model in the runner.                                                                  |
|   `wrap_distributed_model`   | Removed, we auto wrap the model in the runner.                                                                  |
|     `auto_select_device`     | Removed, we auto select the device in the runner.                                                               |

# Migration from MMSelfSup 0.x

## Config

This section illustrates the changes of our config files in the `_base_` folder, which includes three parts

- Datasets: `configs/_base_/datasets`
- Models: `configs/_base_/models`
- Schedules: `configs/_base_/schedules`

### Dataset settings

In **MMSelfSup 0.x**, we use key `data` to summarize all information, such as `samples_per_gpu`, `train`, `val`, etc.

In **MMPretrain 1.x**, we separate `train_dataloader`, `val_dataloader` to summarize information correspodingly and the key `data` has been **removed**.

<table class="docutils">
<tr>
<td>Original</td>
<td>

```python
data = dict(
    samples_per_gpu=32,  # total 32*8(gpu)=256
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_source=dict(
            type=data_source,
            data_prefix='data/imagenet/train',
            ann_file='data/imagenet/meta/train.txt',
        ),
        num_views=[1, 1],
        pipelines=[train_pipeline1, train_pipeline2],
        prefetch=prefetch,
    ),
    val=...)
```

</td>

<tr>
<td>New</td>
<td>

```python
train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='meta/train.txt',
        data_prefix=dict(img_path='train/'),
        pipeline=train_pipeline))
val_dataloader = ...
```

</td>
</tr>
</table>

Besides, we **remove** the key of `data_source` to keep the pipeline format consistent with that in other OpenMMLab projects. Please refer to [Config](user_guides/config.md) for more details.

Changes in **`pipeline`**:

Take MAE as an example of `pipeline`:

```python
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        scale=224,
        crop_ratio_range=(0.2, 1.0),
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackInputs')
]
```

### Model settings

In the config of models, there are two main different parts from MMSeflSup 0.x.

1. There is a new key called `data_preprocessor`, which is responsible for preprocessing the data, like normalization, channel conversion, etc. For example:

```python
data_preprocessor=dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True)
model = dict(
    type='MAE',
    data_preprocessor=dict(
        mean=[127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5],
        bgr_to_rgb=True),
    backbone=...,
    neck=...,
    head=...,
    init_cfg=...)
```

2. There is a new key `loss` in `head` in MMPretrain 1.x, to determine the loss function of the algorithm. For example:

```python
model = dict(
    type='MAE',
    backbone=...,
    neck=...,
    head=dict(
        type='MAEPretrainHead',
        norm_pix=True,
        patch_size=16,
        loss=dict(type='MAEReconstructionLoss')),
    init_cfg=...)
```

## Package

The table below records the general modification of the folders and files.

| MMSelfSup 0.x            | MMPretrain 1.x      | Remark                                                                                                                                                        |
| ------------------------ | ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| apis                     | apis                | The high level APIs are updated.                                                                                                                              |
| core                     | engine              | The `core` folder has been renamed to `engine`, which includes `hooks`, `opimizers`. ([API link](mmpretrain.engine))                                          |
| datasets                 | datasets            | The datasets is implemented according to different datasets, such as ImageNet, Places205. ([API link](mmpretrain.datasets))                                   |
| datasets/data_sources    | /                   | The `data_sources` has been **removed** and the directory of `datasets` now is consistent with other OpenMMLab projects.                                      |
| datasets/pipelines       | datasets/transforms | The `pipelines` folder has been renamed to `transforms`. ([API link](mmpretrain.datasets.transforms))                                                         |
| /                        | evaluation          | The `evaluation` is created for some evaluation functions or classes. ([API link](mmpretrain.evaluation))                                                     |
| models/algorithms        | selfsup             | The algorithms are moved to `selfsup` folder. ([API link](mmpretrain.models.selfsup))                                                                         |
| models/backbones         | selfsup             | The re-implemented backbones are moved to corresponding self-supervised learning algorithm `.py` files. ([API link](mmpretrain.models.selfsup))               |
| models/target_generators | selfsup             | The target generators are moved to corresponding self-supervised learning algorithm `.py` files. ([API link](mmpretrain.models.selfsup))                      |
| /                        | models/losses       | The `losses` folder is created to provide different loss implementations, which is from `heads`. ([API link](mmpretrain.models.losses))                       |
| /                        | structures          | The `structures` folder is for the implementation of data structures. In MMPretrain, we implement a new data structure, `DataSample`,  to pass and receive data throughout the training/val process. ([API link](mmpretrain.structures)) |
| /                        | visualization       | The `visualization` folder contains the visualizer, which is responsible for some visualization tasks like visualizing data augmentation. ([API link](mmpretrain.visualization)) |
