# Migration from MMClassification 0.x

We introduce some modifications in MMClassification 1.x, and some of them are BC-breading. To migrate your projects from MMClassification 0.x smoothly, please read this tutorial.

## New dependencies

MMClassification 1.x depends on some new packages, you can prepare a new clean environment and install again
according to the [install tutorial](./get_started.md). Or install the below packages manually.

1. [MMEngine](https://github.com/open-mmlab/mmengine): MMEngine is the core the OpenMMLab 2.0 architecture,
   and we splited many compentents unrelated to computer vision from MMCV to MMEngine.
2. [MMCV](https://github.com/open-mmlab/mmcv): The computer vision package of OpenMMLab. This is not a new
   dependency, but you need to upgrade it to above `2.0.0rc1` version.
3. [rich](https://github.com/Textualize/rich): A terminal formatting package, and we use it to beautify some
   outputs in the terminal.

## Configuration files

In MMClassification 1.x, we refactored the structure of configuration files, and the original files are not usable.

<!-- TODO: migration tool -->

In this section, we will introduce all changes of the configuration files. And we assume you already have
ideas of the [config files](./user_guides/config.md).

### Model settings

No changes in `model.backbone`, `model.neck` and `model.head` fields.

Changes in **`model.train_cfg`**:

- `BatchMixup` is renamed to [`Mixup`](mmcls.models.utils.batch_augments.Mixup).
- `BatchCutMix` is renamed to [`CutMix`](mmcls.models.utils.batch_augments.CutMix).
- `BatchResizeMix` is renamed to [`ResizeMix`](mmcls.models.utils.batch_augments.ResizeMix).
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

- The original formatting transforms **`ToTensor`**、**`ImageToTensor`**、**`Collect`** are combined as [`PackClsInputs`](mmcls.datasets.transforms.PackClsInputs).
- We don't recommend to do **`Normalize`** in the dataset pipeline. Please remove it from pipelines and set it in the `data_preprocessor` field.
- The argument `flip_prob` in [**`RandomFlip`**](mmcv.transforms.RandomFlip) is renamed to `flip`.
- The argument `size` in [**`RandomCrop`**](mmcls.datasets.transforms.RandomCrop) is renamed to `crop_size`.
- The argument `size` in [**`RandomResizedCrop`**](mmcls.datasets.transforms.RandomResizedCrop) is renamed to `scale`.
- The argument `size` in [**`Resize`**](mmcv.transforms.Resize) is renamed to `scale`. And `Resize` won't support size like `(256, -1)`, please use [`ResizeEdge`](mmcls.datasets.transforms.ResizeEdge) to replace it.
- The argument `policies` in [**`AutoAugment`**](mmcls.datasets.transforms.AutoAugment) and [**`RandAugment`**](mmcls.datasets.transforms.RandAugment) supports using string to specify preset policies. `AutoAugment` supports "imagenet" and `RandAugment` supports "timm_increasing".
- **`RandomResizedCrop`** and **`CenterCrop`** won't supports `efficientnet_style`, and please use [`EfficientNetRandomCrop`](mmcls.datasets.transforms.EfficientNetRandomCrop) and [`EfficientNetCenterCrop`](mmcls.datasets.transforms.EfficientNetCenterCrop) to replace them.

```{note}
We move some work of data transforms to the data preprocessor, like normalization, see [the documentation](mmcls.models.utils.data_preprocessor) for
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
    dict(type='PackClsInputs'),
]
```

</td>
</tr>
</table>

Changes in **`evaluation`**:

- The **`evaluation`** field is splited to `val_evaluator` and `test_evaluator`. And it won't supports `interval` and `save_best` arguments.
  The `interval` is moved to `train_cfg.val_interval`, see [the schedule settings](./user_guides/config.md#schedule-settings) and the `save_best`
  is moved to `default_hooks.checkpoint.save_best`, see [the runtime settings](./user_guides/config.md#runtime-settings).
- The 'accuracy' metric is renamed to [`Accuracy`](mmcls.evaluation.Accuracy).
- The 'precision'，'recall'，'f1-score' and 'support' are combined as [`SingleLabelMetric`](mmcls.evaluation.SingleLabelMetric), and use `items` argument to specify to calculate which metric.
- The 'mAP' is renamed to [`AveragePrecision`](mmcls.evaluation.AveragePrecision).
- The 'CP', 'CR', 'CF1', 'OP', 'OR', 'OF1' are combined as [`MultiLabelMetric`](mmcls.evaluation.MultiLabelMetric), and use `items` and `average` arguments to specify to calculate which metric.

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

### Schedule settings

Changes in **`optimizer`** and **`optimizer_config`**:

- Now we use `optim_wrapper` field to specify all configuration about the optimization process. And the
  `optimizer` is a sub field of `optim_wrapper` now.
- `paramwise_cfg` is also a sub field of `optim_wrapper`, instead of `optimizer`.
- `optimizer_config` is removed now, and all configurations of it are moved to `optim_wrapper`.
- `grad_clip` is renamed to `clip_grad`.

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

- The `lr_config` field is removed and we use new `param_scheduler` to replace it.
- The `warmup` related arguments are removed, since we use schedulers combination to implement this
  functionality.

The new schedulers combination mechanism is very flexible, and you can use it to design many kinds of learning
rate / momentum curves. See {external+mmengine:doc}`the tutorial <tutorials/param_scheduler>` for more details.

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

Most configuration in the original `runner` field is moved to `train_cfg`, `val_cfg` and `test_cfg`, which
configure the loop in training, validation and test.

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

In fact, in OpenMMLab 2.0, we introduced `Loop` to control the behaviors in training, validation and test. And
the functionalities of `Runner` are also changed. You can find more details in {external+mmengine:doc}`the MMEngine tutorials <tutorials/runner>`.

### Runtime settings

Changes in **`checkpoint_config`** and **`log_config`**:

The `checkpoint_config` are moved to `default_hooks.checkpoint` and the `log_config` are moved to `default_hooks.logger`.
And we move many hooks settings from the script code to the `default_hooks` field in the runtime configuration.

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

In addition, we splited the original logger to logger and visualizer. The logger is used to record
information and the visualizer is used to show the logger in different backends, like terminal, TensorBoard
and Wandb.

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
    type='ClsVisualizer',
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

Changes in **`dist_params`**: The `dist_params` field is a sub field of `env_cfg` now. And there are some new
configurations in the `env_cfg`.

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

New field **`visualizer`**: The visualizer is a new design in OpenMMLab 2.0 architecture. We use a
visualizer instance in the runner to handle results & log visualization and save to different backends.
See the {external+mmengine:doc}`MMEngine tutorial <advanced_tutorials/visualization>` for more details.

```python
visualizer = dict(
    type='ClsVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        # Uncomment the below line to save the log and visualization results to TensorBoard.
        # dict(type='TensorboardVisBackend')
    ]
)
```

New field **`default_scope`**: The start point to search module for all registries. The `default_scope` in MMClassification is `mmcls`. See {external+mmengine:doc}`the registry tutorial <advanced_tutorials/registry>` for more details.

## Packages

### `mmcls.apis`

The documentation can be found [here](mmcls.apis).

|       Function       | Changes                                         |
| :------------------: | :---------------------------------------------- |
|     `init_model`     | No changes                                      |
|  `inference_model`   | No changes                                      |
|    `train_model`     | Removed, use `runner.train` to train.           |
|   `multi_gpu_test`   | Removed, use `runner.test` to test.             |
|  `single_gpu_test`   | Removed, use `runner.test` to test.             |
| `show_result_pyplot` | Waiting for support.                            |
|  `set_random_seed`   | Removed, use `mmengine.runner.set_random_seed`. |
|  `init_random_seed`  | Removed, use `mmengine.dist.sync_random_seed`.  |

### `mmcls.core`

The `mmcls.core` package is renamed to [`mmcls.engine`](mmcls.engine).

|   Sub package   | Changes                                                                                                                           |
| :-------------: | :-------------------------------------------------------------------------------------------------------------------------------- |
|  `evaluation`   | Removed, use the metrics in [`mmcls.evaluation`](mmcls.evaluation).                                                               |
|     `hook`      | Moved to [`mmcls.engine.hooks`](mmcls.engine.hooks)                                                                               |
|  `optimizers`   | Moved to [`mmcls.engine.optimizers`](mmcls.engine.optimizers)                                                                     |
|     `utils`     | Removed, the distributed environment related functions can be found in the [`mmengine.dist`](mmengine.dist) package.              |
| `visualization` | Removed, the related functionalities are implemented in [`mmengine.visualization.Visualizer`](mmengine.visualization.Visualizer). |

The `MMClsWandbHook` in `hooks` package is waiting for implementation.

The `CosineAnnealingCooldownLrUpdaterHook` in `hooks` package is removed, and we support this functionality by
the combination of parameter schedulers, see [the tutorial](./advanced_guides/schedule.md).

### `mmcls.datasets`

The documentation can be found [here](mmcls.datasets).

|                                  Dataset class                                  | Changes                                                                        |
| :-----------------------------------------------------------------------------: | :----------------------------------------------------------------------------- |
|                 [`CustomDataset`](mmcls.datasets.CustomDataset)                 | Add `data_root` argument as the common prefix of `data_prefix` and `ann_file`. |
|                      [`ImageNet`](mmcls.datasets.ImageNet)                      | Same as `CustomDataset`.                                                       |
|                   [`ImageNet21k`](mmcls.datasets.ImageNet21k)                   | Same as `CustomDataset`.                                                       |
|   [`CIFAR10`](mmcls.datasets.CIFAR10) & [`CIFAR100`](mmcls.datasets.CIFAR100)   | The `test_mode` argument is a required argument now.                           |
| [`MNIST`](mmcls.datasets.MNIST) & [`FashionMNIST`](mmcls.datasets.FashionMNIST) | The `test_mode` argument is a required argument now.                           |
|                           [`VOC`](mmcls.datasets.VOC)                           | Requires `data_root`, `image_set_path` and `test_mode` now.                    |
|                           [`CUB`](mmcls.datasets.CUB)                           | Requires `data_root` and `test_mode` now.                                      |

The `mmcls.datasets.pipelines` is renamed to `mmcls.datasets.transforms`.

|         Transform class         | Changes                                                                                                                                                                   |
| :-----------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
|       `LoadImageFromFile`       | Removed, use [`mmcv.transforms.LoadImageFromFile`](mmcv.transforms.LoadImageFromFile).                                                                                    |
|          `RandomFlip`           | Removed, use [`mmcv.transforms.RandomFlip`](mmcv.transforms.RandomFlip). The argument `flip_prob` is renamed to `prob`.                                                   |
|          `RandomCrop`           | The argument `size` is renamed to `crop_size`.                                                                                                                            |
|       `RandomResizedCrop`       | The argument `size` is renamed to `scale`. The argument `scale` is renamed to `crop_ratio_range`. Won't support `efficientnet_style`, use [`EfficientNetRandomCrop`](mmcls.datasets.transforms.EfficientNetRandomCrop). |
|          `CenterCrop`           | Removed, use [`mmcv.transforms.CenterCrop`](mmcv.transforms.CenterCrop). Won't support `efficientnet_style`, use [`EfficientNetCenterCrop`](mmcls.datasets.transforms.EfficientNetCenterCrop). |
|            `Resize`             | Removed, use [`mmcv.transforms.Resize`](mmcv.transforms.Resize). The argument `size` is renamed to `scale`. Won't support size like `(256, -1)`, use [`ResizeEdge`](mmcls.datasets.transforms.ResizeEdge). |
| `AutoAugment` & `RandomAugment` | The argument `policies` supports using string to specify preset policies.                                                                                                 |
|            `Compose`            | Removed, use [`mmcv.transforms.Compose`](mmcv.transforms.Compose).                                                                                                        |

### `mmcls.models`

The documentation can be found [here](mmcls.models). The interface of all **backbones**, **necks** and **losses** didn't change.

Changes in [`ImageClassifier`](mmcls.models.classifiers.ImageClassifier):

| Method of classifiers | Changes                                                                                                                                                            |
| :-------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|    `extract_feat`     | No changes                                                                                                                                                         |
|       `forward`       | Now only accepts three arguments: `inputs`, `data_samples` and `mode`. See [the documentation](mmcls.models.classifiers.ImageClassifier.forward) for more details. |
|    `forward_train`    | Replaced by `loss`.                                                                                                                                                |
|     `simple_test`     | Replaced by `predict`.                                                                                                                                             |
|     `train_step`      | The `optimizer` argument is replaced by `optim_wrapper` and it accepts [`OptimWrapper`](mmengine.optim.OptimWrapper).                                              |
|      `val_step`       | The original `val_step` is the same as `train_step`, now it calls `predict`.                                                                                       |
|      `test_step`      | New method, and it's the same as `val_step`.                                                                                                                       |

Changes in [heads](mmcls.models.heads):

| Method of heads | Changes                                                                                                                                              |
| :-------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------- |
|  `pre_logits`   | No changes                                                                                                                                           |
| `forward_train` | Replaced by `loss`.                                                                                                                                  |
|  `simple_test`  | Replaced by `predict`.                                                                                                                               |
|     `loss`      | It accepts `data_samples` instead of `gt_labels` to calculate loss. The `data_samples` is a list of [ClsDataSample](mmcls.structures.ClsDataSample). |
|    `forward`    | New method, and it returns the output of the classification head without any post-processs like softmax or sigmoid.                                  |

### `mmcls.utils`

|           Function           | Changes                                                       |
| :--------------------------: | :------------------------------------------------------------ |
|        `collect_env`         | No changes                                                    |
|      `get_root_logger`       | Removed, use `mmengine.MMLogger.get_current_instance`         |
|       `load_json_log`        | Waiting for support                                           |
|   `setup_multi_processes`    | Removed, use `mmengine.utils.dl_utils.setup_multi_processes`. |
| `wrap_non_distributed_model` | Removed, we auto wrap the model in the runner.                |
|   `wrap_distributed_model`   | Removed, we auto wrap the model in the runner.                |
|     `auto_select_device`     | Removed, we auto select the device in the runner.             |

### Other changes

- We moved the definition of all registries in different packages to the `mmcls.registry` package.
