# 迁移文档

我们在 MMPretrain 1.x 版本中引入了一些修改，可能会产生兼容性问题。请按照本教程从 MMClassification 0.x 或是 MMSelfSup 0.x 迁移您的项目。

## 新的依赖

```{warning}
MMPretrain 1.x 版本依赖于一些新的代码包，您应该根据 [安装教程](./get_started.md) 来创建新的环境，尽管你可能已经拥有了一个可以正常运行 MMClassification 0.x 或 MMSelfSup 0.x 的环境。请参考[安装文档](./get_started.md) 对依赖库进行对应的安装。
```

1. [MMEngine](https://github.com/open-mmlab/mmengine)：MMEngine 是 OpenMMLab 2.0 架构的核心库，我们将许多与计算机视觉无关的组件从 MMCV 拆分到了 MMEngine。
2. [MMCV](https://github.com/open-mmlab/mmcv)：OpenMMLab 计算机视觉基础库，这不是一个新的依赖，但你需要将其升级到 `2.0.0rc1` 版本以上。
3. [rich](https://github.com/Textualize/rich)：一个命令行美化库，用以在命令行中呈现更美观的输出。

# 配置文件的通用改变

在这个部分，我们将介绍一些旧版本 (**MMClassification 0.x** 或 **MMSelfSup 0.x**) 和 **MMPretrain 1.x** 之间通用的变化规范。

## 训练策略设置

| MMCls or MMSelfSup 0.x | MMPretrain 1.x  | 备注                                                                                                     |
| ---------------------- | --------------- | -------------------------------------------------------------------------------------------------------- |
| optimizer_config       | /               | `optimizer_config` 已经被**移除**。                                                                      |
| /                      | optim_wrapper   | `optim_wrapper` 提供了参数更新的相关字段。                                                               |
| lr_config              | param_scheduler | `param_scheduler` 是一个列表设置学习率或者是其它参数，这将比之前更加灵活。                               |
| runner                 | train_cfg       | `train_cfg` 中的循环设置（如 `EpochBasedTrainLoop`，`IterBasedTrainLoop`）将控制模型训练过程中的工作流。 |

**`optimizer`** 和 **`optimizer_config`** 字段的变化：

- 现在我们使用 `optim_wrapper` 字段指定与优化过程有关的所有配置。而 `optimizer` 字段是 `optim_wrapper` 的一个
  子字段。
- `paramwise_cfg` 字段不再是 `optimizer` 的子字段，而是 `optim_wrapper` 的子字段。
- `optimizer_config` 字段被移除，其配置项被移入 `optim_wrapper` 字段。
- `grad_clip` 被重命名为 `clip_grad`

<table class="docutils">
<tr>
<td>原配置</td>
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
<td>新配置</td>
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

**`lr_config`** 字段的变化：

- `lr_config` 字段被移除，我们使用新的 `param_scheduler` 配置取代。
- `warmup` 相关的字段都被移除，因为学习率预热可以通过多个学习率规划器的组合来实现，因此不再单独实现。

新的优化器参数规划器组合机制非常灵活，你可以使用它来设计多种学习率、动量曲线，详见{external+mmengine:doc}`MMEngine 中的教程 <tutorials/param_scheduler>`。

<table class="docutils">
<tr>
<td>原配置</td>
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
<td>新配置</td>
<td>

```python
param_scheduler = [
    # 学习率预热
    dict(
        type='LinearLR',
        start_factor=0.01,
        by_epoch=True,
        end=5,
        # 每轮迭代都更新学习率，而不是每个 epoch
        convert_to_iter_based=True),
    # 主学习率规划器
    dict(type='CosineAnnealingLR', by_epoch=True, begin=5),
]
```

</td>
</tr>
</table>

**`runner`** 字段的变化：

原 `runner` 字段被拆分为 `train_cfg`，`val_cfg` 和 `test_cfg` 三个字段，分别配置训练、验证和测试循环。

<table class="docutils">
<tr>
<td>原配置</td>
<td>

```python
runner = dict(type='EpochBasedRunner', max_epochs=100)
```

</td>
<tr>
<td>新配置</td>
<td>

```python
# `val_interval` 字段来自原配置中 `evaluation.interval` 字段
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
val_cfg = dict()   # 空字典表示使用默认验证配置
test_cfg = dict()  # 空字典表示使用默认测试配置
```

</td>
</tr>
</table>

在 OpenMMLab 2.0 中，我们引入了“循环控制器”来控制训练、验证和测试行为，而原先 `Runner` 功能也相应地发生了变化。详细介绍参见 MMEngine 中的{external+mmengine:doc}`执行器教程 <design/runner>`。

## 运行设置

**`checkpoint_config`** 和 **`log_config`** 字段的变化：

`checkpoint_config` 被移动至 `default_hooks.checkpoint`，`log_config` 被移动至 `default_hooks.logger`。同时，
我们将很多原先在训练脚本中隐式定义的钩子移动到了 `default_hooks` 字段。

```python
default_hooks = dict(
    # 记录每轮迭代的耗时
    timer=dict(type='IterTimerHook'),

    # 每 100 轮迭代打印一次日志
    logger=dict(type='LoggerHook', interval=100),

    # 启用优化器参数规划器
    param_scheduler=dict(type='ParamSchedulerHook'),

    # 每个 epoch 保存一次模型权重文件，并且自动保存最优权重文件
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='auto'),

    # 在分布式环境中设置采样器种子
    sampler_seed=dict(type='DistSamplerSeedHook'),

    # 可视化验证结果，将 `enable` 设为 True 来启用这一功能。
    visualization=dict(type='VisualizationHook', enable=False),
)
```

此外，我们将原来的日志功能拆分为日志记录和可视化器。日志记录负责按照指定间隔保存日志数据，以及进行数据平滑等处理，可视化器用于在不同的后端记录日志，如终端、TensorBoard 和 WandB。

<table class="docutils">
<tr>
<td>原配置</td>
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
<td>新配置</td>
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

**`load_from`** 和 **`resume_from`** 字段的变动：

- `resume_from` 字段被移除。我们现在使用 `resume` 和 `load_from` 字段实现以下功能：
  - 如 `resume=True` 且 `load_from` 不为 None，从 `load_from` 指定的权重文件恢复训练。
  - 如 `resume=True` 且 `load_from` 为 None，尝试从工作目录中最新的权重文件恢复训练。
  - 如 `resume=False` 且 `load_from` 不为 None，仅加载指定的权重文件，不恢复训练。
  - 如 `resume=False` 且 `load_from` 为 None，不进行任何操作。

**`dist_params`** 字段的变动：`dist_params` 字段被移动为 `env_cfg` 字段的一个子字段。以下为 `env_cfg` 字段的所
有配置项：

```python
env_cfg = dict(
    # 是否启用 cudnn benchmark
    cudnn_benchmark=False,

    # 设置多进程相关参数
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),

    # 设置分布式相关参数
    dist_cfg=dict(backend='nccl'),
)
```

**`workflow`** 字段的变动：`workflow` 相关的功能现已被移除。

新字段 **`visualizer`**：可视化器是 OpenMMLab 2.0 架构中的新设计，我们使用可视化器进行日志、结果的可视化与多后
端的存储。详见 MMEngine 中的{external+mmengine:doc}`可视化教程 <advanced_tutorials/visualization>`。

```python
visualizer = dict(
    type='UniversalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        # 将下行取消注释，即可将日志和可视化结果保存至 TesnorBoard
        # dict(type='TensorboardVisBackend')
    ]
)
```

新字段 **`default_scope`**：指定所有注册器进行模块搜索默认的起点。MMPretrain 中的 `default_scope` 字段为 `mmpretrain`，大部分情况下不需要修改。详见 MMengine 中的{external+mmengine:doc}`注册器教程 <advanced_tutorials/registry>`。

## 其他变动

我们将所有注册器的定义从各个包移动到了 `mmpretrain.registry`。

# 从 MMClassification 0.x 迁移

## 配置文件

在 MMPretrain 1.x 中，我们重构了配置文件的结构，绝大部分原来的配置文件无法直接使用。

在本节中，我们将介绍配置文件的所有变化。我们假设您已经对[配置文件](./user_guides/config.md)有所了解。

### 模型设置

`model.backbone`、`model.neck` 和 `model.head` 字段没有变化。

**`model.train_cfg`** 字段的变化：

- `BatchMixup` 被重命名为 [`Mixup`](mmpretrain.models.utils.batch_augments.Mixup)
- `BatchCutMix` 被重命名为 [`CutMix`](mmpretrain.models.utils.batch_augments.CutMix)
- `BatchResizeMix` 被重命名为 [`ResizeMix`](mmpretrain.models.utils.batch_augments.ResizeMix)
- 以上增强中的 `prob` 参数均被移除，现在在 `train_cfg` 中使用一个统一的 `probs` 字段指定每个增强的概率。如果没
  有指定 `probs` 字段，现在将均匀地随机选择一种增强。

<table class="docutils">
<tr>
<td>原配置</td>
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
<td>新配置</td>
<td>

```python
model = dict(
    ...
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8),
        dict(type='CutMix', alpha=1.0),
    ]
)
```

</td>
</tr>
</table>

### 数据设置

**`data`** 字段的变化：

- 原先的 `data` 字段被拆分为 `train_dataloader`，`val_dataloader` 和 `test_dataloader` 字段。这允许我们进行更
  加细粒度的配置。比如在训练和测试中指定不同的采样器、批次大小等。
- `samples_per_gpu` 字段被重命名为 `batch_size`
- `workers_per_gpu` 字段被重命名为 `num_workers`

<table class="docutils">
<tr>
<td>原配置</td>
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
<td>新配置</td>
<td>

```python
train_dataloader = dict(
    batch_size=32,
    num_workers=2,
    dataset=dict(...),
    sampler=dict(type='DefaultSampler', shuffle=True)  # 必要的
)

val_dataloader = dict(
    batch_size=32,
    num_workers=2,
    dataset=dict(...),
    sampler=dict(type='DefaultSampler', shuffle=False)  # 必要的
)

test_dataloader = val_dataloader
```

</td>
</tr>
</table>

**`pipeline`** 字段的变化：

- 原先的 **`ToTensor`**、**`ImageToTensor`** 和 **`Collect`** 被合并为 [`PackInputs`](mmpretrain.datasets.transforms.PackInputs)
- 我们建议去除数据集流水线中的 **`Normalize`** 变换，转而使用 `data_preprocessor` 字段进行归一化预处理。
- [**`RandomFlip`**](mmcv.transforms.RandomFlip) 中的 `flip_prob` 参数被重命名为 `prob`
- [**`RandomCrop`**](mmpretrain.datasets.transforms.RandomCrop) 中的 `size` 参数被重命名为 `crop_size`
- [**`RandomResizedCrop`**](mmpretrain.datasets.transforms.RandomResizedCrop) 中的 `size` 参数被重命名为 `scale`
- [**`Resize`**](mmcv.transforms.Resize) 中的 `size` 参数被重命名为 `scale`。并且不再支持形如 `(256, -1)` 的尺寸，请使用 [`ResizeEdge`](mmpretrain.datasets.transforms.ResizeEdge)
- [**`AutoAugment`**](mmpretrain.datasets.transforms.AutoAugment) 和 [**`RandAugment`**](mmpretrain.datasets.transforms.RandAugment) 中的 `policies` 参数现在支持使用字符串来指定某些预设的策略集，`AutoAugment` 支持 "imagenet"，`RandAugment` 支持 "timm_increasing"
- **`RandomResizedCrop`** 和 **`CenterCrop`** 不再支持 `efficientnet_style` 参数，请使用 [`EfficientNetRandomCrop`](mmpretrain.datasets.transforms.EfficientNetRandomCrop) 和 [`EfficientNetCenterCrop`](mmpretrain.datasets.transforms.EfficientNetCenterCrop)

```{note}
我们将一些数据变换工作移至数据预处理器进行，如归一化，请参阅[文档](mmpretrain.models.utils.data_preprocessor)了解更多详细信息。
```

<table class="docutils">
<tr>
<td>原配置</td>
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
<td>新配置</td>
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

**`evaluation`** 字段的变化：

- 原先的 **`evaluation`** 字段被拆分为 `val_evaluator` 和 `test_evaluator`，并且不再支持 `interval` 和 `save_best`
  参数。`interval` 参数被移动至 `train_cfg.val_interval` 字段，详见[训练策略配置](./user_guides/config.md#训练策略)。而 `save_best` 参数被移动至 `default_hooks.checkpoint.save_best` 字段，详见 [运行设置](./user_guides/config.md#运行设置)。
- 'accuracy' 指标被重命名为 [`Accuracy`](mmpretrain.evaluation.Accuracy)
- 'precision'，'recall'，'f1-score' 和 'support' 指标被组合为 [`SingleLabelMetric`](mmpretrain.evaluation.SingleLabelMetric)，并使用 `items` 参数指定具体计算哪些指标。
- 'mAP' 指标被重命名为 [`AveragePrecision`](mmpretrain.evaluation.AveragePrecision)
- 'CP'，'CR'，'CF1'，'OP'，'OR' 和 'OF1' 指标被组合为 [`MultiLabelMetric`](mmpretrain.evaluation.MultiLabelMetric)，并使用 `items` 和 `average` 参数指定具体计算哪些指标。

<table class="docutils">
<tr>
<td>原配置</td>
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
<td>新配置</td>
<td>

```python
val_evaluator = dict(type='Accuracy', topk=(1, 5))
test_evaluator = val_evaluator
```

</td>
</tr>
</table>
<table class="docutils">
<tr>
<td>原配置</td>
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
<td>新配置</td>
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

## 模块变动

### `mmpretrain.apis`

详见[包文档](mmpretrain.apis)

|         函数         | 变动                                                                                                                              |
| :------------------: | :-------------------------------------------------------------------------------------------------------------------------------- |
|     `init_model`     | 无变动                                                                                                                            |
|  `inference_model`   | 无变动，但我们推荐使用功能更强的 [`mmpretrain.ImageClassificationInferencer`](mmpretrain.apis.ImageClassificationInferencer)。    |
|    `train_model`     | 移除，直接使用 `runner.train` 进行训练。                                                                                          |
|   `multi_gpu_test`   | 移除，直接使用 `runner.test` 进行测试。                                                                                           |
|  `single_gpu_test`   | 移除，直接使用 `runner.test` 进行测试。                                                                                           |
| `show_result_pyplot` | 移除，使用 [`mmpretrain.ImageClassificationInferencer`](mmpretrain.apis.ImageClassificationInferencer) 进行模型推理和结果可视化。 |
|  `set_random_seed`   | 移除，使用 `mmengine.runner.set_random_seed`.                                                                                     |
|  `init_random_seed`  | 移除，使用 `mmengine.dist.sync_random_seed`.                                                                                      |

### `mmpretrain.core`

`mmpretrain.core` 包被重命名为 [`mmpretrain.engine`](mmpretrain.engine)

|      子包       | 变动                                                                                                                              |
| :-------------: | :-------------------------------------------------------------------------------------------------------------------------------- |
|  `evaluation`   | 移除，使用 [`mmpretrain.evaluation`](mmpretrain.evaluation)                                                                       |
|     `hook`      | 移动至 [`mmpretrain.engine.hooks`](mmpretrain.engine.hooks)                                                                       |
|  `optimizers`   | 移动至 [`mmpretrain.engine.optimizers`](mmpretrain.engine.optimizers)                                                             |
|     `utils`     | 移除，分布式环境相关的函数统一至 [`mmengine.dist`](api/dist) 包                                                                   |
| `visualization` | 移除，其中可视化相关的功能被移动至 [`mmpretrain.visualization.UniversalVisualizer`](mmpretrain.visualization.UniversalVisualizer) |

`hooks` 包中的 `MMClsWandbHook` 尚未实现。

`hooks` 包中的 `CosineAnnealingCooldownLrUpdaterHook` 被移除。我们现在支持使用学习率规划器的组合实现该功能。详见[自定义训练优化策略](./advanced_guides/schedule.md)。

### `mmpretrain.datasets`

详见[包文档](mmpretrain.datasets)

|                                         数据集类                                          | 变动                                                                     |
| :---------------------------------------------------------------------------------------: | :----------------------------------------------------------------------- |
|                   [`CustomDataset`](mmpretrain.datasets.CustomDataset)                    | 增加了 `data_root` 参数，作为 `data_prefix` 和 `ann_file` 的共同根路径。 |
|                        [`ImageNet`](mmpretrain.datasets.ImageNet)                         | 与 `CustomDataset` 相同。                                                |
|                     [`ImageNet21k`](mmpretrain.datasets.ImageNet21k)                      | 与 `CustomDataset` 相同。                                                |
|   [`CIFAR10`](mmpretrain.datasets.CIFAR10) & [`CIFAR100`](mmpretrain.datasets.CIFAR100)   | `test_mode` 参数目前是必要参数。                                         |
| [`MNIST`](mmpretrain.datasets.MNIST) & [`FashionMNIST`](mmpretrain.datasets.FashionMNIST) | `test_mode` 参数目前是必要参数。                                         |
|                             [`VOC`](mmpretrain.datasets.VOC)                              | 现在需要指定 `data_root`，`image_set_path` 和 `test_mode` 参数。         |
|                             [`CUB`](mmpretrain.datasets.CUB)                              | 现在需要指定 `data_root` 和 `test_mode` 参数。                           |

`mmpretrain.datasets.pipelines` 包被重命名为 `mmpretrain.datasets.transforms`

|           数据变换类            | 变动                                                                                                                                                                      |
| :-----------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
|       `LoadImageFromFile`       | 移除，使用 [`mmcv.transforms.LoadImageFromFile`](mmcv.transforms.LoadImageFromFile)                                                                                       |
|          `RandomFlip`           | 移除，使用 [`mmcv.transforms.RandomFlip`](mmcv.transforms.RandomFlip)，其中 `flip_prob` 参数被重命名为 `prob`                                                             |
|          `RandomCrop`           | `size` 参数被重命名为 `crop_size`                                                                                                                                         |
|       `RandomResizedCrop`       | `size` 参数被重命名为 `scale`；`scale` 参数被重命名为 `crop_ratio_range`；不再支持 `efficientnet_style`，请使用 [`EfficientNetRandomCrop`](mmpretrain.datasets.transforms.EfficientNetRandomCrop) |
|          `CenterCrop`           | 移除，使用 [`mmcv.transforms.CenterCrop`](mmcv.transforms.CenterCrop)；不再支持 `efficientnet_style`，请使用 [`EfficientNetCenterCrop`](mmpretrain.datasets.transforms.EfficientNetCenterCrop) |
|            `Resize`             | 移除，使用 [`mmcv.transforms.Resize`](mmcv.transforms.Resize)；`size` 参数被重命名为 `scale`，且不再支持形如 `(256, -1)` 参数，使用 [`ResizeEdge`](mmpretrain.datasets.transforms.ResizeEdge) |
| `AutoAugment` & `RandomAugment` | `policies` 参数现在支持使用字符串指定预设的策略集。                                                                                                                       |
|            `Compose`            | 移除，使用 [`mmcv.transforms.Compose`](mmcv.transforms.Compose)                                                                                                           |

### `mmpretrain.models`

详见[包文档](mmpretrain.models)，**backbones**、**necks** 和 **losses** 的结构没有变动。

[`ImageClassifier`](mmpretrain.models.classifiers.ImageClassifier) 的变动：

|  分类器的方法   | 变动                                                                                                                    |
| :-------------: | :---------------------------------------------------------------------------------------------------------------------- |
| `extract_feat`  | 无变动                                                                                                                  |
|    `forward`    | 现在需要三个输入：`inputs`、`data_samples` 和 `mode`。详见[文档](mmpretrain.models.classifiers.ImageClassifier.forward) |
| `forward_train` | 变更为 `loss` 方法。                                                                                                    |
|  `simple_test`  | 变更为 `predict` 方法。                                                                                                 |
|  `train_step`   | `optimizer` 参数被修改为 `optim_wrapper`，接受 [`OptimWrapper`](mmengine.optim.OptimWrapper)                            |
|   `val_step`    | 原先的 `val_step` 与 `train_step` 一致，现在该方法将会调用 `predict`                                                    |
|   `test_step`   | 新方法，与 `val_step` 一致。                                                                                            |

[heads](mmpretrain.models.heads) 中的变动：

|  分类头的方法   | 变动                                                                                                                                     |
| :-------------: | :--------------------------------------------------------------------------------------------------------------------------------------- |
|  `pre_logits`   | 无变动                                                                                                                                   |
| `forward_train` | 变更为 `loss` 方法。                                                                                                                     |
|  `simple_test`  | 变更为 `predict` 方法。                                                                                                                  |
|     `loss`      | 现在接受 `data_samples` 参数，而不是 `gt_labels`，`data_samples` 参数应当接受 [ClsDataSample](mmpretrain.structures.DataSample) 的列表。 |
|    `forward`    | 新方法，它将返回分类头的输出，不会进行任何后处理（包括 softmax 或 sigmoid）。                                                            |

### `mmpretrain.utils`

详见[包文档](mmpretrain.utils)

|             函数             | 变动                                                                                                          |
| :--------------------------: | :------------------------------------------------------------------------------------------------------------ |
|        `collect_env`         | 无变动                                                                                                        |
|      `get_root_logger`       | 移除，使用 [`mmengine.logging.MMLogger.get_current_instance`](mmengine.logging.MMLogger.get_current_instance) |
|       `load_json_log`        | 输出格式发生变化。                                                                                            |
|   `setup_multi_processes`    | 移除，使用 [`mmengine.utils.dl_utils.set_multi_processing`](mmengine.utils.dl_utils.set_multi_processing)     |
| `wrap_non_distributed_model` | 移除，现在 runner 会自动包装模型。                                                                            |
|   `wrap_distributed_model`   | 移除，现在 runner 会自动包装模型。                                                                            |
|     `auto_select_device`     | 移除，现在 runner 会自动选择设备。                                                                            |

# 从 MMSelfSup 0.x 迁移

## 配置文件

本章节将介绍 `_base_` 文件夹中的配置文件的变化，主要包含以下三个部分：

- 数据集：`configs/_base_/datasets`
- 模型：`configs/_base_/models`
- 优化器及调度：`configs/_base_/schedules`

### 数据集

在 **MMSelfSup 0.x** 中，我们使用字段 `data` 来整合数据相关信息, 例如 `samples_per_gpu`，`train`，`val` 等。

在 **MMPretrain 1.x** 中，我们分别使用字段  `train_dataloader`, `val_dataloader` 整理训练和验证的数据相关信息，并且 `data` 字段已经被 **移除**。

<table class="docutils">
<tr>
<td>旧版本</td>
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
<td>新版本</td>
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

另外，我们 **移除** 了字段 `data_source`，以此来保证我们项目和其它 OpenMMLab 项目数据流的一致性。请查阅 [Config](user_guides/config.md) 获取更详细的信息。

**`pipeline`** 中的变化：

以 MAE 的 `pipeline` 作为例子，新的写法如下：

```python
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        size=224,
        scale=(0.2, 1.0),
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSelfSupInputs', meta_keys=['img_path'])
]
```

### 模型

在模型的配置文件中，和 MMSeflSup 0.x 版本相比，主要有两点不同。

1. 有一个新的字段 `data_preprocessor`，主要负责对数据进行预处理，例如归一化，通道转换等。例子如下：

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

2. 在新版本的 `head` 字段中，我们新增加了 `loss`，主要负责损失函数的构建。例子如下：

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

## 模块变动

下列表格记录了代码模块、文件夹的主要改变。

| MMSelfSup 0.x            | MMPretrain 1.x      | Remark                                                                                                                                                        |
| ------------------------ | ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| apis                     | /                   | 目前 `apis` 文件夹已暂时被**移除**，在未来可能会再添加回来。                                                                                                  |
| core                     | engine              | `core` 文件夹重命名为 `engine`，包含了 `hooks`，`opimizers`。([API link](mmpretrain.engine))                                                                  |
| datasets                 | datasets            | 数据集相关类主要基于不同的数据集实现，例如 ImageNet，Places205。([API link](mmpretrain.datasets))                                                             |
| datasets/data_sources    | /                   | `data_sources` 已经被**移除**，并且现在 `datasets` 的逻辑和 OpenMMLab 其它项目保持一致。                                                                      |
| datasets/pipelines       | datasets/transforms | `pipelines` 文件夹已经重命名为 `transforms`。([API link](mmpretrain.datasets.transforms))                                                                     |
| /                        | evaluation          | `evaluation` 主要负责管理一些评测函数或者是类。([API link](mmpretrain.evaluation))                                                                            |
| models/algorithms        | selfsup             | 算法文件移动至 `selfsup` 文件夹。([API link](mmpretrain.models.selfsup))                                                                                      |
| models/backbones         | selfsup             | 自监督学习算法对应的，重新实现的主干网络移动到算法的 `.py` 文件中。([API link](mmpretrain.models.selfsup))                                                    |
| models/target_generators | selfsup             | 目标生成器的实现移动到算法的 `.py` 文件中。([API link](mmpretrain.models.selfsup))                                                                            |
| /                        | models/losses       | `losses` 文件夹提供了各种不同损失函数的实现。([API link](mmpretrain.models.losses))                                                                           |
| /                        | structures          | `structures` 文件夹提供了数据结构的实现。在 MMPretrain 中，我们实现了一种新的数据结构，`DataSample`，在训练/验证过程中来传输和接受数据信息。([API link](mmpretrain.structures)) |
| /                        | visualization       | `visualization` 文件夹包含了 visualizer，主要负责一些可视化的工作，例如数据增强的可视化。([API link](mmpretrain.visualization))                               |
