# 自定义实验运行参数

模型运行参数文件包括许多有用的功能，如权重文件保存、日志配置等等，在本教程中，我们将介绍如何配置这些功能。

<!-- TODO: Link to MMEngine docs instead of API reference after the MMEngine English docs is done. -->

## 保存权重文件

权重文件保存功能是一个默认训练钩子。你可以在配置文件的`default_hooks.checkpoint`字段中对其配置。


```{note}
钩子机制在 OpenMMLab 开源算法库中应用非常广泛。通过钩子，你可以在不修改运行器的主要执行逻辑的情况下插入许多功能。

可以通过{external+mmengine:doc}`相关文章 <tutorials/hook>`进一步理解钩子。
```

**默认配置:**

```python
default_hooks = dict(
    ...
    checkpoint = dict(type='CheckpointHook', interval=1)
    ...
)
```

下面是一些常用的参数，所有可用的参数都可以在[权重文件钩子(CheckpointHook)](mmengine.hooks.CheckpointHook)中找到。

- **`interval`** (int): 文件保存周期。如果使用-1，它将永远不会保存权重。
- **`by_epoch`** (bool): 选择 **`interval`** 是基于epoch还是基于iteration， 默认为 `True`.
- **`out_dir`** (str): 保存检查点的根目录。如果不指定，检查点将被保存在工作目录中。如果指定，检查点将被保存在 **`out_dir`** 的子文件夹中。
- **`max_keep_ckpts`** (int): 要保留的权重文件数量。在某些情况下，我们只想要最近的几个检查点，并希望删除旧的检查点以节省磁盘空间。默认为-1，也就是无限制。
- **`save_best`** (str, List[str]): 如果指定，它将保存具有最佳评估结果的权重。
  通常情况下，你可以直接使用`save_best="auto"`来自动选择评估指标。
  而如果你想要更高级的配置，请参考[权重文件钩子(CheckpointHook)](mmengine.hooks.CheckpointHook)。

## 权重加载 / 断点训练

在配置文件中，你可以指定模型权重加载和断点继续训练的功能，如下所示:

```python
# 从指定权重文件加载
load_from = "Your checkpoint path"

# 是否从加载的断点继续训练
resume = False
```

`load_from`字段可以是本地路径，也可以是HTTP路径。你可以从检查点恢复训练，方法是指定 `resume=True`。

```{tip}
你也可以通过指定 `load_from=None` 和 `resume=True` 启用从最新的断点自动恢复。
Runner执行器将自动从工作目录中找到最新的权重文件。
```

如果你用我们的`tools/train.py`脚本来训练模型，你也可以使用`--resume`参数来恢复训练，而不用手动修改配置文件。

```bash
# 自动从最新的断点恢复
python tools/train.py configs/resnet/resnet50_8xb32_in1k.py --resume

# 从指定的断点恢复
python tools/train.py configs/resnet/resnet50_8xb32_in1k.py --resume checkpoints/resnet.pth
```

## 随机性(Randomness)配置

在 `randomness` 字段配置中，我们提供了一些选项，以使实验尽可能的可重复。

默认情况下，我们不会在配置文件中指定随机数种子，在每次实验中，程序会生成一个不同的随机数种子。

**默认配置:**

```python
randomness = dict(seed=None, deterministic=False)
```

为了使实验更具可重复性，你可以指定一个种子并设置`deterministic=True`。
`deterministic`选项的使用效果可以在[这里](https://pytorch.org/docs/stable/notes/randomness.html#cuda-convolution-benchmarking)找到。


## 日志配置

日志的配置与多个字段有关。

在`log_level`字段中，你可以指定全局日志级别。参见 {external+python:ref}`Logging Levels<levels>` 以获得级别列表。

```python
log_level = 'INFO'
```

在 `default_hooks.logger` 字段中，你可以指定训练和测试期间的日志间隔。
而所有可用的参数可以在[日志钩子文档](mmengine.hooks.LoggerHook)中找到。

```python
default_hooks = dict(
    ...
    # 每100次迭代就打印一次日志
    logger=dict(type='LoggerHook', interval=100),
    ...
)
```

在 `log_processor` 字段中，你可以指定日志信息的平滑方法。
通常，我们使用一个长度为10的窗口来平滑日志中的值，并输出所有信息的平均值。
如果你想特别指定某些信息的平滑方法，请参阅[日志处理器文档](mmengine.runner.LogProcessor)

```python
# 默认设置，它将通过一个10长度的窗口平滑训练日志中的值
log_processor = dict(window_size=10)
```

在 `visualizer` 字段中，你可以指定多个后端来保存日志信息，如TensorBoard和WandB。
更多的细节可以在[可视化工具](#visualizer)找到。

## 自定义钩子

上述许多功能是由钩子实现的，你也可以通过修改 `custom_hooks` 字段来插入其他的自定义钩子。
下面是MMEngine和MMClassification中的一些钩子，你可以直接使用，例如：

- [EMAHook](mmcls.engine.hooks.EMAHook)
- [SyncBuffersHook](mmengine.hooks.SyncBuffersHook)
- [EmptyCacheHook](mmengine.hooks.EmptyCacheHook)
- [ClassNumCheckHook](mmcls.engine.hooks.ClassNumCheckHook)
- ......

例如，EMA（Exponential Moving Average）在模型训练中被广泛使用，你可以如下方式启用它：

```python
custom_hooks = [
    dict(type='EMAHook', momentum=4e-5, priority='ABOVE_NORMAL'),
]
```

## 可视化验证(Visualize Validation)

验证的可视化功能是验证过程中的一个默认注册的钩子。
你可以在 `default_hooks.visualization` 字段中来配置它。

默认情况下，我们禁用这个钩子，你可以通过指定`enable=True`来启用它。而更多的参数可以在
[可视化钩子文档](mmcls.engine.hooks.VisualizationHook)中找到。

```python
default_hooks = dict(
    ...
    visualization=dict(type='VisualizationHook', enable=False),
    ...
)
```

这个钩子将在验证数据集中选择一些图像，并在每次验证过程中对这些图像的预测结果进行标记。
你可以用它来观察训练期间模型在实际图像上的性能变化。

此外，如果你的验证数据集中的图像很小（\<100），你可以在可视化之前通过指定 `rescale_factor=2.` 或更高来重新缩放它们。

## Visualizer

Visualizer用于记录训练和测试过程中的各种信息，包括日志、图像和标量。
默认情况下，记录的信息将被保存在工作目录下的 `vis_data` 文件夹中。

**默认配置:**

```python
visualizer = dict(
    type='ClsVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ]
)
```

通常，最有用的功能是将日志和标量如 `loss` 保存到不同的后端。
例如，要把它们保存到TensorBoard，只需像下面这样设置：

```python
visualizer = dict(
    type='ClsVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ]
)
```

或者像下面这样把它们保存到WandB：

```python
visualizer = dict(
    type='ClsVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='WandbVisBackend'),
    ]
)
```

## 环境配置

在 `env_cfg` 字段中，你可以配置一些底层的参数，如cuDNN、多进程和分布式通信。

**在修改这些参数之前，请确保你理解这些参数的含义。**

```python
env_cfg = dict(
    # 是否启用cudnn基准测试
    cudnn_benchmark=False,

    # 设置多进程参数
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),

    # 设置分布式参数
    dist_cfg=dict(backend='nccl'),
)
```

## FAQ

1. ** `load_from` 和 `init_cfg` 之间的关系是什么？**

   - `load_from`: 如果`resume=False`，只导入模型权重，主要用于加载训练过的模型；
     如果`resume=True`，加载所有的模型权重、优化器状态和其他训练信息，这主要用于恢复中断的训练。

   - `init_cfg`: 你也可以指定`init=dict(type="Pretrained", checkpoint=xxx)`来加载权重，
     这意味着在模型权重初始化时加载权重。 也就是说，它只在训练的开始阶段进行。
     它主要用于微调预训练的模型，你可以在骨干配置中设置它，并使用`prefix` 字段来只加载骨干权重，例如：

     ```python
     model = dict(
        backbone=dict(
            type='ResNet',
            depth=50,
            init_cfg=dict(type='Pretrained', checkpoints=xxx, prefix='backbone'),
        )
        ...
     )
     ```

     参见 [微调模型](../user_guides/finetune.md) 以了解更多关于模型微调的细节。

2. ** `default_hooks` 和 `custom_hooks` 之间有什么区别？**
   
   几乎没有区别。通常，`default_hooks` 字段用于指定几乎所有实验都会使用的钩子，
   而`custom_hooks`字段只用于一些实验。

   另一个区别是 `default_hooks` 是一个字典，而 `custom_hooks` 是一个列表，请不要混淆。

3. ** 在训练期间，我没有收到训练日志，这是什么原因？ **

   如果你的训练数据集很小，而批处理量却很大，我们默认的日志间隔可能太大，无法记录你的训练日志。

   你可以缩减日志间隔，再试一次，比如:

   ```python
   default_hooks = dict(
       ...
       logger=dict(type='LoggerHook', interval=10),
       ...
   )
   ```

