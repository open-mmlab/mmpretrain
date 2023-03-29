# 自定义运行参数

运行参数配置包括许多有用的功能，如权重文件保存、日志配置等等，在本教程中，我们将介绍如何配置这些功能。

## 保存权重文件

权重文件保存功能是一个在训练阶段默认注册的钩子， 你可以通过配置文件中的 `default_hooks.checkpoint` 字段配置它。

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

下面是一些[权重文件钩子(CheckpointHook)](mmengine.hooks.CheckpointHook)的常用可配置参数。

- **`interval`** (int): 文件保存周期。如果使用-1，它将永远不会保存权重。
- **`by_epoch`** (bool): 选择 **`interval`** 是基于epoch还是基于iteration， 默认为 `True`.
- **`out_dir`** (str): 保存权重文件的根目录。如果不指定，检查点将被保存在工作目录中。如果指定，检查点将被保存在 **`out_dir`** 的子文件夹中。
- **`max_keep_ckpts`** (int): 要保留的权重文件数量。在某些情况下，为了节省磁盘空间，我们希望只保留最近的几个权重文件。默认为 -1，也就是无限制。
- **`save_best`** (str, List[str]): 如果指定，它将保存具有最佳评估结果的权重。
  通常情况下，你可以直接使用`save_best="auto"`来自动选择评估指标。

而如果你想要更高级的配置，请参考[权重文件钩子(CheckpointHook)](tutorials/hook.md#checkpointhook)。

## 权重加载 / 断点训练

在配置文件中，你可以加载指定模型权重或者断点继续训练，如下所示:

```python
# 从指定权重文件加载
load_from = "Your checkpoint path"

# 是否从加载的断点继续训练
resume = False
```

`load_from` 字段可以是本地路径，也可以是HTTP路径。你可以从检查点恢复训练，方法是指定 `resume=True`。

```{tip}
你也可以通过指定 `load_from=None` 和 `resume=True` 启用从最新的断点自动恢复。
Runner执行器将自动从工作目录中找到最新的权重文件。
```

如果你用我们的 `tools/train.py` 脚本来训练模型，你只需使用 `--resume` 参数来恢复训练，就不用手动修改配置文件了。如下所示:

```bash
# 自动从最新的断点恢复
python tools/train.py configs/resnet/resnet50_8xb32_in1k.py --resume

# 从指定的断点恢复
python tools/train.py configs/resnet/resnet50_8xb32_in1k.py --resume checkpoints/resnet.pth
```

## 随机性(Randomness)配置

为了让实验尽可能是可复现的， 我们在 `randomness` 字段中提供了一些控制随机性的选项。

默认情况下，我们不会在配置文件中指定随机数种子，在每次实验中，程序会生成一个不同的随机数种子。

**默认配置:**

```python
randomness = dict(seed=None, deterministic=False)
```

为了使实验更具可复现性，你可以指定一个种子并设置 `deterministic=True`。
`deterministic` 选项的使用效果可以在[这里](https://pytorch.org/docs/stable/notes/randomness.html#cuda-convolution-benchmarking)找到。

## 日志配置

日志的配置与多个字段有关。

在`log_level`字段中，你可以指定全局日志级别。参见 {external+python:ref}`Logging Levels<levels>` 以获得日志级别列表。

```python
log_level = 'INFO'
```

在 `default_hooks.logger` 字段中，你可以指定训练和测试期间的日志间隔。
而所有可用的参数可以在[日志钩子文档](tutorials/hook.md#loggerhook)中找到。

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
如果你想特别指定某些信息的平滑方法，请参阅{external+mmengine:doc}`日志处理器文档 <advanced_tutorials/logging>`。

```python
# 默认设置，它将通过一个10长度的窗口平滑训练日志中的值
log_processor = dict(window_size=10)
```

在 `visualizer` 字段中，你可以指定多个后端来保存日志信息，如TensorBoard和WandB。
更多的细节可以在[可视化工具](#visualizer)找到。

## 自定义钩子

上述许多功能是由钩子实现的，你也可以通过修改 `custom_hooks` 字段来插入其他的自定义钩子。
下面是 MMEngine 和 MMPretrain 中的一些钩子，你可以直接使用，例如：

- [EMAHook](mmpretrain.engine.hooks.EMAHook)
- [SyncBuffersHook](mmengine.hooks.SyncBuffersHook)
- [EmptyCacheHook](mmengine.hooks.EmptyCacheHook)
- [ClassNumCheckHook](mmpretrain.engine.hooks.ClassNumCheckHook)
- ......

例如，EMA（Exponential Moving Average）在模型训练中被广泛使用，你可以以下方式启用它：

```python
custom_hooks = [
    dict(type='EMAHook', momentum=4e-5, priority='ABOVE_NORMAL'),
]
```

## 验证可视化

验证可视化钩子是一个验证过程中默认注册的钩子。
你可以在 `default_hooks.visualization` 字段中来配置它。

默认情况下，我们禁用这个钩子，你可以通过指定 `enable=True` 来启用它。而更多的参数可以在
[可视化钩子文档](mmpretrain.engine.hooks.VisualizationHook)中找到。

```python
default_hooks = dict(
    ...
    visualization=dict(type='VisualizationHook', enable=False),
    ...
)
```

这个钩子将在验证数据集中选择一部分图像，在每次验证过程中记录并可视化它们的预测结果。
你可以用它来观察训练期间模型在实际图像上的性能变化。

此外，如果你的验证数据集中的图像很小（\<100， 如Cifra数据集），
你可以指定 `rescale_factor` 来缩放它们，如 `rescale_factor=2.`, 将可视化的图像放大两倍。

## Visualizer

`Visualizer` 用于记录训练和测试过程中的各种信息，包括日志、图像和标量。
默认情况下，记录的信息将被保存在工作目录下的 `vis_data` 文件夹中。

**默认配置:**

```python
visualizer = dict(
    type='UniversalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ]
)
```

通常，最有用的功能是将日志和标量如 `loss` 保存到不同的后端。
例如，要把它们保存到 TensorBoard，只需像下面这样设置：

```python
visualizer = dict(
    type='UniversalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ]
)
```

或者像下面这样把它们保存到 WandB：

```python
visualizer = dict(
    type='UniversalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='WandbVisBackend'),
    ]
)
```

## 环境配置

在 `env_cfg` 字段中，你可以配置一些底层的参数，如 cuDNN、多进程和分布式通信。

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
