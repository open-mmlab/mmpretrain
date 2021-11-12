# 教程 7：如何自定义模型运行参数

在本教程中，我们将介绍如何在运行自定义模型时，进行自定义工作流和钩子的方法。

<!-- TOC -->

- [定制工作流](#定制工作流)
- [钩子](#钩子)
  - [默认训练钩子](#默认训练钩子)
    - [权重文件钩子](#权重文件钩子)
    - [日志钩子](#日志钩子)
    - [验证钩子](#验证钩子)
  - [使用内置钩子](#使用内置钩子)
  - [自定义钩子](#自定义钩子)
    - [1. 创建一个新钩子](#1-创建一个新钩子)
    - [2. 注册新钩子](#2-注册新钩子)
    - [3. 修改配置](#3-修改配置)
- [常见问题](#常见问题)

<!-- TOC -->

## 定制工作流

默认情况下，MMClassification 推荐用户在训练周期中使用 **`EvalHook`** 进行模型验证，也可以选择 `val` 工作流模型进行模型验证。

工作流是一个形如 (工作流名, 周期数) 的列表，用于指定运行顺序和周期。其默认设置为：

```python
workflow = [('train', 1)]
```

其代表要进行一轮周期的训练。
有时，用户可能希望检查有关验证集中模型的某些指标（例如，损失，准确性）。
在这种情况下，可以将工作流程设置为

```python
[('train', 1), ('val', 1)]
```

从而将迭代运行1个训练时间和1个验证时间。

```{note}
1. 在验证周期时不会更新模型参数。
2. 配置文件内的关键词 `max_epochs` 控制训练时期数，并且不会影响验证工作流程。
3. 工作流 `[('train', 1), ('val', 1)]` 和 `[('train', 1)]` 不会改变 `EvalHook` 的行为。
   因为 `EvalHook` 由 `after_train_epoch` 调用，而验证工作流只会影响 `after_val_epoch` 调用的钩子。
   因此，`[('train', 1), ('val', 1)]` 和 ``[('train', 1)]`` 的区别在于，runner 在完成每一轮训练后，会计算验证集上的损失。
```

## 钩子

钩子机制在 OpenMMLab 开源算法库中应用非常广泛，结合执行器可以实现对训练过程的整个生命周期进行管理，可以通过[相关文章](https://zhuanlan.zhihu.com/p/355272220)进一步理解钩子。

钩子只有在构造器中被注册才起作用，目前钩子主要分为两类：

- 默认训练钩子

默认训练钩子由运行器默认注册，一般为一些基础型功能的钩子，已经有确定的优先级，一般不需要修改优先级。

- 定制钩子

定制钩子通过 `custom_hooks` 注册，一般为一些增强型功能的钩子，需要在配置文件中指定优先级，不指定该钩子的优先级将默被设定为 'NORMAL'。

**优先级列表**

| Level           | Value      |
|:--:|:--:|
| HIGHEST         | 0          |
| VERY_HIGH       | 10         |
| HIGH            | 30         |
| ABOVE_NORMAL    | 40         |
| NORMAL(default) | 50         |
| BELOW_NORMAL    | 60         |
| LOW             | 70         |
| VERY_LOW        | 90         |
| LOWEST          | 100        |

优先级确定钩子的执行顺序，每次训练前，日志会打印出各个阶段钩子的执行顺序，方便调试。

### 默认训练钩子

有一些常见的钩子未通过 `custom_hooks` 注册，但会在运行器（`Runner`）中默认注册，它们是：

| Hooks                  | Priority                |
|:--:|:--:|
| `LrUpdaterHook`        | VERY_HIGH (10)          |
| `MomentumUpdaterHook`  | HIGH (30)               |
| `OptimizerHook`        | ABOVE_NORMAL (40)       |
| `CheckpointSaverHook`  | NORMAL (50)             |
| `IterTimerHook`        | LOW (70)                |
| `EvalHook`             | LOW (70)                |
| `LoggerHook(s)`        | VERY_LOW (90)           |


`OptimizerHook`，`MomentumUpdaterHook`和 `LrUpdaterHook` 已经在 [优化策略](./sehedule) 介绍过了，
`IterTimerHook` 用于记录所用时间，目前不支持修改;

下面介绍如何使用去定制 `LoggerHooks`、`CheckpointHook` 以及 `EvalHook`。

#### 权重文件钩子

MMCV 的 runner 使用 `checkpoint_config` 来初始化 [`CheckpointHook`](https://github.com/open-mmlab/mmcv/blob/9ecd6b0d5ff9d2172c49a182eaa669e9f27bb8e7/mmcv/runner/hooks/checkpoint.py#L9)。

```python
checkpoint_config = dict(interval=1)
```

用户可以设置 “max_keep_ckpts” 来仅保存少量模型权重文件，或者通过 “save_optimizer” 决定是否存储优化器的状态字典。
更多细节可参考 [这里](https://mmcv.readthedocs.io/zh_CN/latest/api.html#mmcv.runner.CheckpointHook)。

#### 日志钩子

`log_config` 包装了多个记录器钩子，并可以设置间隔。
目前，MMCV 支持 `TextLoggerHook`、 `WandbLoggerHook`、`MlflowLoggerHook` 和 `TensorboardLoggerHook`。
更多细节可参考[这里](https://mmcv.readthedocs.io/zh_CN/latest/api.html#mmcv.runner.LoggerHook)。

```python
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
```

#### 验证钩子

评估的配置将用于初始化 [`EvalHook`](https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/evaluation.py)。
除了键 `interval` 外，其他参数，如 “metrics” 也将传递给 `dataset.evaluate()`。

```python
evaluation = dict(interval=1, metric='accuracy', metric_options={'topk': (1, )})
```

也可以通过修改参数 `save_best` 保存取得最好验证结果时的模型权重：

```python
evaluation = dict(interval=1, save_best=True, metric='accuracy', metric_options={'topk': (1, )})
```

在跑一些大型实验时，可以通过修改参数 `start` 跳过训练靠前轮次时的验证步骤，以节约时间。如下：

```python
evaluation = dict(interval=1, start=200, metric='accuracy', metric_options={'topk': (1, )})
```

表示在第 200 轮之前，只执行训练流程，不执行验证；从轮次 200 开始，在每一轮训练之后进行验证。

```{note}
在 MMClassification 的默认配置文件中， evaluation 字段一般被放在 datasets 基础配置文件中。
```

### 使用内置钩子

一些钩子已在 MMCV 和 MMClassification 中实现：

- [EMAHook](https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/ema.py)
- [SyncBuffersHook](https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/sync_buffer.py)
- [EmptyCacheHook](https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/memory.py)
- [ProfilerHook](https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/profiler.py)
- ......

可以直接修改配置以使用该钩子，如下格式：

```python
custom_hooks = [
    dict(type='MMCVHook', a=a_value, b=b_value, priority='NORMAL')
]
```

例如使用 `EMAHook`，进行一次 EMA 的间隔是100个迭代：

```python
custom_hooks = [
    dict(type='EMAHook', interval=100, priority='HIGH')
]
```

## 自定义钩子

### 1. 创建一个新钩子

这里举一个在 MMClassification 中创建一个新钩子，并在训练中使用它的示例：

```python
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class MyHook(Hook):

    def __init__(self, a, b):
        pass

    def before_run(self, runner):
        pass

    def after_run(self, runner):
        pass

    def before_epoch(self, runner):
        pass

    def after_epoch(self, runner):
        pass

    def before_iter(self, runner):
        pass

    def after_iter(self, runner):
        pass
```

根据钩子的功能，用户需要指定钩子在训练的每个阶段将要执行的操作，比如 `before_run`，`after_run`，`before_epoch`，`after_epoch`，`before_iter` 和 `after_iter`。

### 2. 注册新钩子

之后，需要导入 `MyHook`。假设该文件在 `mmcls/core/utils/my_hook.py`，有两种办法导入它：

- 修改 `mmcls/core/utils/__init__.py` 进行导入

    新定义的模块应导入到 `mmcls/core/utils/__init__py` 中，以便注册表能找到并添加新模块：

```python
from .my_hook import MyHook

__all__ = ['MyHook']
```

- 使用配置文件中的 `custom_imports` 变量手动导入

```python
custom_imports = dict(imports=['mmcls.core.utils.my_hook'], allow_failed_imports=False)
```

### 3. 修改配置

```python
custom_hooks = [
    dict(type='MyHook', a=a_value, b=b_value)
]
```

还可通过 `priority` 参数设置钩子优先级，如下所示：

```python
custom_hooks = [
    dict(type='MyHook', a=a_value, b=b_value, priority='NORMAL')
]
```

默认情况下，在注册过程中，钩子的优先级设置为 “NORMAL”。


## 常见问题

### 1. resume_from， load_from，init_cfg.Pretrained 区别

- `load_from` ：仅仅加载模型权重，主要用于加载预训练或者训练好的模型；

- `resume_from` ：不仅导入模型权重，还会导入优化器信息，当前轮次（epoch）信息，主要用于从断点继续训练。

- `init_cfg.Pretrained` ：在权重初始化期间加载权重，您可以指定要加载的模块。 这通常在微调模型时使用，请参阅[教程 2：如何微调模型](./finetune.md)
