# 自定义训练优化策略

在我们的算法库中，已经提供了通用数据集（如ImageNet，CIFAR）的[默认训练策略配置](https://github.com/open-mmlab/mmpretrain/blob/main/configs/_base_/schedules)。如果想要在这些数据集上继续提升模型性能，或者在不同数据集和方法上进行新的尝试，我们通常需要修改这些默认的策略。

在本教程中，我们将介绍如何在运行自定义训练时，通过修改配置文件进行构造优化器、参数化精细配置、梯度裁剪、梯度累计以及定制动量调整策略等。同时也会通过模板简单介绍如何自定义开发优化器和构造器。

## 配置训练优化策略

我们通过 `optim_wrapper` 来配置主要的优化策略，包括优化器的选择，混合精度训练的选择，参数化精细配置，梯度裁剪以及梯度累计。接下来将分别介绍这些内容。

### 构造 PyTorch 内置优化器

MMPretrain 支持 PyTorch 实现的所有优化器，仅需在配置文件中，指定优化器封装需要的 `optimizer` 字段。

如果要使用 [`SGD`](torch.optim.SGD)，则修改如下。这里要注意所有优化相关的配置都需要封装在 `optim_wrapper` 配置里。

```python
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.0003, weight_decay=0.0001)
)
```

```{note}
配置文件中的 'type' 不是构造时的参数，而是 PyTorch 内置优化器的类名。
更多优化器选择可以参考{external+torch:ref}`PyTorch 支持的优化器列表<optim:algorithms>`。
```

要修改模型的学习率，只需要在优化器的配置中修改 `lr` 即可。
要配置其他参数，可直接根据 [PyTorch API 文档](torch.optim) 进行。

例如，如果想使用 [`Adam`](torch.optim.Adam) 并设置参数为 `torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)`。
则需要进行如下修改：

```python
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer = dict(
        type='Adam',
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0,
        amsgrad=False),
)
```

````{note}
考虑到对于单精度训练来说，优化器封装的默认类型就是 `OptimWrapper`，我们在这里可以直接省略，因此配置文件可以进一步简化为：

```python
optim_wrapper = dict(
    optimizer=dict(
        type='Adam',
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0,
        amsgrad=False))
```
````

### 混合精度训练

如果我们想要使用混合精度训练（Automactic Mixed Precision），我们只需简单地将 `optim_wrapper` 的类型改为 `AmpOptimWrapper`。

```python
optim_wrapper = dict(type='AmpOptimWrapper', optimizer=...)
```

另外，为了方便，我们同时在启动训练脚本 `tools/train.py` 中提供了 `--amp` 参数作为开启混合精度训练的开关，更多细节可以参考[训练教程](../user_guides/train.md)。

### 参数化精细配置

在一些模型中，不同的优化策略需要适应特定的参数，例如不在 BatchNorm 层使用权重衰减，或者在不同层使用不同的学习率等等。
我们需要用到 `optim_wrapper` 中的 `paramwise_cfg` 参数来进行精细化配置。

- **为不同类型的参数设置超参乘子**

  例如，我们可以在 `paramwise_cfg` 配置中设置 `norm_decay_mult=0.` 来改变归一化层权重和偏移的衰减为0。

  ```python
  optim_wrapper = dict(
      optimizer=dict(type='SGD', lr=0.8, weight_decay=1e-4),
      paramwise_cfg=dict(norm_decay_mult=0.))
  ```

  支持更多类型的参数配置，参考以下列表：

  - `bias_lr_mult`：偏置的学习率系数（不包括正则化层的偏置以及可变形卷积的 offset），默认值为 1
  - `bias_decay_mult`：偏置的权值衰减系数（不包括正则化层的偏置以及可变形卷积的 offset），默认值为 1
  - `norm_decay_mult`：正则化层权重和偏置的权值衰减系数，默认值为 1
  - `flat_decay_mult`: 一维参数的权值衰减系数，默认值为 1
  - `dwconv_decay_mult`：Depth-wise 卷积的权值衰减系数，默认值为 1
  - `bypass_duplicate`：是否跳过重复的参数，默认为 `False`
  - `dcn_offset_lr_mult`：可变形卷积（Deformable Convolution）的学习率系数，默认值为 1

- **为特定参数设置超参乘子**

  MMPretrain 通过 `paramwise_cfg` 的 `custom_keys` 参数来配置特定参数的超参乘子。

  例如，我们可以通过以下配置来设置所有 `backbone.layer0` 层的学习率和权重衰减为0， `backbone` 的其余层和优化器保持一致，另外 `head` 层的学习率为0.001.

  ```python
  optim_wrapper = dict(
      optimizer=dict(type='SGD', lr=0.01, weight_decay=0.0001),
      paramwise_cfg=dict(
          custom_keys={
              'backbone.layer0': dict(lr_mult=0, decay_mult=0),
              'backbone': dict(lr_mult=1),
              'head': dict(lr_mult=0.1)
          }))
  ```

### 梯度裁剪

在训练过程中，损失函数可能接近于一些异常陡峭的区域，从而导致梯度爆炸。而梯度裁剪可以帮助稳定训练过程，更多介绍可以参见[该页面](https://paperswithcode.com/method/gradient-clipping)。

目前我们支持在 `optim_wrapper` 字段中添加 `clip_grad` 参数来进行梯度裁剪，更详细的参数可参考 [PyTorch 文档](torch.nn.utils.clip_grad_norm_)。

用例如下：

```python
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, weight_decay=0.0001),
    # norm_type: 使用的范数类型，此处使用范数2。
    clip_grad=dict(max_norm=35, norm_type=2))
```

### 梯度累计

计算资源缺乏缺乏时，每个训练批次的大小（batch size）只能设置为较小的值，这可能会影响模型的性能。

可以使用梯度累计来规避这一问题。我们支持在 `optim_wrapper` 字段中添加 `accumulative_counts` 参数来进行梯度累计。

用例如下：

```python
train_dataloader = dict(batch_size=64)
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, weight_decay=0.0001),
    accumulative_counts=4)
```

表示训练时，每 4 个 iter 执行一次反向传播。由于此时单张 GPU 上的批次大小为 64，也就等价于单张 GPU 上一次迭代的批次大小为 256，也即：

```python
train_dataloader = dict(batch_size=256)
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, weight_decay=0.0001))
```

## 配置参数优化策略

在训练过程中，优化参数例如学习率、动量，通常不会是固定不变，而是随着训练进程的变化而调整。PyTorch 支持一些学习率调整的调度器，但是不足以完成复杂的策略。在 MMPretrain 中，我们提供 `param_scheduler` 来更好地控制不同优化参数的策略。

### 配置学习率调整策略

深度学习研究中，广泛应用学习率衰减来提高网络的性能。我们支持大多数 PyTorch 学习率调度器， 其中包括 `ExponentialLR`, `LinearLR`, `StepLR`, `MultiStepLR` 等等。

- **单个学习率策略**

  多数情况下，我们使用单一学习率策略，这里 `param_scheduler` 会是一个字典。比如在默认的 ResNet 网络训练中，我们使用阶梯式的学习率衰减策略 [`MultiStepLR`](mmengine.optim.MultiStepLR)，配置文件为：

  ```python
  param_scheduler = dict(
      type='MultiStepLR',
      by_epoch=True,
      milestones=[100, 150],
      gamma=0.1)
  ```

  或者我们想使用 [`CosineAnnealingLR`](mmengine.optim.CosineAnnealingLR) 来进行学习率衰减：

  ```python
  param_scheduler = dict(
      type='CosineAnnealingLR',
      by_epoch=True,
      T_max=num_epochs)
  ```

- **多个学习率策略**

  然而在一些其他情况下，为了提高模型的精度，通常会使用多种学习率策略。例如，在训练的早期阶段，网络容易不稳定，而学习率的预热就是为了减少这种不稳定性。

  整个学习过程中，学习率将会通过预热从一个很小的值逐步提高到预定值，再会通过其他的策略进一步调整。

  在 MMPretrain 中，我们同样使用 `param_scheduler` ，将多种学习策略写成列表就可以完成上述预热策略的组合。

  例如：

  1. 在前50次迭代中逐**迭代次数**地**线性**预热

  ```python
    param_scheduler = [
        # 逐迭代次数，线性预热
        dict(type='LinearLR',
            start_factor=0.001,
            by_epoch=False,  # 逐迭代次数
            end=50),  # 只预热50次迭代次数
        # 主要的学习率策略
        dict(type='MultiStepLR',
            by_epoch=True,
            milestones=[8, 11],
            gamma=0.1)
    ]
  ```

  2. 在前10轮迭代中逐**迭代次数**地**线性**预热

  ```python
    param_scheduler = [
        # 在前10轮迭代中，逐迭代次数，线性预热
        dict(type='LinearLR',
            start_factor=0.001,
            by_epoch=True,
            end=10,
            convert_to_iter_based=True,  # 逐迭代次数更新学习率.
        ),
        # 在 10 轮次后，通过余弦退火衰减
        dict(type='CosineAnnealingLR', by_epoch=True, begin=10)
    ]
  ```

  注意这里增加了 `begin` 和 `end` 参数，这两个参数指定了调度器的**生效区间**。生效区间通常只在多个调度器组合时才需要去设置，使用单个调度器时可以忽略。当指定了 `begin` 和 `end` 参数时，表示该调度器只在 [begin, end) 区间内生效，其单位是由 `by_epoch` 参数决定。在组合不同调度器时，各调度器的 `by_epoch` 参数不必相同。如果没有指定的情况下，`begin` 为 0， `end` 为最大迭代轮次或者最大迭代次数。

  如果相邻两个调度器的生效区间没有紧邻，而是有一段区间没有被覆盖，那么这段区间的学习率维持不变。而如果两个调度器的生效区间发生了重叠，则对多组调度器叠加使用，学习率的调整会按照调度器配置文件中的顺序触发（行为与 PyTorch 中 [`ChainedScheduler`](torch.optim.lr_scheduler.ChainedScheduler) 一致）。

  ```{tip}
  为了避免学习率曲线与预期不符， 配置完成后，可以使用 MMPretrain 提供的 [学习率可视化工具](../useful_tools/scheduler_visualization.md) 画出对应学习率调整曲线。
  ```

### 配置动量调整策略

MMPretrain 支持动量调度器根据学习率修改优化器的动量，从而使损失函数收敛更快。用法和学习率调度器一致。

我们支持的动量策略和详细的使用细节可以参考[这里](https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/scheduler/momentum_scheduler.py)。我们只将调度器中的 `LR` 替换为了 `Momentum`，动量策略可以直接追加 `param_scheduler` 列表中。

这里是一个用例：

```python
param_scheduler = [
    # 学习率策略
    dict(type='LinearLR', ...),
    # 动量策略
    dict(type='LinearMomentum',
         start_factor=0.001,
         by_epoch=False,
         begin=0,
         end=1000)
]
```

## 新增优化器或者优化器构造器

```{note}
本部分将修改 MMPretrain 源码或者向 MMPretrain 框架添加代码，初学者可跳过。
```

### 新增优化器

在学术研究和工业实践中，可能需要使用 MMPretrain 未实现的优化方法，可以通过以下方法添加。

1. 定义一个新的优化器

   一个自定义的优化器可根据如下规则进行定制：

   假设我们想添加一个名为 `MyOptimzer` 的优化器，其拥有参数 `a`, `b` 和 `c`。
   可以创建一个名为 `mmpretrain/engine/optimizer` 的文件夹，并在目录下的一个文件，如 `mmpretrain/engine/optimizer/my_optimizer.py` 中实现该自定义优化器：

   ```python
   from mmpretrain.registry import OPTIMIZERS
   from torch.optim import Optimizer


   @OPTIMIZERS.register_module()
   class MyOptimizer(Optimizer):

       def __init__(self, a, b, c):
           ...

       def step(self, closure=None):
           ...
   ```

2. 注册优化器

   要注册上面定义的上述模块，首先需要将此模块导入到主命名空间中。有两种方法可以实现它。

   修改 `mmpretrain/engine/optimizers/__init__.py`，将其导入至 `mmpretrain.engine` 包。

   ```python
   # 在 mmpretrain/engine/optimizers/__init__.py 中
   ...
   from .my_optimizer import MyOptimizer # MyOptimizer 是我们自定义的优化器的名字

   __all__ = [..., 'MyOptimizer']
   ```

   在运行过程中，我们会自动导入 `mmpretrain.engine` 包并同时注册 `MyOptimizer`。

3. 在配置文件中指定优化器

   之后，用户便可在配置文件的 `optim_wrapper.optimizer` 域中使用 `MyOptimizer`：

   ```python
   optim_wrapper = dict(
       optimizer=dict(type='MyOptimizer', a=a_value, b=b_value, c=c_value))
   ```

### 新增优化器构造器

某些模型可能具有一些特定于参数的设置以进行优化，例如为所有 BatchNorm 层设置不同的权重衰减。

尽管我们已经可以使用 [`optim_wrapper.paramwise_cfg` 字段](#参数化精细配置)来配置特定参数的优化设置，但可能仍然无法覆盖你的需求。

当然你可以在此基础上进行修改。我们默认使用 [`DefaultOptimWrapperConstructor`](mmengine.optim.DefaultOptimWrapperConstructor) 来构造优化器。在构造过程中，通过 `paramwise_cfg` 来精细化配置不同设置。这个默认构造器可以作为新优化器构造器实现的模板。

我们可以新增一个优化器构造器来覆盖这些行为。

```python
# 在 mmpretrain/engine/optimizers/my_optim_constructor.py 中
from mmengine.optim import DefaultOptimWrapperConstructor
from mmpretrain.registry import OPTIM_WRAPPER_CONSTRUCTORS


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class MyOptimWrapperConstructor:

    def __init__(self, optim_wrapper_cfg, paramwise_cfg=None):
        ...

    def __call__(self, model):
        ...
```

这是一个已实现的 [OptimWrapperConstructor](mmpretrain.engine.optimizers.LearningRateDecayOptimWrapperConstructor) 具体例子。

接下来类似 [新增优化器教程](#新增优化器) 来导入并使用新的优化器构造器。

1. 修改 `mmpretrain/engine/optimizers/__init__.py`，将其导入至 `mmpretrain.engine` 包。

   ```python
   # 在 mmpretrain/engine/optimizers/__init__.py 中
   ...
   from .my_optim_constructor import MyOptimWrapperConstructor

   __all__ = [..., 'MyOptimWrapperConstructor']
   ```

2. 在配置文件的 `optim_wrapper.constructor` 字段中使用 `MyOptimWrapperConstructor` 。

   ```python
   optim_wrapper = dict(
       constructor=dict(type='MyOptimWrapperConstructor'),
       optimizer=...,
       paramwise_cfg=...,
   )
   ```
