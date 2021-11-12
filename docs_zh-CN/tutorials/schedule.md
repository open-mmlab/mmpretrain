# 教程 6：如何自定义优化策略

在本教程中，我们将介绍如何在运行自定义模型时，进行构造优化器、定制学习率及动量调整策略、梯度裁剪、梯度累计以及用户自定义优化方法等。

<!-- TOC -->

- [构造 PyTorch 内置优化器](#构造-pytorch-内置优化器)
- [定制学习率调整策略](#定制学习率调整策略)
  - [学习率调整曲线](#定制学习率调整曲线)
  - [学习率预热策略](#定制学习率预热策略)
- [定制动量调整策略](#定制动量调整策略)
- [参数化精细配置](#参数化精细配置)
- [梯度裁剪与梯度累计](#梯度裁剪与梯度累计)
  - [梯度裁剪](#梯度裁剪)
  - [梯度累计](#梯度累计)
- [用户自定义优化方法](#用户自定义优化方法)
  - [自定义优化器](#自定义优化器)
  - [自定义优化器构造器](#自定义优化器构造器)

<!-- TOC -->

## 构造 PyTorch 内置优化器

MMClassification 支持 PyTorch 实现的所有优化器，仅需在配置文件中，指定 “optimizer” 字段。
例如，如果要使用 “SGD”，则修改如下。

```python
optimizer = dict(type='SGD', lr=0.0003, weight_decay=0.0001)
```

要修改模型的学习率，用户只需要在优化程序的配置中修改 “lr” 即可。
用户可根据 [PyTorch API 文档](https://pytorch.org/docs/stable/optim.html?highlight=optim#module-torch.optim) 进行参数设置。

```{note}
配置文件中的 'type' 不是构造时的参数，而是 PyTorch 内置优化器的类名。
```

例如，如果想使用 `Adam` 并设置参数为 `torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)`，
则需要进行如下修改

```python
optimizer = dict(type='Adam', lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
```

## 定制学习率调整策略

### 定制学习率调整曲线

在配置文件中使用默认值的逐步学习率调整，它调用 MMCV 中的 [`StepLRHook`](https://github.com/open-mmlab/mmcv/blob/f48241a65aebfe07db122e9db320c31b685dc674/mmcv/runner/hooks/lr_updater.py#L153)。

- Step:

    ```python
    lr_config = dict(policy='step', step=[100, 150])
    ```

此外，也支持其他学习率调整方法，如 `CosineAnnealing` 和 `Poly` 等。 详情可见 [这里](https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py)


- ConsineAnnealing:

    ```python
    lr_config = dict(policy='CosineAnnealing', min_lr_ratio=1e-5)
    ```

- Poly:

    ```python
    lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
    ```

### 定制学习率预热策略

在 MMClassification 中，使用 `lr_config` 配置学习率预热策略，主要的参数有以下几个：

- warmup : 学习率预热曲线类别，必须为 'constant'、 'linear'， 'exp' 或者 `None` 其一， 如果为 `None`, 则不使用学习率预热策略。
- warmup_by_epoch : 是否以轮次 (epoch) 预热。
- warmup_iters :  预热的迭代次数，当 `warmup_by_epoch=True` 时，单位为轮次 (epoch)；
    当 `warmup_by_epoch=False` 时，单位为迭代次数 (iter)。
- warmup_ratio : 预测的初始学习率 `lr = lr * warmup_ratio`。

例如：

1. linear & warmup by iter

    ```python
    lr_config = dict(
        policy='CosineAnnealing',
        by_epoch=False,
        min_lr_ratio=1e-2,
        warmup='linear',
        warmup_ratio=1e-3,
        warmup_iters=20 * 1252,
        warmup_by_epoch=False)
    ```

2. exp & warmup by epoch

    ```python
    lr_config = dict(
        policy='CosineAnnealing',
        min_lr=0,
        warmup='exp',
        warmup_iters=5,
        warmup_ratio=0.1,
        warmup_by_epoch=True)
    ```

**配置完成后，可以使用 MMClassification 提供的 [学习率可视化工具](https://mmclassification.readthedocs.io/zh_CN/latest/tools/visualization.html#id3) 画出对应学习率调整曲线。**

## 定制动量调整策略

MMClassification 支持动量调整器根据学习率修改模型的动量，从而使模型收敛更快。
动量调整程序通常与学习率调整器一起使用，例如，以下配置用于加速收敛。
更多细节可参考 [CyclicLrUpdater](https://github.com/open-mmlab/mmcv/blob/f48241a65aebfe07db122e9db320c31b685dc674/mmcv/runner/hooks/lr_updater.py#L327) 和 [CyclicMomentumUpdater](https://github.com/open-mmlab/mmcv/blob/f48241a65aebfe07db122e9db320c31b685dc674/mmcv/runner/hooks/momentum_updater.py#L130)。

```python
lr_config = dict(
    policy='cyclic',
    target_ratio=(10, 1e-4),
    cyclic_times=1,
    step_ratio_up=0.4,
)
momentum_config = dict(
    policy='cyclic',
    target_ratio=(0.85 / 0.95, 1),
    cyclic_times=1,
    step_ratio_up=0.4,
)
```

## 参数化精细配置

一些模型可能具有一些特定于参数的设置以进行优化，例如 BatchNorm 层不添加权重衰减或者对不同的网络层使用不同的学习率。
MMClassification 提供了 `paramwise_cfg` 进行配置，可以参考[MMCV](https://mmcv.readthedocs.io/en/latest/_modules/mmcv/runner/optimizer/default_constructor.html#DefaultOptimizerConstructor)。


- 使用指定选项

    MMClassification 提供了包括 `bias_lr_mult`、 `bias_decay_mult`、 `norm_decay_mult`、 `dwconv_decay_mult`、 `dcn_offset_lr_mult` 和 `bypass_duplicate` 选项，指定相关所有的 `bais`、 `norm`、 `dwconv`、 `dcn` 和 `bypass` 参数。例如：

    模型中所有的 BN 不进行参数衰减

    ```python
    paramwise_cfg = dict(norm_decay_mult=0.)
    ```

- 使用 `custom_keys` 指定参数

    MMClassification 可通过 `custom_keys` 指定不同的参数使用不同的学习率或者权重衰减，例如：

    对特定的参数不使用权重衰减

    ```python
    paramwise_cfg = dict(
        custom_keys={
            '.backbone.cls_token': dict(decay_mult=0.0),
            '.backbone.pos_embed': dict(decay_mult=0.0)
        })
    ```

    对 backbone 使用更小的学习率与衰减系数

    ```python
    paramwise_cfg = dict(custom_keys={'.backbone': dict(lr_mult=0.1, decay_mult=0.9)})s
    # backbone 的 'lr' and 'weight_decay' 分别为 0.1 * lr 和 0.9 * weight_decay
    ```

## 梯度裁剪与梯度累计

MMCV 在 PyTorch 基础优化器的基础上，对优化器的功能进行增强，例如梯度裁剪、梯度累计等，参考 [MMCV](https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/optimizer.py)。

### 梯度裁剪

训练过程中，异常点可能会导致一些模型梯度爆炸，需要使用梯度裁剪以稳定训练过程。
目前支持 `clip_grad_norm_`，可参考 [PyTorch 文档](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html)。
例子如下：

```python
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
```

不指定优化器钩子类型时，默认使用 `OptimizerHook`, 上述等价于：

```python
optimizer_config = dict(type="OptimizerHook", grad_clip=dict(max_norm=35, norm_type=2))
# norm_type: 使用的范数类型，此处使用范数2。
```

### 梯度累计

计算资源缺乏缺乏时，batchsize 只能设置较小的值，影响所得模型的效果，可以使用梯度累计来规避这一问题。
例子如下：

```python
optimizer_config = dict(type="GradientCumulativeOptimizerHook", cumulative_iters=4)
```

表示训练时，每4个 iter 执行一次反向传播。
如果此时的  `DataLoader` 的 batch_size 为 64，那么上述等价于：

```
loader = DataLoader(data, batch_size=256)
optim_hook = OptimizerHook()
```

```{note}
当在 `optimizer_config` 不指定优化器钩子类型时，默认使用 `OptimizerHook`。
```

## 用户自定义优化方法

在学术研究和工业实践中，可能需要使用 MMClassification 未实现的优化方法，用户可以通过以下方法添加。

```{note}
本部分将修改 MMClassification 源码或者向 MMClassification 框架添加代码，初学者可跳过。
```

### 自定义优化器

#### 1. 定义一个新的优化器

一个自定义的优化器可根据如下规则进行定制

假设用户想添加一个名为 `MyOptimzer` 的优化器，其拥有参数 `a`, `b` 和 `c`，
可以创建一个名为 `mmcls/core/optimizer` 的文件夹，并在目录下的文件进行构建，如 `mmcls/core/optimizer/my_optimizer.py`：

```python
from mmcv.runner import OPTIMIZERS
from torch.optim import Optimizer


@OPTIMIZERS.register_module()
class MyOptimizer(Optimizer):

    def __init__(self, a, b, c):

```

#### 2. 注册优化器

要找到上面定义的上述模块，首先应将此模块导入到主命名空间中。有两种方法可以实现它。

- 修改 `mmcls/core/optimizer/__init__.py` 来进行调用

    创建 `mmcls/core/optimizer/__init__.py` 文件。
    新定义的模块应导入到 `mmcls/core/optimizer/__init__.py` 中，以便注册器能找到新模块并将其添加：

```python
from .my_optimizer import MyOptimizer # MyOptimizer maybe other class name

__all__ = ['MyOptimizer']
```

- 在配置中使用 `custom_imports` 手动导入

```python
custom_imports = dict(imports=['mmcls.core.optimizer.my_optimizer'], allow_failed_imports=False)
```

`mmcls.core.optimizer.my_optimizer` 模块将会在程序开始阶段被导入，`MyOptimizer` 类会随之自动被注册。
注意，只有包含 `MyOptmizer` 类的包会被导入。`mmcls.core.optimizer.my_optimizer.MyOptimizer` **不会** 被直接导入。

#### 3. 在配置文件中指定优化器

之后，用户便可在配置文件的 `optimizer` 域中使用 `MyOptimizer`。
在配置中，优化器由 “optimizer” 字段定义，如下所示：

```python
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
```

要使用自定义的优化器，可以将该字段更改为

```python
optimizer = dict(type='MyOptimizer', a=a_value, b=b_value, c=c_value)
```

### 自定义优化器构造器

某些模型可能具有一些特定于参数的设置以进行优化，例如 BatchNorm 层的权重衰减。
用户可以通过自定义优化器构造函数来进行那些细粒度的参数调整。

```python
from mmcv.runner.optimizer import OPTIMIZER_BUILDERS


@OPTIMIZER_BUILDERS.register_module()
class MyOptimizerConstructor:

    def __init__(self, optimizer_cfg, paramwise_cfg=None):
        pass

    def __call__(self, model):
        ...    # 在这里实现自己的优化器构造器。
        return my_optimizer
```

默认的优化器构造器被创建于[此](https://github.com/open-mmlab/mmcv/blob/9ecd6b0d5ff9d2172c49a182eaa669e9f27bb8e7/mmcv/runner/optimizer/default_constructor.py#L11)，可被视为新优化器构造器的模板。
