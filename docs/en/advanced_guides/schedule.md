# Customize Training Schedule

In our codebase, [default training schedules](https://github.com/open-mmlab/mmclassification/blob/master/configs/_base_/schedules) have beed provided for common datasets such as CIFAR, ImageNet, etc. If we attempt to experiment on these datasets for higher accuracy or on different new methods and datasets, we might possibly need to modify the strategies.

In this tutorial, we will introduce how to modify configs to construct optimizers, use parameter-wise finely configuration, gradient clipping, gradient accumulation

learning rate and momentum schedules, as well as how to use parameter-wise finely configuration, gradient clipping, gradient accumulation, and customize self-implemented methods for the project.

<!-- TOC -->

- [Customize optimization](#customize-optimization)
  - [Use optimizers supported by PyTorch](#use-optimizers-supported-by-pytorch)
  - [Use AMP training](#use-amp-training)
  - [Parameter-wise finely configuration](#parameter-wise-finely-configuration)
  - [Gradient clipping](#gradient-clipping)
  - [Gradient accumulation](#gradient-accumulation)
- [Customize parameter schedules](#customize-parameter-schedules)
  - [Customize learning rate schedules](#customize-learning-rate-schedules)
  - [Customize momentum schedules](#customize-momentum-schedules)
- [Customize self-implemented methods](#customize-self-implemented-methods)
  - [Customize self-implemented optimizer](#customize-self-implemented-optimizer)
  - [Customize optimizer constructor](#customize-optimizer-constructor)

<!-- TOC -->

## Customize optimization

We use a wrapper for major strategies of optimization, which includes choices of optimizer, choices of automatic mixed precision training, parameter-wise configurations, gradient clipping and accumulation.

### Use optimizers supported by PyTorch

We support all the optimizers implemented by PyTorch, and to use them, please change the `optimizer` field of config files. Refers to [List of optimizers supported by PyTorch](https://pytorch.org/docs/stable/optim.html#algorithms) for more details.

For example, if you want to use `SGD`, the modification in config file could be as the following. Notice that optimization related settings should all wrapped inside the `OptimWrapper`.

```python
optimizer = dict(type='SGD', lr=0.0003, weight_decay=0.0001)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)
```

To modify the learning rate of the model, just modify the `lr` in the config of optimizer.
You can also directly set other arguments according to the [API doc](https://pytorch.org/docs/stable/optim.html?highlight=optim#module-torch.optim) of PyTorch.

For example, if you want to use `Adam` with settings like `torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)` in PyTorch,
and considering the `OptimWrapper` type is for default standard single precision training, we can omit the wrapper type here, therefore the config should looks like below.

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

### Use AMP training

If we want to use the automatic mixed precision training, we can simply change the type of `optim_wrapper` to `AmpOptimWrapper` in config files.

```python
optim_wrapper = dict(type='AmpOptimWrapper', optimizer=optimizer)
```

Alternatively, for conveniency, we can set `--amp` parameter to turn on the AMP option directly in the `tools/train.py` script. Refers to [Training and test](../user_guides/train_test.md) tutorial for details of starting a training.

### Parameter-wise finely configuration

Some models may have parameter-specific settings for optimization, for example, no weight decay to the BatchNorm layers or using different learning rates for different network layers.
To finely configure them, we can use the `paramwise_cfg` option in `optim_wrapper`.

- **Set different hyper-parameter multipliers for different types of parameters.**

  For instance, we can set `norm_decay_mult=0.` in `paramwise_cfg` to change the weight decay of weight and bias of normalization layers to zero.

  ```python
  optim_wrapper = dict(
      optimizer=dict(type='SGD', lr=0.8, weight_decay=1e-4),
      paramwise_cfg=dict(norm_decay_mult=0.))
  ```

  More types of parameters are supported to configured, list as follow:

  - `lr_mult`: Multiplier for learning rate of all parameters.
  - `decay_mult`: Multiplier for weight decay of all parameters.
  - `bias_lr_mult`: Multiplier for learning rate of bias (Not include normalization layers' biases and deformable convolution layers' offsets). Defaults to 1.
  - `bias_decay_mult`: Multiplier for weight decay of bias (Not include normalization layers' biases and deformable convolution layers' offsets). Defaults to 1.
  - `norm_decay_mult`: Multiplier for weight decay of weigh and bias of normalization layers. Defaults to 1.
  - `dwconv_decay_mult`: Multiplier for weight decay of depth-wise convolution layers. Defaults to 1.
  - `bypass_duplicate`: Whether to bypass duplicated parameters. Defaults to `False`.
  - `dcn_offset_lr_mult`: Multiplier for learning rate of deformable convolution layers.Defaults to 1.

- **Set different hyper-parameter multipliers for specific parameters.**

  MMClassification can use `custom_keys` to specify different parameters to use different learning rates or weight decay.

  For example, to set all learning rates and weight decays of `backbone.layer0` to 0, the rest of `backbone` remains the same as optimizer and the learning rate of `head` to 0.001, use the configs below.

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

### Gradient clipping

During the training process, the loss function may get close to a cliffy region and cause gradient explosion. And gradient clipping is helpful to stabilize the training process. More introduction can be found in [this page](https://paperswithcode.com/method/gradient-clipping).

Currently we support `clip_grad` option in `optim_wrapper` for gradient clipping, refers to [PyTorch Documentation](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html).

Here is an example:

```python
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, weight_decay=0.0001),
    # norm_type: type of the used p-norm, here norm_type is 2.
    clip_grad=dict(max_norm=35, norm_type=2))
```

### Gradient accumulation

When computing resources are lacking, the batch size can only be set to a small value, which may affect the performance of models. Gradient accumulation can be used to solve this problem. We support `accumulative_counts` option in `optim_wrapper` for gradient accumulation.

Here is an example:

```python
train_dataloader = dict(batch_size=64)
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, weight_decay=0.0001),
    accumulative_counts=4)
```

Indicates that during training, back-propagation is performed every 4 iters. And the above is equivalent to:

```python
train_dataloader = dict(batch_size=256)
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, weight_decay=0.0001))
```

## Customize parameter schedules

In training, the optimzation parameters such as learing rate, momentum, are usually not fixed but changing through iterations or epochs. PyTorch supports several learning rate schedulers, which are not sufficient for complex strategies. In MMClassification, we provide `param_scheduler` for better controls of different parameter schedules.

### Customize learning rate schedules

#### Single learning rate schedule

Learning rate schedulers are widely used to improve performance. We support most of the PyTorch schedulers, including `ExponentialLR`, `LinearLR`, `StepLR`, `MultiStepLR`, etc. We use `MultiStepLR` as the default learning rate schedule for ResNet.

For example:

```python
param_scheduler = dict(
    type='MultiStepLR',
    by_epoch=True,
    milestones=[100, 150],
    gamma=0.1)
```

Other supported learning rate schedules and detailed usages can be found [here](https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/scheduler/lr_scheduler.py), such as `CosineAnnealingLR` schedule:

```python
param_scheduler = dict(
    type='CosineAnnealingLR',
    by_epoch=True,
    T_max=num_epochs)
```

#### Multiple learning rate schedules

However, in some of the training cases, multiple learning rate schedules are applied for higher accuracy. For example ,in the early stage, training is easy to be volatile, and warmup is a technique to reduce volatility. The learning rate will increase gradually from a minor value to the expected value by warmup and decay afterwards by other schedules.

In MMClassification, simply combines desired schedules in `param_scheduler` as a list can achieve the warmup strategy.

Here are some examples:

1. linear & warmup by iter

   ```python
    param_scheduler = [
        # linear warm-up by iters
        dict(type='LinearLR',
            start_factor=0.001,
            by_epoch=False,  # by iters
            begin=0,
            end=50),  # only warm up for first 50 iters
        # main learing rate schedule
        dict(type='MultiStepLR',
            by_epoch=True,
            milestones=[8, 11],
            gamma=0.1)
    ]
   ```

2. exp & warmup by epoch

   ```python
    param_scheduler = [
        # use exponential schedule in [0, 100) epochs
        dict(type='ExponentialLR',
            gamma=0.1,
            by_epoch=True,
            begin=0,
            end=100),
        # use CosineAnnealing schedule in [100, 600) epochs
        dict(type='CosineAnnealingLR',
            T_max=800,
            by_epoch=True,
            begin=100,
            end=600)
    ]
   ```

Notice that, we use `begin` and `end` arguments here to assign the valid range, which is \[`begin`, `end`) for this schedule. And the range unit is defined by `by_epoch` argument. If the ranges for all schedules are not continuous, the learning rate will stay constant in ignored range, otherwise all valid schedulers will be executed in order in a specific stage, which behaves the same as PyTorch [`ChainedScheduler`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ChainedScheduler.html#chainedscheduler).

```{tip}
In case that output learning rates are not as expected, after completing your configuration file，you could use [learning rate visualization tool](https://mmclassification.readthedocs.io/en/latest/tools/visualization.html#learning-rate-schedule-visualization) to draw the corresponding learning rate adjustment curve.
```

### Customize momentum schedules

We support the momentum scheduler to modify the model's momentum according to learning rate, which could make the model converge in a faster way. The usage is the same as learning rate schedule's.

Supported momentum schedules and detailed usages can be found [here](https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/scheduler/momentum_scheduler.py). We just replace the `LR` in scheduler names to `Momentum`. In config file, the needed momentum schedule can be directly appended to the `param_scheduler` list.

Here is an example:

```python
param_scheduler = [
    # the lr scheduler
    dict(type='LinearLR', ...),
    # the momentum scheduler
    dict(type='LinearMomentum',
         start_factor=0.001,
         by_epoch=False,
         begin=0,
         end=1000)
]
```

## Customize self-implemented methods

In academic research and industrial practice, it may be necessary to use optimization methods not implemented by MMClassification, and you can add them through the following methods.

```{note}
This part will modify the MMClassification source code or add code to the MMClassification framework, beginners can skip it.
```

### Customize self-implemented optimizer

#### 1. Define a new optimizer

A customized optimizer could be defined as below.

Assume you want to add an optimizer named `MyOptimizer`, which has arguments `a`, `b`, and `c`.
You need to create a new directory named `mmcls/engine/optimizers`.
And then implement the new optimizer in a file, e.g., in `mmcls/engine/optimizers/my_optimizer.py`:

```python
from mmengine.registry import OPTIMIZERS
from torch.optim import Optimizer


@OPTIMIZERS.register_module()
class MyOptimizer(Optimizer):

    def __init__(self, a, b, c):

```

#### 2. Add the optimizer to registry

To find the above module defined above, this module should be imported into the main namespace at first. There are two ways to achieve it.

- Modify `mmcls/engine/optimizers/__init__.py` to import it into `optimizer` package.

  Create the `mmcls/engine/optimizers` folder and the `mmcls/engine/optimizers/__init__.py` file if they don't exist. The newly defined module should be imported in `mmcls/engine/optimizers/__init__.py` so that the registry will find the new module and add it:

```python
# In mmcls/engine/optimizers/__init__.py
...
from .my_optimizer import MyOptimizer # MyOptimizer maybe other class name

__all__ = [..., 'MyOptimizer']
```

- Use `custom_imports` in the config file to manually import it

```python
custom_imports = dict(imports=['mmcls.core.optimizer.my_optimizer'], allow_failed_imports=False)
```

The module `mmcls.engine.optimizer.my_optimizer` will be imported at the beginning of the program and the class `MyOptimizer` is then automatically registered.
Note that only the package containing the class `MyOptimizer` should be imported. `mmcls.engine.optimizer.my_optimizer.MyOptimizer` **cannot** be imported directly.

#### 3. Specify the optimizer in the config file

Then you can use `MyOptimizer` in `optim_wrapper` field of config files.
In the configs, the optimizers are defined by the field `optim_wrapper` like the following:

```python
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001))
```

To use your own optimizer, the field can be changed to

```python
optim_wrapper = dict(
    optimizer=dict(type='MyOptimizer', a=a_value, b=b_value, c=c_value))
```

### Customize optimizer constructor

Some models may have some parameter-specific settings for optimization, e.g. weight decay for BatchNorm layers.

Although our `DefaultOptimWrapperConstructor` is powerful, it may still not cover your need. If that, you can do those fine-grained parameter tuning through customizing optimizer constructor.

```python
from mmengine.optim import DefaultOptimWrapperConstructor
from mmengine.registry import OPTIM_WRAPPER_CONSTRUCTORS


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class MyOptimWrapperConstructor(DefaultOptimWrapperConstructor):

    def __init__(self, optim_wrapper_cfg, paramwise_cfg=None):
        ...

    def add_params(self, params, module, prefix='' ,lr=None):
        """Add all parameters of module to the params list."""
        ...

```

The default optimizer constructor is implemented [here](https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/optimizer/default_constructor.py), which could also serve as a template for new optimizer constructor.
