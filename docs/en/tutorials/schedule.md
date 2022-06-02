# Tutorial 6: Customize Schedule

In this tutorial, we will introduce some methods about how to construct optimizers, customize learning rate and momentum schedules, parameter-wise finely configuration, gradient clipping, gradient accumulation, and customize self-implemented methods for the project.

<!-- TOC -->

- [Customize optimizer supported by PyTorch](#customize-optimizer-supported-by-pytorch)
- [Customize learning rate schedules](#customize-learning-rate-schedules)
  - [Learning rate decay](#learning-rate-decay)
  - [Warmup strategy](#warmup-strategy)
- [Customize momentum schedules](#customize-momentum-schedules)
- [Parameter-wise finely configuration](#parameter-wise-finely-configuration)
- [Gradient clipping and gradient accumulation](#gradient-clipping-and-gradient-accumulation)
  - [Gradient clipping](#gradient-clipping)
  - [Gradient accumulation](#gradient-accumulation)
- [Customize self-implemented methods](#customize-self-implemented-methods)
  - [Customize self-implemented optimizer](#customize-self-implemented-optimizer)
  - [Customize optimizer constructor](#customize-optimizer-constructor)

<!-- TOC -->

## Customize optimizer supported by PyTorch

We already support to use all the optimizers implemented by PyTorch, and to use and modify them, please change the `optimizer` field of config files.

For example, if you want to use `SGD`, the modification could be as the following.

```python
optimizer = dict(type='SGD', lr=0.0003, weight_decay=0.0001)
```

To modify the learning rate of the model, just modify the `lr` in the config of optimizer.
You can also directly set other arguments according to the [API doc](https://pytorch.org/docs/stable/optim.html?highlight=optim#module-torch.optim) of PyTorch.

For example, if you want to use `Adam` with the setting like `torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)` in PyTorch,
the config should looks like.

```python
optimizer = dict(type='Adam', lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
```

## Customize learning rate schedules

### Learning rate decay

Learning rate decay is widely used to improve performance. And to use learning rate decay, please set the `lr_confg` field in config files.

For example, we use step policy as the default learning rate decay policy of ResNet, and the config is:

```python
lr_config = dict(policy='step', step=[100, 150])
```

Then during training, the program will call [`StepLRHook`](https://github.com/open-mmlab/mmcv/blob/f48241a65aebfe07db122e9db320c31b685dc674/mmcv/runner/hooks/lr_updater.py#L153) periodically to update the learning rate.

We also support many other learning rate schedules [here](https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py), such as `CosineAnnealing` and `Poly` schedule. Here are some examples

- ConsineAnnealing schedule:

  ```python
  lr_config = dict(
      policy='CosineAnnealing',
      warmup='linear',
      warmup_iters=1000,
      warmup_ratio=1.0 / 10,
      min_lr_ratio=1e-5)
  ```

- Poly schedule:

  ```python
  lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
  ```

### Warmup strategy

In the early stage, training is easy to be volatile, and warmup is a technique
to reduce volatility. With warmup, the learning rate will increase gradually
from a minor value to the expected value.

In MMClassification, we use `lr_config` to configure the warmup strategy, the main parameters are as follows：

- `warmup`: The warmup curve type. Please choose one from 'constant', 'linear', 'exp' and `None`, and `None` means disable warmup.
- `warmup_by_epoch` : if warmup by epoch or not, default to be True, if set to be False, warmup by iter.
- `warmup_iters` : the number of warm-up iterations, when `warmup_by_epoch=True`, the unit is epoch; when `warmup_by_epoch=False`, the unit is the number of iterations (iter).
- `warmup_ratio` : warm-up initial learning rate will calculate as `lr = lr * warmup_ratio`。

Here are some examples

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

```{tip}
After completing your configuration file，you could use [learning rate visualization tool](https://mmclassification.readthedocs.io/en/latest/tools/visualization.html#learning-rate-schedule-visualization) to draw the corresponding learning rate adjustment curve.
```

## Customize momentum schedules

We support the momentum scheduler to modify the model's momentum according to learning rate, which could make the model converge in a faster way.

Momentum scheduler is usually used with LR scheduler, for example, the following config is used to accelerate convergence.
For more details, please refer to the implementation of [CyclicLrUpdater](https://github.com/open-mmlab/mmcv/blob/f48241a65aebfe07db122e9db320c31b685dc674/mmcv/runner/hooks/lr_updater.py#L327)
and [CyclicMomentumUpdater](https://github.com/open-mmlab/mmcv/blob/f48241a65aebfe07db122e9db320c31b685dc674/mmcv/runner/hooks/momentum_updater.py#L130).

Here is an example

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

## Parameter-wise finely configuration

Some models may have some parameter-specific settings for optimization, for example, no weight decay to the BatchNorm layer or using different learning rates for different network layers.
To finely configuration them, we can use the `paramwise_cfg` option in `optimizer`.

We provide some examples here and more usages refer to [DefaultOptimizerConstructor](https://mmcv.readthedocs.io/en/latest/_modules/mmcv/runner/optimizer/default_constructor.html#DefaultOptimizerConstructor).

- Using specified options

  The `DefaultOptimizerConstructor` provides options including `bias_lr_mult`, `bias_decay_mult`, `norm_decay_mult`, `dwconv_decay_mult`, `dcn_offset_lr_mult` and `bypass_duplicate` to configure special optimizer behaviors of bias, normalization, depth-wise convolution, deformable convolution and duplicated parameter. E.g:

  1. No weight decay to the BatchNorm layer

  ```python
  optimizer = dict(
      type='SGD',
      lr=0.8,
      weight_decay=1e-4,
      paramwise_cfg=dict(norm_decay_mult=0.))
  ```

- Using `custom_keys` dict

  MMClassification can use `custom_keys` to specify different parameters to use different learning rates or weight decays, for example:

  1. No weight decay for specific parameters

  ```python
  paramwise_cfg = dict(
      custom_keys={
          'backbone.cls_token': dict(decay_mult=0.0),
          'backbone.pos_embed': dict(decay_mult=0.0)
      })

  optimizer = dict(
      type='SGD',
      lr=0.8,
      weight_decay=1e-4,
      paramwise_cfg=paramwise_cfg)
  ```

  2. Using a smaller learning rate and a weight decay for the backbone layers

  ```python
  optimizer = dict(
      type='SGD',
      lr=0.8,
      weight_decay=1e-4,
      # 'lr' for backbone and 'weight_decay' are 0.1 * lr and 0.9 * weight_decay
      paramwise_cfg=dict(
          custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=0.9)}))
  ```

## Gradient clipping and gradient accumulation

Besides the basic function of PyTorch optimizers, we also provide some enhancement functions, such as gradient clipping, gradient accumulation, etc., refer to [MMCV](https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/optimizer.py).

### Gradient clipping

During the training process, the loss function may get close to a cliffy region and cause gradient explosion. And gradient clipping is helpful to stabilize the training process. More introduction can be found in [this page](https://paperswithcode.com/method/gradient-clipping).

Currently we support `grad_clip` option in `optimizer_config`, and the arguments refer to [PyTorch Documentation](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html).

Here is an example:

```python
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# norm_type: type of the used p-norm, here norm_type is 2.
```

When inheriting from base and modifying configs, if `grad_clip=None` in base, `_delete_=True` is needed. For more details about `_delete_` you can refer to [TUTORIAL 1: LEARN ABOUT CONFIGS](https://mmclassification.readthedocs.io/en/latest/tutorials/config.html#ignore-some-fields-in-the-base-configs). For example,

```python
_base_ = [./_base_/schedules/imagenet_bs256_coslr.py]

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2), _delete_=True, type='OptimizerHook')
# you can ignore type if type is 'OptimizerHook', otherwise you must add "type='xxxxxOptimizerHook'" here
```

### Gradient accumulation

When computing resources are lacking, the batch size can only be set to a small value, which may affect the performance of models. Gradient accumulation can be used to solve this problem.

Here is an example:

```python
data = dict(samples_per_gpu=64)
optimizer_config = dict(type="GradientCumulativeOptimizerHook", cumulative_iters=4)
```

Indicates that during training, back-propagation is performed every 4 iters. And the above is equivalent to:

```python
data = dict(samples_per_gpu=256)
optimizer_config = dict(type="OptimizerHook")
```

```{note}
When the optimizer hook type is not specified in `optimizer_config`, `OptimizerHook` is used by default.
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
You need to create a new directory named `mmcls/core/optimizer`.
And then implement the new optimizer in a file, e.g., in `mmcls/core/optimizer/my_optimizer.py`:

```python
from mmcv.runner import OPTIMIZERS
from torch.optim import Optimizer


@OPTIMIZERS.register_module()
class MyOptimizer(Optimizer):

    def __init__(self, a, b, c):

```

#### 2. Add the optimizer to registry

To find the above module defined above, this module should be imported into the main namespace at first. There are two ways to achieve it.

- Modify `mmcls/core/optimizer/__init__.py` to import it into `optimizer` package, and then modify `mmcls/core/__init__.py` to import the new `optimizer` package.

  Create the `mmcls/core/optimizer` folder and the `mmcls/core/optimizer/__init__.py` file if they don't exist. The newly defined module should be imported in `mmcls/core/optimizer/__init__.py` and `mmcls/core/__init__.py` so that the registry will find the new module and add it:

```python
# In mmcls/core/optimizer/__init__.py
from .my_optimizer import MyOptimizer # MyOptimizer maybe other class name

__all__ = ['MyOptimizer']
```

```python
# In mmcls/core/__init__.py
...
from .optimizer import *  # noqa: F401, F403
```

- Use `custom_imports` in the config to manually import it

```python
custom_imports = dict(imports=['mmcls.core.optimizer.my_optimizer'], allow_failed_imports=False)
```

The module `mmcls.core.optimizer.my_optimizer` will be imported at the beginning of the program and the class `MyOptimizer` is then automatically registered.
Note that only the package containing the class `MyOptimizer` should be imported. `mmcls.core.optimizer.my_optimizer.MyOptimizer` **cannot** be imported directly.

#### 3. Specify the optimizer in the config file

Then you can use `MyOptimizer` in `optimizer` field of config files.
In the configs, the optimizers are defined by the field `optimizer` like the following:

```python
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
```

To use your own optimizer, the field can be changed to

```python
optimizer = dict(type='MyOptimizer', a=a_value, b=b_value, c=c_value)
```

### Customize optimizer constructor

Some models may have some parameter-specific settings for optimization, e.g. weight decay for BatchNorm layers.

Although our `DefaultOptimizerConstructor` is powerful, it may still not cover your need. If that, you can do those fine-grained parameter tuning through customizing optimizer constructor.

```python
from mmcv.runner.optimizer import OPTIMIZER_BUILDERS


@OPTIMIZER_BUILDERS.register_module()
class MyOptimizerConstructor:

    def __init__(self, optimizer_cfg, paramwise_cfg=None):
        pass

    def __call__(self, model):
        ...      # Construct your optimzier here.
        return my_optimizer
```

The default optimizer constructor is implemented [here](https://github.com/open-mmlab/mmcv/blob/9ecd6b0d5ff9d2172c49a182eaa669e9f27bb8e7/mmcv/runner/optimizer/default_constructor.py#L11), which could also serve as a template for new optimizer constructor.
