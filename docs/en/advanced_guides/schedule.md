# Customize Training Schedule

In our codebase, [default training schedules](https://github.com/open-mmlab/mmpretrain/blob/main/configs/_base_/schedules) have been provided for common datasets such as CIFAR, ImageNet, etc. If we attempt to experiment on these datasets for higher accuracy or on different new methods and datasets, we might possibly need to modify the strategies.

In this tutorial, we will introduce how to modify configs to construct optimizers, use parameter-wise finely configuration, gradient clipping, gradient accumulation as well as customize learning rate and momentum schedules. Furthermore, introduce a template to customize self-implemented optimizationmethods for the project.

## Customize optimization

We use the `optim_wrapper` field to configure the strategies of optimization, which includes choices of optimizer, choices of automatic mixed precision training, parameter-wise configurations, gradient clipping and accumulation. Details are seen below.

### Use optimizers supported by PyTorch

We support all the optimizers implemented by PyTorch, and to use them, please change the `optimizer` field of config files.

For example, if you want to use [`SGD`](torch.optim.SGD), the modification in config file could be as the following. Notice that optimization related settings should all wrapped inside the `optim_wrapper`.

```python
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.0003, weight_decay=0.0001)
)
```

```{note}
`type` in optimizer is not a constructor but a optimizer name in PyTorch.
Refers to {external+torch:ref}`List of optimizers supported by PyTorch <optim:algorithms>` for more choices.
```

To modify the learning rate of the model, just modify the `lr` in the config of optimizer.
You can also directly set other arguments according to the [API doc](torch.optim) of PyTorch.

For example, if you want to use [`Adam`](torch.optim.Adam) with settings like `torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)` in PyTorch. You could use the config below:

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
The default type of `optim_wrapper` field is [`OptimWrapper`](mmengine.optim.OptimWrapper), therefore, you can
omit the type field usually, like:

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

### Use AMP training

If we want to use the automatic mixed precision training, we can simply change the type of `optim_wrapper` to `AmpOptimWrapper` in config files.

```python
optim_wrapper = dict(type='AmpOptimWrapper', optimizer=...)
```

Alternatively, for conveniency, we can set `--amp` parameter to turn on the AMP option directly in the `tools/train.py` script. Refers to [Training tutorial](../user_guides/train.md) for details of starting a training.

### Parameter-wise finely configuration

Some models may have parameter-specific settings for optimization, for example, no weight decay to the BatchNorm layers or using different learning rates for different network layers.
To finely configure them, we can use the `paramwise_cfg` argument in `optim_wrapper`.

- **Set different hyper-parameter multipliers for different types of parameters.**

  For instance, we can set `norm_decay_mult=0.` in `paramwise_cfg` to change the weight decay of weight and bias of normalization layers to zero.

  ```python
  optim_wrapper = dict(
      optimizer=dict(type='SGD', lr=0.8, weight_decay=1e-4),
      paramwise_cfg=dict(norm_decay_mult=0.))
  ```

  More types of parameters are supported to configured, list as follow:

  - `bias_lr_mult`: Multiplier for learning rate of bias (Not include normalization layers' biases and deformable convolution layers' offsets). Defaults to 1.
  - `bias_decay_mult`: Multiplier for weight decay of bias (Not include normalization layers' biases and deformable convolution layers' offsets). Defaults to 1.
  - `norm_decay_mult`: Multiplier for weight decay of weight and bias of normalization layers. Defaults to 1.
  - `flat_decay_mult`: Multiplier for weight decay of all one-dimensional parameters. Defaults to 1.
  - `dwconv_decay_mult`: Multiplier for weight decay of depth-wise convolution layers. Defaults to 1.
  - `bypass_duplicate`: Whether to bypass duplicated parameters. Defaults to `False`.
  - `dcn_offset_lr_mult`: Multiplier for learning rate of deformable convolution layers. Defaults to 1.

- **Set different hyper-parameter multipliers for specific parameters.**

  MMPretrain can use `custom_keys` in `paramwise_cfg` to specify different parameters to use different learning rates or weight decay.

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

Currently we support `clip_grad` option in `optim_wrapper` for gradient clipping, refers to [PyTorch Documentation](torch.nn.utils.clip_grad_norm_).

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

In training, the optimzation parameters such as learing rate, momentum, are usually not fixed but changing through iterations or epochs. PyTorch supports several learning rate schedulers, which are not sufficient for complex strategies. In MMPretrain, we provide `param_scheduler` for better controls of different parameter schedules.

### Customize learning rate schedules

Learning rate schedulers are widely used to improve performance. We support most of the PyTorch schedulers, including `ExponentialLR`, `LinearLR`, `StepLR`, `MultiStepLR`, etc.

All available learning rate scheduler can be found {external+mmengine:doc}`here <api/optim>`, and the
names of learning rate schedulers end with `LR`.

- **Single learning rate schedule**

  In most cases, we use only one learning rate schedule for simplicity. For instance, [`MultiStepLR`](mmengine.optim.MultiStepLR) is used as the default learning rate schedule for ResNet. Here, `param_scheduler` is a dictionary.

  ```python
  param_scheduler = dict(
      type='MultiStepLR',
      by_epoch=True,
      milestones=[100, 150],
      gamma=0.1)
  ```

  Or, we want to use the [`CosineAnnealingLR`](mmengine.optim.CosineAnnealingLR) scheduler to decay the learning rate:

  ```python
  param_scheduler = dict(
      type='CosineAnnealingLR',
      by_epoch=True,
      T_max=num_epochs)
  ```

- **Multiple learning rate schedules**

  In some of the training cases, multiple learning rate schedules are applied for higher accuracy. For example ,in the early stage, training is easy to be volatile, and warmup is a technique to reduce volatility.
  The learning rate will increase gradually from a minor value to the expected value by warmup and decay afterwards by other schedules.

  In MMPretrain, simply combines desired schedules in `param_scheduler` as a list can achieve the warmup strategy.

  Here are some examples:

  1. linear warmup during the first 50 iters.

  ```python
    param_scheduler = [
        # linear warm-up by iters
        dict(type='LinearLR',
            start_factor=0.001,
            by_epoch=False,  # by iters
            end=50),  # only warm up for first 50 iters
        # main learing rate schedule
        dict(type='MultiStepLR',
            by_epoch=True,
            milestones=[8, 11],
            gamma=0.1)
    ]
  ```

  2. linear warmup and update lr by iter during the first 10 epochs.

  ```python
    param_scheduler = [
        # linear warm-up by epochs in [0, 10) epochs
        dict(type='LinearLR',
            start_factor=0.001,
            by_epoch=True,
            end=10,
            convert_to_iter_based=True,  # Update learning rate by iter.
        ),
        # use CosineAnnealing schedule after 10 epochs
        dict(type='CosineAnnealingLR', by_epoch=True, begin=10)
    ]
  ```

  Notice that, we use `begin` and `end` arguments here to assign the valid range, which is [`begin`, `end`) for this schedule. And the range unit is defined by `by_epoch` argument. If not specified, the `begin` is 0 and the `end` is the max epochs or iterations.

  If the ranges for all schedules are not continuous, the learning rate will stay constant in ignored range, otherwise all valid schedulers will be executed in order in a specific stage, which behaves the same as PyTorch [`ChainedScheduler`](torch.optim.lr_scheduler.ChainedScheduler).

  ```{tip}
  To check that the learning rate curve is as expected, after completing your configuration file，you could use [optimizer parameter visualization tool](../useful_tools/scheduler_visualization.md) to draw the corresponding learning rate adjustment curve.
  ```

### Customize momentum schedules

We support using momentum schedulers to modify the optimizer's momentum according to learning rate, which could make the loss converge in a faster way. The usage is the same as learning rate schedulers.

All available learning rate scheduler can be found {external+mmengine:doc}`here <api/optim>`, and the
names of momentum rate schedulers end with `Momentum`.

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

## Add new optimizers or constructors

```{note}
This part will modify the MMPretrain source code or add code to the MMPretrain framework, beginners can skip it.
```

### Add new optimizers

In academic research and industrial practice, it may be necessary to use optimization methods not implemented by MMPretrain, and you can add them through the following methods.

1. Implement a New Optimizer

   Assume you want to add an optimizer named `MyOptimizer`, which has arguments `a`, `b`, and `c`.
   You need to create a new file under `mmpretrain/engine/optimizers`, and implement the new optimizer in the file, for example, in `mmpretrain/engine/optimizers/my_optimizer.py`:

   ```python
   from torch.optim import Optimizer
   from mmpretrain.registry import OPTIMIZERS


   @OPTIMIZERS.register_module()
   class MyOptimizer(Optimizer):

       def __init__(self, a, b, c):
           ...

       def step(self, closure=None):
           ...
   ```

2. Import the Optimizer

   To find the above module defined above, this module should be imported during the running.

   Import it in the `mmpretrain/engine/optimizers/__init__.py` to add it into the `mmpretrain.engine` package.

   ```python
   # In mmpretrain/engine/optimizers/__init__.py
   ...
   from .my_optimizer import MyOptimizer # MyOptimizer maybe other class name

   __all__ = [..., 'MyOptimizer']
   ```

   During running, we will automatically import the `mmpretrain.engine` package and register the `MyOptimizer` at the same time.

3. Specify the Optimizer in Config

   Then you can use `MyOptimizer` in the `optim_wrapper.optimizer` field of config files.

   ```python
   optim_wrapper = dict(
       optimizer=dict(type='MyOptimizer', a=a_value, b=b_value, c=c_value))
   ```

### Add new optimizer constructors

Some models may have some parameter-specific settings for optimization, like different weight decay rate for all `BatchNorm` layers.

Although we already can use [the `optim_wrapper.paramwise_cfg` field](#parameter-wise-finely-configuration) to
configure various parameter-specific optimizer settings. It may still not cover your need.

Of course, you can modify it. By default, we use the [`DefaultOptimWrapperConstructor`](mmengine.optim.DefaultOptimWrapperConstructor)
class to deal with the construction of optimizer. And during the construction, it fine-grainedly configures the optimizer settings of
different parameters according to the `paramwise_cfg`，which could also serve as a template for new optimizer constructor.

You can overwrite these behaviors by add new optimizer constructors.

```python
# In mmpretrain/engine/optimizers/my_optim_constructor.py
from mmengine.optim import DefaultOptimWrapperConstructor
from mmpretrain.registry import OPTIM_WRAPPER_CONSTRUCTORS


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class MyOptimWrapperConstructor:

    def __init__(self, optim_wrapper_cfg, paramwise_cfg=None):
        ...

    def __call__(self, model):
        ...
```

Here is a specific example of [OptimWrapperConstructor](mmpretrain.engine.optimizers.LearningRateDecayOptimWrapperConstructor).

And then, import it and use it almost like [the optimizer tutorial](#add-new-optimizers).

1. Import it in the `mmpretrain/engine/optimizers/__init__.py` to add it into the `mmpretrain.engine` package.

   ```python
   # In mmpretrain/engine/optimizers/__init__.py
   ...
   from .my_optim_constructor import MyOptimWrapperConstructor

   __all__ = [..., 'MyOptimWrapperConstructor']
   ```

2. Use `MyOptimWrapperConstructor` in the `optim_wrapper.constructor` field of config files.

   ```python
   optim_wrapper = dict(
       constructor=dict(type='MyOptimWrapperConstructor'),
       optimizer=...,
       paramwise_cfg=...,
   )
   ```
