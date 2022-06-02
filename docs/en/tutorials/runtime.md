# Tutorial 7: Customize Runtime Settings

In this tutorial, we will introduce some methods about how to customize workflow and hooks when running your own settings for the project.

<!-- TOC -->

- [Customize Workflow](#customize-workflow)
- [Hooks](#hooks)
  - [Default training hooks](#default-training-hooks)
  - [Use other implemented hooks](#use-other-implemented-hooks)
  - [Customize self-implemented hooks](#customize-self-implemented-hooks)
- [FAQ](#faq)

<!-- TOC -->

## Customize Workflow

Workflow is a list of (phase, duration) to specify the running order and duration. The meaning of "duration" depends on the runner's type.

For example, we use epoch-based runner by default, and the "duration" means how many epochs the phase to be executed in a cycle. Usually,
we only want to execute training phase, just use the following config.

```python
workflow = [('train', 1)]
```

Sometimes we may want to check some metrics (e.g. loss, accuracy) about the model on the validate set.
In such case, we can set the workflow as

```python
[('train', 1), ('val', 1)]
```

so that 1 epoch for training and 1 epoch for validation will be run iteratively.

By default, we recommend using **`EvalHook`** to do evaluation after the training epoch, but you can still use `val` workflow as an alternative.

```{note}
1. The parameters of model will not be updated during the val epoch.
2. Keyword `max_epochs` in the config only controls the number of training epochs and will not affect the validation workflow.
3. Workflows `[('train', 1), ('val', 1)]` and `[('train', 1)]` will not change the behavior of `EvalHook` because `EvalHook` is called by `after_train_epoch` and validation workflow only affect hooks that are called through `after_val_epoch`.
   Therefore, the only difference between `[('train', 1), ('val', 1)]` and ``[('train', 1)]`` is that the runner will calculate losses on the validation set after each training epoch.
```

## Hooks

The hook mechanism is widely used in the OpenMMLab open-source algorithm library. Combined with the `Runner`, the entire life cycle of the training process can be managed easily. You can learn more about the hook through [related article](https://www.calltutors.com/blog/what-is-hook/).

Hooks only work after being registered into the runner. At present, hooks are mainly divided into two categories:

- default training hooks

The default training hooks are registered by the runner by default. Generally, they are hooks for some basic functions, and have a certain priority, you don't need to modify the priority.

- custom hooks

The custom hooks are registered through `custom_hooks`. Generally, they are hooks with enhanced functions. The priority needs to be specified in the configuration file. If you do not specify the priority of the hook, it will be set to 'NORMAL' by default.

**Priority list**

|      Level      | Value |
| :-------------: | :---: |
|     HIGHEST     |   0   |
|    VERY_HIGH    |  10   |
|      HIGH       |  30   |
|  ABOVE_NORMAL   |  40   |
| NORMAL(default) |  50   |
|  BELOW_NORMAL   |  60   |
|       LOW       |  70   |
|    VERY_LOW     |  90   |
|     LOWEST      |  100  |

The priority determines the execution order of the hooks. Before training, the log will print out the execution order of the hooks at each stage to facilitate debugging.

### default training hooks

Some common hooks are not registered through `custom_hooks`, they are

|         Hooks         |     Priority      |
| :-------------------: | :---------------: |
|    `LrUpdaterHook`    |  VERY_HIGH (10)   |
| `MomentumUpdaterHook` |     HIGH (30)     |
|    `OptimizerHook`    | ABOVE_NORMAL (40) |
|   `CheckpointHook`    |    NORMAL (50)    |
|    `IterTimerHook`    |     LOW (70)      |
|      `EvalHook`       |     LOW (70)      |
|    `LoggerHook(s)`    |   VERY_LOW (90)   |

`OptimizerHook`, `MomentumUpdaterHook` and `LrUpdaterHook` have been introduced in [sehedule strategy](./schedule.md).
`IterTimerHook` is used to record elapsed time and does not support modification.

Here we reveal how to customize `CheckpointHook`, `LoggerHooks`, and `EvalHook`.

#### CheckpointHook

The MMCV runner will use `checkpoint_config` to initialize [`CheckpointHook`](https://github.com/open-mmlab/mmcv/blob/9ecd6b0d5ff9d2172c49a182eaa669e9f27bb8e7/mmcv/runner/hooks/checkpoint.py).

```python
checkpoint_config = dict(interval=1)
```

We could set `max_keep_ckpts` to save only a small number of checkpoints or decide whether to store state dict of optimizer by `save_optimizer`.
More details of the arguments are [here](https://mmcv.readthedocs.io/en/latest/api.html#mmcv.runner.CheckpointHook)

#### LoggerHooks

The `log_config` wraps multiple logger hooks and enables to set intervals. Now MMCV supports `TextLoggerHook`, `WandbLoggerHook`, `MlflowLoggerHook`, `NeptuneLoggerHook`, `DvcliveLoggerHook` and `TensorboardLoggerHook`.
The detailed usages can be found in the [doc](https://mmcv.readthedocs.io/en/latest/api.html#mmcv.runner.LoggerHook).

```python
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
```

#### EvalHook

The config of `evaluation` will be used to initialize the [`EvalHook`](https://github.com/open-mmlab/mmclassification/blob/master/mmcls/core/evaluation/eval_hooks.py).

The `EvalHook` has some reserved keys, such as `interval`, `save_best` and `start`, and the other arguments such as `metrics` will be passed to the `dataset.evaluate()`

```python
evaluation = dict(interval=1, metric='accuracy', metric_options={'topk': (1, )})
```

You can save the model weight when the best verification result is obtained by modifying the parameter `save_best`:

```python
# "auto" means automatically select the metrics to compare.
# You can also use a specific key like "accuracy_top-1".
evaluation = dict(interval=1, save_best="auto", metric='accuracy', metric_options={'topk': (1, )})
```

When running some large experiments, you can skip the validation step at the beginning of training by modifying the parameter `start` as below:

```python
evaluation = dict(interval=1, start=200, metric='accuracy', metric_options={'topk': (1, )})
```

This indicates that, before the 200th epoch, evaluations would not be executed. Since the 200th epoch, evaluations would be executed after the training process.

```{note}
In the default configuration files of MMClassification, the evaluation field is generally placed in the datasets configs.
```

### Use other implemented hooks

Some hooks have been already implemented in MMCV and MMClassification, they are:

- [EMAHook](https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/ema.py)
- [SyncBuffersHook](https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/sync_buffer.py)
- [EmptyCacheHook](https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/memory.py)
- [ProfilerHook](https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/profiler.py)
- ......

If the hook is already implemented in MMCV, you can directly modify the config to use the hook as below

```python
mmcv_hooks = [
    dict(type='MMCVHook', a=a_value, b=b_value, priority='NORMAL')
]
```

such as using `EMAHook`, interval is 100 iters:

```python
custom_hooks = [
    dict(type='EMAHook', interval=100, priority='HIGH')
]
```

## Customize self-implemented hooks

### 1. Implement a new hook

Here we give an example of creating a new hook in MMClassification and using it in training.

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

Depending on the functionality of the hook, the users need to specify what the hook will do at each stage of the training in `before_run`, `after_run`, `before_epoch`, `after_epoch`, `before_iter`, and `after_iter`.

### 2. Register the new hook

Then we need to make `MyHook` imported. Assuming the file is in `mmcls/core/utils/my_hook.py` there are two ways to do that:

- Modify `mmcls/core/utils/__init__.py` to import it.

  The newly defined module should be imported in `mmcls/core/utils/__init__.py` so that the registry will
  find the new module and add it:

```python
from .my_hook import MyHook
```

- Use `custom_imports` in the config to manually import it

```python
custom_imports = dict(imports=['mmcls.core.utils.my_hook'], allow_failed_imports=False)
```

### 3. Modify the config

```python
custom_hooks = [
    dict(type='MyHook', a=a_value, b=b_value)
]
```

You can also set the priority of the hook as below:

```python
custom_hooks = [
    dict(type='MyHook', a=a_value, b=b_value, priority='ABOVE_NORMAL')
]
```

By default, the hook's priority is set as `NORMAL` during registration.

## FAQ

### 1. `resume_from` and `load_from` and `init_cfg.Pretrained`

- `load_from` : only imports model weights, which is mainly used to load pre-trained or trained models;

- `resume_from` : not only import model weights, but also optimizer information, current epoch information, mainly used to continue training from the checkpoint.

- `init_cfg.Pretrained` : Load weights during weight initialization, and you can specify which module to load. This is usually used when fine-tuning a model, refer to [Tutorial 2: Fine-tune Models](./finetune.md).
