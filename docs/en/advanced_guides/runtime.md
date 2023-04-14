# Customize Runtime Settings

The runtime configurations include many helpful functionalities, like checkpoint saving, logger configuration,
etc. In this tutorial, we will introduce how to configure these functionalities.

## Save Checkpoint

The checkpoint saving functionality is a default hook during training. And you can configure it in the
`default_hooks.checkpoint` field.

```{note}
The hook mechanism is widely used in all OpenMMLab libraries. Through hooks, you can plug in many
functionalities without modifying the main execution logic of the runner.

A detailed introduction of hooks can be found in {external+mmengine:doc}`Hooks <tutorials/hook>`.
```

**The default settings**

```python
default_hooks = dict(
    ...
    checkpoint = dict(type='CheckpointHook', interval=1)
    ...
)
```

Here are some usual arguments, and all available arguments can be found in the [CheckpointHook](mmengine.hooks.CheckpointHook).

- **`interval`** (int): The saving period. If use -1, it will never save checkpoints.
- **`by_epoch`** (bool): Whether the **`interval`** is by epoch or by iteration. Defaults to `True`.
- **`out_dir`** (str): The root directory to save checkpoints. If not specified, the checkpoints will be saved in the work directory. If specified, the checkpoints will be saved in the sub-folder of the **`out_dir`**.
- **`max_keep_ckpts`** (int): The maximum checkpoints to keep. In some cases, we want only the latest few checkpoints and would like to delete old ones to save disk space. Defaults to -1, which means unlimited.
- **`save_best`** (str, List[str]): If specified, it will save the checkpoint with the best evaluation result.
  Usually, you can simply use `save_best="auto"` to automatically select the evaluation metric.

And if you want more advanced configuration, please refer to the [CheckpointHook docs](tutorials/hook.md#checkpointhook).

## Load Checkpoint / Resume Training

In config files, you can specify the loading and resuming functionality as below:

```python
# load from which checkpoint
load_from = "Your checkpoint path"

# whether to resume training from the loaded checkpoint
resume = False
```

The `load_from` field can be either a local path or an HTTP path. And you can resume training from the checkpoint by
specify `resume=True`.

```{tip}
You can also enable auto resuming from the latest checkpoint by specifying `load_from=None` and `resume=True`.
Runner will find the latest checkpoint from the work directory automatically.
```

If you are training models by our `tools/train.py` script, you can also use `--resume` argument to resume
training without modifying the config file manually.

```bash
# Automatically resume from the latest checkpoint.
python tools/train.py configs/resnet/resnet50_8xb32_in1k.py --resume

# Resume from the specified checkpoint.
python tools/train.py configs/resnet/resnet50_8xb32_in1k.py --resume checkpoints/resnet.pth
```

## Randomness Configuration

In the `randomness` field, we provide some options to make the experiment as reproducible as possible.

By default, we won't specify seed in the config file, and in every experiment, the program will generate a random seed.

**Default settings:**

```python
randomness = dict(seed=None, deterministic=False)
```

To make the experiment more reproducible, you can specify a seed and set `deterministic=True`. The influence
of the `deterministic` option can be found [here](https://pytorch.org/docs/stable/notes/randomness.html#cuda-convolution-benchmarking).

## Log Configuration

The log configuration relates to multiple fields.

In the `log_level` field, you can specify the global logging level. See {external+python:ref}`Logging Levels<levels>` for a list of levels.

```python
log_level = 'INFO'
```

In the `default_hooks.logger` field, you can specify the logging interval during training and testing. And all
available arguments can be found in the [LoggerHook docs](tutorials/hook.md#loggerhook).

```python
default_hooks = dict(
    ...
    # print log every 100 iterations.
    logger=dict(type='LoggerHook', interval=100),
    ...
)
```

In the `log_processor` field, you can specify the log smooth method. Usually, we use a window with length of 10
to smooth the log and output the mean value of all information. If you want to specify the smooth method of
some information finely, see the {external+mmengine:doc}`LogProcessor docs <advanced_tutorials/logging>`.

```python
# The default setting, which will smooth the values in training log by a 10-length window.
log_processor = dict(window_size=10)
```

In the `visualizer` field, you can specify multiple backends to save the log information, such as TensorBoard
and WandB. More details can be found in the [Visualizer section](#visualizer).

## Custom Hooks

Many above functionalities are implemented by hooks, and you can also plug-in other custom hooks by modifying
`custom_hooks` field. Here are some hooks in MMEngine and MMPretrain that you can use directly, such as:

- [EMAHook](mmpretrain.engine.hooks.EMAHook)
- [SyncBuffersHook](mmengine.hooks.SyncBuffersHook)
- [EmptyCacheHook](mmengine.hooks.EmptyCacheHook)
- [ClassNumCheckHook](mmpretrain.engine.hooks.ClassNumCheckHook)
- ......

For example, EMA (Exponential Moving Average) is widely used in the model training, and you can enable it as
below:

```python
custom_hooks = [
    dict(type='EMAHook', momentum=4e-5, priority='ABOVE_NORMAL'),
]
```

## Visualize Validation

The validation visualization functionality is a default hook during validation. And you can configure it in the
`default_hooks.visualization` field.

By default, we disabled it, and you can enable it by specifying `enable=True`. And more arguments can be found in
the [VisualizationHook docs](mmpretrain.engine.hooks.VisualizationHook).

```python
default_hooks = dict(
    ...
    visualization=dict(type='VisualizationHook', enable=False),
    ...
)
```

This hook will select some images in the validation dataset, and tag the prediction results on these images
during every validation process. You can use it to watch the varying of model performance on actual images
during training.

In addition, if the images in your validation dataset are small (\<100), you can rescale them before
visualization by specifying `rescale_factor=2.` or higher.

## Visualizer

The visualizer is used to record all kinds of information during training and test, including logs, images and
scalars. By default, the recorded information will be saved at the `vis_data` folder under the work directory.

**Default settings:**

```python
visualizer = dict(
    type='UniversalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ]
)
```

Usually, the most useful function is to save the log and scalars like `loss` to different backends.
For example, to save them to TensorBoard, simply set them as below:

```python
visualizer = dict(
    type='UniversalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ]
)
```

Or save them to WandB as below:

```python
visualizer = dict(
    type='UniversalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='WandbVisBackend'),
    ]
)
```

## Environment Configuration

In the `env_cfg` field, you can configure some low-level parameters, like cuDNN, multi-process, and distributed
communication.

**Please make sure you understand the meaning of these parameters before modifying them.**

```python
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,

    # set multi-process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),

    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)
```
