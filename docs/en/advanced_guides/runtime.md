# Customize Runtime Settings

The runtime configurations include many helpful functionalities, like checkpoint saving, logger configuration,
etc.. In this tutorial, we will introduce how to configure these functionalities.

<!-- TODO: Link to MMEngine docs instead of API reference after the MMEngine English docs is done. -->

## Checkpoint Saving

The checkpoint saving functionality is a default hook during training. And you can configure it in the
`default_hooks.checkpoint` field.

**The default settings**

```python
default_hooks = [
    ...
    checkpoint = dict(type='CheckpointHook', interval=1)
    ...
]
```

Here are some usual arguments, and all available arguments can be found in the [CheckpointHook](mmengine.hooks.CheckpointHook).

- **`interval`** (int): The saving period. If use -1, it will never save checkpoints.
- **`by_epoch`** (bool): Whether the **`interval`** is by epoch or by iteration. Defaults to `True`.
- **`out_dir`** (str): The root directory to save checkpoints. If not specified, the checkpoints will be saved at the work directory. If specified, the checkpoints will saved at the sub-folder of the **`out_dir`**.
- **`max_keep_ckpts`** (int): The maximum checkpoints to keep. In some cases we want only the latest few checkpoints and would like to delete old ones to save the disk space. Defaults to -1, which means unlimited.
- **`save_best`** (str, List\[str\]): If specified, it will save the checkpoint with the best evaluation result.
  Usually, you can simply use `save_best="auto"` to automatically select the evaluation metric. And if you
  want more advanced configuration, please refer to the [CheckpointHook docs](mmengine.hooks.CheckpointHook).

## Load Checkpoint / Resume Training

In config files, you can specify the loading and resuming functionality as below:

```python
# load from which checkpoint
load_from = "Your checkpoint path"

# whether to resume training from the loaded checkpoint
resume = False
```

The `load_from` field can be both local path and http path. And you can resume training from the checkpoint by
speicfy `resume=True`.

```{tip}
You can also enable auto resuming from the latest checkpoint by speicfy `load_from=None` and `resume=True`.
```

If you are training models by our `tools/train.py` script, you can also use `--resume` argument to resume
training without modify config file manually.

```bash
# Automatically resume from the latest checkpoint.
python tools/train.py configs/resnet/resnet50_8xb32_in1k.py --resume

# Resume from the specified checkpoint.
python tools/train.py configs/resnet/resnet50_8xb32_in1k.py --resume checkpoints/resnet.pth
```

## Randomness Configuration

In the `randomness` field, we provide some options to make the experiment as reproducible as possible.

By default, we won't speicfy seed in the config file, and in every experiment, the program will generate a random seed.

**Default settings:**

```python
randomness = dict(seed=None, deterministic=False)
```

To make the experiment more reproducible, you can speicfy a seed and set `deterministic=True`. The influence
of the `deterministic` option can be found [here](https://pytorch.org/docs/stable/notes/randomness.html#cuda-convolution-benchmarking).

## Log Configuration

The log configuration relate to multiple fields.

In the `log_level` field, you can speicfy the global logging level. See {external+python:ref}`Logging Levels<levels>` for a list of levels.

```python
log_level = 'INFO'
```

In the `default_hooks.logger` field, you can specify the logging interval during training and test. And all
available arguments can be found in the [LoggerHook docs](mmengine.hooks.LoggerHook).

```python
default_hooks = [
    ...
    # print log every 100 iterations.
    logger=dict(type='LoggerHook', interval=100),
    ...
]
```

In the `log_processor` field, you can specify the log smooth method. Usually, we use a window with length 10
to smooth the log and output the mean value of these information. If you want specify the smooth method of
some information finely, see the [LogProcessor docs](mmengine.runner.LogProcessor).

In the `visualizer` field, you can specify multiple backends to save the log information, such as TensorBoard
and WandB. More details can be found in the [Visualizer section](#visualizer).

## Custom Hooks

The hook mechanism is widely used in all OpenMMLab libraries. Through hooks, you can plug-in many
functionalities without modify the source code of the runner.

A details introduction of hooks can be found in {external+mmengine:doc}`Hooks <tutorials/hook>`. And we have
already implemented many hooks in MMEngine and MMClassification, such as:

- [EMAHook](mmengine.hooks.EMAHook)
- [SyncBuffersHook](mmengine.hooks.SyncBuffersHook)
- [EmptyCacheHook](mmengine.hooks.EmptyCacheHook)
- [ClassNumCheckHook](mmcls.engine.hooks.ClassNumCheckHook)
- ......

You can directly use these hooks by modifying the `custom_hooks` field. For example, EMA (Exponential Moving
Average) is widely used in the model training, and you can enable it as below:

```python
custom_hooks = [
    dict(type='EMAHook', momentum=4e-5, priority='ABOVE_NORMAL'),
]
```

## Validation Visualization

The validation visualization functionality is a default hook during validation. And you can configure it in the
`default_hooks.visualization` field.

By default, we disabled it, and you can enable it by specify `enable=True`. And more arguments can be found in
the [VisualizationHook docs](mmcls.engine.hooks.VisualizationHook).

```python
default_hooks = dict(
    ...
    visualization=dict(type='VisualizationHook', enable=False),
    ...
)
```

This hook will select some images in the validation dataset, and tag the prediction result on these images
during every validation process. You can use it to watch the varying of model performance on actual images
during training.

In addition, if the images in your validation dataset are small (\<100), you can rescale them before
visualization by specify `rescale_factor=2.` or higher.

## Visualizer

The visualizer is used to record all kinds of information during training and test, include logs, images and
scalars.

**Default settings:**

```python
visualizer = dict(
    type='ClsVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ]
)
```

Usually, the most useful functionality is to save the log and scalars like `loss` to different backends.
For example, to save them to TensorBoard, simply set as below:

```python
visualizer = dict(
    type='ClsVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ]
)
```

Or save them to WandB as below:

```python
visualizer = dict(
    type='ClsVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='WandbVisBackend'),
    ]
)
```

## Environment Configuration

In the `env_cfg` field, you can configure some low-level parameters, like cuDNN, multi-process and distributed
communication.

Please make sure you understand the meaning of these parameters before modifying them.

```python
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,

    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),

    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)
```

## FAQ

1. **What's the relationship between the `load_from` and the `init_cfg`?**

   - `load_from`: If `resume=False`, only imports model weights, which is mainly used to load trained models;
     If `resume=True`, load all of model weights, optimizer state, and other training information, which is
     mainly used to resume training.

   - `init_cfg`: You can also specify `init=dict(type="Pretrained", checkpoint=xxx)` to load checkpoint, it
     means load the weights during model weights initialization. That is, it will be only done at the
     beginning of the training. It's mainly used to fine-tune on a pre-trained model, and you can set it in
     the backbone config to only load backbone weights, for example:

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

     See the [Fine-tune Models](../user_guides/finetune.md) for more details about fine-tuning.

2. **What's the difference between `default_hooks` and `custom_hooks`?**

   Almost no difference. Usually, the `default_hooks` field is used to speicfy the hooks will be used in almost
   all experiments, and the `custom_hooks` field is used in only some experiments.

   Another difference is the `default_hooks` is a dict while the `custom_hooks` is a list, please don't be
   confused.

3. **During training, I got no training log, what's the reason?**

   If your training dataset is small while the batch size is large, our default log interval may be too large to
   record your training log.

   You can shrink the log interval and try again, like:

   ```python
   default_hooks = dict(
       ...
       logger=dict(type='LoggerHook', interval=10),
       ...
   )
   ```
