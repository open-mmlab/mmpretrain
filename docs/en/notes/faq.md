# Frequently Asked Questions

We list some common troubles faced by many users and their corresponding
solutions here. Feel free to enrich the list if you find any frequent issues
and have ways to help others to solve them. If the contents here do not cover
your issue, please create an issue using the
[provided templates](https://github.com/open-mmlab/mmpretrain/issues/new/choose)
and make sure you fill in all required information in the template.

## Installation

- Compatibility issue between MMEngine, MMCV and MMPretrain

  Compatible MMPretrain and MMEngine, MMCV versions are shown as below. Please
  choose the correct version of MMEngine and MMCV to avoid installation issues.

  | MMPretrain version | MMEngine version  |   MMCV version   |
  | :----------------: | :---------------: | :--------------: |
  |    1.1.1 (main)    | mmengine >= 0.8.3 |  mmcv >= 2.0.0   |
  |       1.0.0        | mmengine >= 0.8.0 |  mmcv >= 2.0.0   |
  |      1.0.0rc8      | mmengine >= 0.7.1 | mmcv >= 2.0.0rc4 |
  |      1.0.0rc7      | mmengine >= 0.5.0 | mmcv >= 2.0.0rc4 |

  ```{note}
  Since the `dev` branch is under frequent development, the MMEngine and MMCV
  version dependency may be inaccurate. If you encounter problems when using
  the `dev` branch, please try to update MMEngine and MMCV to the latest version.
  ```

- Using Albumentations

  If you would like to use `albumentations`, we suggest using `pip install -r requirements/albu.txt` or
  `pip install -U albumentations --no-binary qudida,albumentations`.

  If you simply use `pip install albumentations>=0.3.2`, it will install `opencv-python-headless` simultaneously
  (even though you have already installed `opencv-python`). Please refer to the
  [official documentation](https://albumentations.ai/docs/getting_started/installation/#note-on-opencv-dependencies)
  for details.

## General Questions

### Do I need to reinstall mmpretrain after some code modifications?

If you follow [the best practice](../get_started.md#best-practices) and install mmpretrain from source,
any local modifications made to the code will take effect without
reinstallation.

### How to develop with multiple MMPretrain versions?

Generally speaking, we recommend to use different virtual environments to
manage MMPretrain in different working directories. However, you
can also use the same environment to develop MMPretrain in different
folders, like mmpretrain-0.21, mmpretrain-0.23. When you run the train or test shell script,
it will adopt the mmpretrain package in the current folder. And when you run other Python
script, you can also add `` PYTHONPATH=`pwd`  `` at the beginning of your command
to use the package in the current folder.

Conversely, to use the default MMPretrain installed in the environment
rather than the one you are working with, you can remove the following line
in those shell scripts:

```shell
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
```

### What's the relationship between the `load_from` and the `init_cfg`?

- `load_from`: If `resume=False`, only imports model weights, which is mainly used to load trained models;
  If `resume=True`, load all of the model weights, optimizer state, and other training information, which is
  mainly used to resume interrupted training.

- `init_cfg`: You can also specify `init=dict(type="Pretrained", checkpoint=xxx)` to load checkpoint, it
  means load the weights during model weights initialization. That is, it will be only done at the
  beginning of the training. It's mainly used to fine-tune a pre-trained model, and you can set it in
  the backbone config and use `prefix` field to only load backbone weights, for example:

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

See the [Fine-tune Models](./finetune_custom_dataset.md) for more details about fine-tuning.

### What's the difference between `default_hooks` and `custom_hooks`?

Almost no difference. Usually, the `default_hooks` field is used to specify the hooks that will be used in almost
all experiments, and the `custom_hooks` field is used in only some experiments.

Another difference is the `default_hooks` is a dict while the `custom_hooks` is a list, please don't be
confused.

### During training, I got no training log, what's the reason?

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

### How to train with other datasets, like my own dataset or COCO?

We provide [specific examples](./pretrain_custom_dataset.md) to show how to train with other datasets.
