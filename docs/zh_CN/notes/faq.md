# 常见问题

我们在这里列出了一些常见问题及其相应的解决方案。如果您发现任何常见问题并有方法
帮助解决，欢迎随时丰富列表。如果这里的内容没有涵盖您的问题，请按照
[提问模板](https://github.com/open-mmlab/mmpretrain/issues/new/choose)
在 GitHub 上提出问题，并补充模板中需要的信息。

## 安装

- MMEngine, MMCV 与 MMPretrain 的兼容问题

  这里我们列举了各版本 MMPretrain 对 MMEngine 和 MMCV 版本的依赖，请选择合适的 MMEngine 和 MMCV 版本来避免安装和使用中的问题。

  | MMPretrain 版本 |   MMEngine 版本   |    MMCV 版本     |
  | :-------------: | :---------------: | :--------------: |
  |  1.2.0 (main)   | mmengine >= 0.8.3 |  mmcv >= 2.0.0   |
  |      1.1.1      | mmengine >= 0.8.3 |  mmcv >= 2.0.0   |
  |      1.0.0      | mmengine >= 0.8.0 |  mmcv >= 2.0.0   |
  |    1.0.0rc8     | mmengine >= 0.7.1 | mmcv >= 2.0.0rc4 |
  |    1.0.0rc7     | mmengine >= 0.5.0 | mmcv >= 2.0.0rc4 |

  ```{note}
  由于 `dev` 分支处于频繁开发中，MMEngine 和 MMCV 版本依赖可能不准确。如果您在使用
  `dev` 分支时遇到问题，请尝试更新 MMEngine 和 MMCV 到最新版。
  ```

- 使用 Albumentations

  如果你希望使用 `albumentations` 相关的功能，我们建议使用 `pip install -r requirements/optional.txt` 或者
  `pip install -U albumentations>=0.3.2 --no-binary qudida,albumentations` 命令进行安装。

  如果你直接使用 `pip install albumentations>=0.3.2` 来安装，它会同时安装 `opencv-python-headless`
  （即使你已经安装了 `opencv-python`）。具体细节可参阅
  [官方文档](https://albumentations.ai/docs/getting_started/installation/#note-on-opencv-dependencies)。

## 通用问题

### 如果我对源码进行了改动，需要重新安装以使改动生效吗？

如果你遵照[最佳实践](../get_started.md#最佳实践)的指引，从源码安装 mmpretrain，那么任何本地修改都不需要重新安装即可生效。

### 如何在多个 MMPretrain 版本下进行开发？

通常来说，我们推荐通过不同虚拟环境来管理多个开发目录下的 MMPretrain。
但如果你希望在不同目录（如 mmpretrain-0.21, mmpretrain-0.23 等）使用同一个环境进行开发，
我们提供的训练和测试 shell 脚本会自动使用当前目录的 mmpretrain，其他 Python 脚本
则可以在命令前添加 `` PYTHONPATH=`pwd`  `` 来使用当前目录的代码。

反过来，如果你希望 shell 脚本使用环境中安装的 MMPretrain，而不是当前目录的，
则可以去掉 shell 脚本中如下一行代码：

```shell
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
```

### `load_from` 和 `init_cfg` 之间的关系是什么？

- `load_from`: 如果`resume=False`，只导入模型权重，主要用于加载训练过的模型；
  如果 `resume=True` ，加载所有的模型权重、优化器状态和其他训练信息，主要用于恢复中断的训练。

- `init_cfg`: 你也可以指定`init=dict(type="Pretrained", checkpoint=xxx)`来加载权重，
  表示在模型权重初始化时加载权重，通常在训练的开始阶段执行。
  主要用于微调预训练模型，你可以在骨干网络的配置中配置它，还可以使用 `prefix` 字段来只加载对应的权重，例如：

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

参见 [微调模型](./finetune_custom_dataset.md) 以了解更多关于模型微调的细节。

### `default_hooks` 和 `custom_hooks` 之间有什么区别？

几乎没有区别。通常，`default_hooks` 字段用于指定几乎所有实验都会使用的钩子，
而 `custom_hooks` 字段指部分实验特有的钩子。

另一个区别是 `default_hooks` 是一个字典，而 `custom_hooks` 是一个列表，请不要混淆。

### 在训练期间，我没有收到训练日志，这是什么原因？

如果你的训练数据集很小，而批处理量却很大，我们默认的日志间隔可能太大，无法记录你的训练日志。

你可以缩减日志间隔，再试一次，比如:

```python
default_hooks = dict(
    ...
    logger=dict(type='LoggerHook', interval=10),
    ...
)
```

### 如何基于其它数据集训练，例如我自己的数据集或者是 COCO 数据集？

我们提供了 [具体示例](./pretrain_custom_dataset.md) 来展示如何在其它数据集上进行训练。
