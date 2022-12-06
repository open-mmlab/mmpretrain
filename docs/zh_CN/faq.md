# 常见问题

我们在这里列出了一些常见问题及其相应的解决方案。如果您发现任何常见问题并有方法
帮助解决，欢迎随时丰富列表。如果这里的内容没有涵盖您的问题，请按照
[提问模板](https://github.com/open-mmlab/mmclassification/issues/new/choose)
在 GitHub 上提出问题，并补充模板中需要的信息。

## 安装

- MMCV 与 MMClassification 的兼容问题。如遇到
  "AssertionError: MMCV==xxx is used but incompatible. Please install mmcv>=xxx, \<=xxx."

  这里我们列举了各版本 MMClassification 对 MMCV 版本的依赖，请选择合适的 MMCV
  版本来避免安装和使用中的问题。

  | MMClassification version |      MMCV version      |
  | :----------------------: | :--------------------: |
  |           dev            |  mmcv>=1.7.0, \<1.9.0  |
  |     0.25.0 (master)      |  mmcv>=1.4.2, \<1.9.0  |
  |          0.24.1          |  mmcv>=1.4.2, \<1.9.0  |
  |          0.23.2          |  mmcv>=1.4.2, \<1.7.0  |
  |          0.22.1          |  mmcv>=1.4.2, \<1.6.0  |
  |          0.21.0          | mmcv>=1.4.2, \<=1.5.0  |
  |          0.20.1          | mmcv>=1.4.2, \<=1.5.0  |
  |          0.19.0          | mmcv>=1.3.16, \<=1.5.0 |
  |          0.18.0          | mmcv>=1.3.16, \<=1.5.0 |
  |          0.17.0          | mmcv>=1.3.8, \<=1.5.0  |
  |          0.16.0          | mmcv>=1.3.8, \<=1.5.0  |
  |          0.15.0          | mmcv>=1.3.8, \<=1.5.0  |
  |          0.15.0          | mmcv>=1.3.8, \<=1.5.0  |
  |          0.14.0          | mmcv>=1.3.8, \<=1.5.0  |
  |          0.13.0          | mmcv>=1.3.8, \<=1.5.0  |
  |          0.12.0          | mmcv>=1.3.1, \<=1.5.0  |
  |          0.11.1          | mmcv>=1.3.1, \<=1.5.0  |
  |          0.11.0          |      mmcv>=1.3.0       |
  |          0.10.0          |      mmcv>=1.3.0       |
  |          0.9.0           |      mmcv>=1.1.4       |
  |          0.8.0           |      mmcv>=1.1.4       |
  |          0.7.0           |      mmcv>=1.1.4       |
  |          0.6.0           |      mmcv>=1.1.4       |

  ```{note}
  由于 `dev` 分支处于频繁开发中，MMCV 版本依赖可能不准确。如果您在使用
  `dev` 分支时遇到问题，请尝试更新 MMCV 到最新版。
  ```

- 使用 Albumentations

  如果你希望使用 `albumentations` 相关的功能，我们建议使用 `pip install -r requirements/optional.txt` 或者
  `pip install -U albumentations>=0.3.2 --no-binary qudida,albumentations` 命令进行安装。

  如果你直接使用 `pip install albumentations>=0.3.2` 来安装，它会同时安装 `opencv-python-headless`
  （即使你已经安装了 `opencv-python`）。具体细节可参阅
  [官方文档](https://albumentations.ai/docs/getting_started/installation/#note-on-opencv-dependencies)。

## 开发

- 如果我对源码进行了改动，需要重新安装以使改动生效吗？

  如果你遵照[最佳实践](install.md)的指引，从源码安装 mmcls，那么任何本地修改都不需要重新安装即可生效。

- 如何在多个 MMClassification 版本下进行开发？

  通常来说，我们推荐通过不同虚拟环境来管理多个开发目录下的 MMClassification。
  但如果你希望在不同目录（如 mmcls-0.21, mmcls-0.23 等）使用同一个环境进行开发，
  我们提供的训练和测试 shell 脚本会自动使用当前目录的 mmcls，其他 Python 脚本
  则可以在命令前添加 `` PYTHONPATH=`pwd`  `` 来使用当前目录的代码。

  反过来，如果你希望 shell 脚本使用环境中安装的 MMClassification，而不是当前目录的，
  则可以去掉 shell 脚本中如下一行代码：

  ```shell
  PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
  ```
