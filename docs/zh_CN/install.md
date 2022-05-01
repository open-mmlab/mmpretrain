# 安装

## 安装依赖包

- Python 3.6+
- PyTorch 1.5+
- [MMCV](https://github.com/open-mmlab/mmcv)

MMClassification 和 MMCV 的适配关系如下，请安装正确版本的 MMCV 以避免安装问题

|   MMClassification 版本  |       MMCV 版本       |
|:------------------------:|:---------------------:|
| dev                      | mmcv>=1.4.8, <1.6.0   |
| 0.23.0 (master)          | mmcv>=1.4.2, <1.6.0   |
| 0.22.1                   | mmcv>=1.4.2, <1.6.0   |
| 0.21.0                   | mmcv>=1.4.2, <=1.5.0  |
| 0.20.1                   | mmcv>=1.4.2, <=1.5.0  |
| 0.19.0                   | mmcv>=1.3.16, <=1.5.0 |
| 0.18.0                   | mmcv>=1.3.16, <=1.5.0 |
| 0.17.0                   | mmcv>=1.3.8, <=1.5.0  |
| 0.16.0                   | mmcv>=1.3.8, <=1.5.0  |
| 0.15.0                   | mmcv>=1.3.8, <=1.5.0  |
| 0.15.0                   | mmcv>=1.3.8, <=1.5.0  |
| 0.14.0                   | mmcv>=1.3.8, <=1.5.0  |
| 0.13.0                   | mmcv>=1.3.8, <=1.5.0  |
| 0.12.0                   | mmcv>=1.3.1, <=1.5.0  |
| 0.11.1                   | mmcv>=1.3.1, <=1.5.0  |
| 0.11.0                   | mmcv>=1.3.0           |
| 0.10.0                   | mmcv>=1.3.0           |
| 0.9.0                    | mmcv>=1.1.4           |
| 0.8.0                    | mmcv>=1.1.4           |
| 0.7.0                    | mmcv>=1.1.4           |
| 0.6.0                    | mmcv>=1.1.4           |

```{note}
由于 `dev` 分支处于频繁开发中，`mmcv` 版本依赖可能不准确。如果您在使用
`dev` 分支时遇到问题，请尝试更新 `mmcv` 到最新版。
```

## 安装依赖环境

1. 创建 conda 虚拟环境，并激活

   ```shell
   conda create -n open-mmlab python=3.8 -y
   conda activate open-mmlab
   ```

2. 按照 [官方指南](https://pytorch.org/) 安装 PyTorch 和 TorchVision，如：

   ```shell
   conda install pytorch torchvision -c pytorch
   ```

   ```{note}
   请确保 CUDA 编译版本和运行版本相匹配。
   可以参照 [PyTorch 官网](https://pytorch.org/) 对预编译包所支持的 CUDA 版本进行核对。
   ```

   *例 1*：如果你已经安装了 CUDA 10.1 版本，并且想要安装 PyTorch 1.5.1 版本，
   则需要安装 CUDA 10.1 下预编译的 PyTorch。

   ```shell
   conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=10.1 -c pytorch
   ```

   *例 2*：如果你已经安装了 CUDA 11.3 版本，并且想要安装 PyTorch 1.10.1 版本，
   则需要安装 CUDA 11.3 下预编译的 PyTorch。

   ```shell
   conda install pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch
   ```

   如果 PyTorch 是由源码进行编译安装（而非直接下载预编译好的安装包），则可以使用更多的 CUDA 版本（如 9.0 版本）。

3. 安装 MMCV

   MMCV 是 MMClassification 的基础库。它有两个版本：

   - **mmcv**：精简版，不包含 CUDA 算子但包含其余所有特性和功能，类似 MMCV 1.0 之前的版本。如果你不需要使用 CUDA 算子的话，精简版可以作为一个考虑选项。
   - **mmcv-full**：完整版，包含所有的特性以及丰富的开箱即用的 CUDA 算子。注意完整版本可能需要更长时间来编译。

   如果你希望安装 mmcv-full，你可以根据 [该教程](https://mmcv.readthedocs.io/en/latest/get_started/installation.html) 完成安装或编译。

   一个更好的选择是使用 [MIM](https://github.com/open-mmlab/mim) 来自动选择适合你的 mmcv-full 版本。在使用 MIM 安装 MMClassification 的时候，它就会自动完成 mmcv-full 的安装。

   ```shell
   pip install openmim
   ```

## 安装 MMClassification 库

根据你的需求，我们支持两种安装模式。

- [仅作为 Python 包使用](#仅作为-python-包使用)：该模式下，你只希望在你的项目中调用 MMClassification 的 API，或者导入 MMClassification 中的模块
- [基于 MMClassification 进行开发（推荐）](#基于-mmclassification-进行开发)：该模式下，你希望基于 MMClassification 框架开发你自己的图像分类任务，需要添加新的功能，比如新的模型或是数据集。并且你可以使用我们提供的所有工具。

### 仅作为 Python 包使用

如果你已经安装了 MIM，那么只需要使用 `mim install mmcls` 命令来安装 MMClassification。MIM 将会根据你的环境选择安装合适的 mmcv-full 版本。另外，MIM 还提供了一系列其他功能来协助进行训练、参数搜索及模型筛选等。

或者，你可以直接通过 pip 来安装，使用 `pip install mmcls` 命令。这种情况下，如果你希望使用 mmcv-full，那么需要提前手动安装 mmcv-full。

### 基于 MMClassification 进行开发

在该模式下，任何本地修改都会直接生效，不需要重新安装（除非提交了一些 commit，并且希望提升版本号）。

1. 克隆最新的 MMClassification 仓库

   ```shell
   git clone https://github.com/open-mmlab/mmclassification.git
   cd mmclassification
   ```

2. 【可选】 签出到 `dev` 分支

   ```shell
   git checkout dev
   ```

   *我需要做这一步吗？* `dev` 分支是开发中的分支，包含了一些试验性的功能。如果你需要这些功能，或者准备参与 MMClassification 开发，那么需要做这一步。

2. 安装依赖包和 MMClassification

   使用 MIM，MIM 会自动安装适合你环境的 mmcv-full。

   ```shell
   mim install -e .
   ```

   或者使用 pip，如果你希望使用 mmcv-full，你需要提前手动安装。

   ```shell
   pip install -e .
   ```

## 利用 Docker 镜像安装 MMClassification

MMClassification 提供 [Dockerfile](https://github.com/open-mmlab/mmclassification/blob/master/docker/Dockerfile) ，可以通过以下命令创建 docker 镜像。

```shell
# 创建基于 PyTorch 1.8.1, CUDA 10.2, CUDNN 7 以及最近版本的 MMCV-full 的镜像 。
docker build -f ./docker/Dockerfile --rm -t mmcls:latest .
```

```{important}
请确保已经安装了 [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).
```

运行一个基于上述镜像的容器：

```shell
docker run --gpus all --shm-size=8g -it -v {DATA_DIR}:/workspace/mmclassification/data mmcls:latest /bin/bash
```

## 在多个 MMClassification 版本下进行开发

MMClassification 的训练和测试脚本已经修改了 `PYTHONPATH` 变量，以确保其能够运行当前目录下的 MMClassification。

如果想要运行环境下默认的 MMClassification，用户需要在训练和测试脚本中去除这一行：

```shell
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
```
