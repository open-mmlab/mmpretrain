# 安装

## 安装依赖包

- Python 3.6+
- PyTorch 1.5+
- [MMCV](https://github.com/open-mmlab/mmcv)

MMClassification 和 MMCV 的适配关系如下，请安装正确版本的 MMCV 以避免安装问题

| MMClassification 版本 |       MMCV 版本       |
|:---------------------:|:---------------------:|
|        master         | mmcv>=1.3.16, <=1.5.0 |
|        0.18.0         | mmcv>=1.3.16, <=1.5.0 |
|        0.17.0         | mmcv>=1.3.8, <=1.5.0  |
|        0.16.0         | mmcv>=1.3.8, <=1.5.0  |
|        0.15.0         | mmcv>=1.3.8, <=1.5.0  |
|        0.14.0         | mmcv>=1.3.8, <=1.5.0  |
|        0.13.0         | mmcv>=1.3.8, <=1.5.0  |
|        0.12.0         | mmcv>=1.3.1, <=1.5.0  |
|        0.11.1         | mmcv>=1.3.1, <=1.5.0  |
|        0.11.0         | mmcv>=1.3.0           |
|        0.10.0         | mmcv>=1.3.0           |
|         0.9.0         | mmcv>=1.1.4           |
|         0.8.0         | mmcv>=1.1.4           |
|         0.7.0         | mmcv>=1.1.4           |
|         0.6.0         | mmcv>=1.1.4           |

```{note}
由于 `master` 分支处于频繁开发中，`mmcv` 版本依赖可能不准确。如果您在使用
`master` 分支时遇到问题，请尝试更新 `mmcv` 到最新版。
```

## 安装 MMClassification 步骤

a. 创建 conda 虚拟环境，并激活

```shell
conda create -n open-mmlab python=3.8 -y
conda activate open-mmlab
```

b. 按照 [官方指南](https://pytorch.org/) 安装 PyTorch 和 TorchVision，如：

```shell
conda install pytorch torchvision -c pytorch
```

```{note}
请确保 CUDA 编译版本和运行版本相匹配。
可以参照 [PyTorch 官网](https://pytorch.org/) 对预编译包所支持的 CUDA 版本进行核对。
```

`例 1`：如果用户的 `/usr/local/cuda` 文件夹下已安装 CUDA 10.1 版本，并且想要安装 PyTorch 1.5.1 版本，
则需要安装 CUDA 10.1 下预编译的 PyTorch。

```shell
conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=10.1 -c pytorch
```

`例 2`：如果用户的 `/usr/local/cuda` 文件夹下已安装 CUDA 11.3 版本，并且想要安装 PyTorch 1.10.0 版本，
则需要安装 CUDA 11.3 下预编译的 PyTorch。

```shell
conda install pytorch==1.10.0 torchvision==0.11.1 cudatoolkit=11.3 -c pytorch
```

如果 PyTorch 是由源码进行编译安装（而非直接下载预编译好的安装包），则可以使用更多的 CUDA 版本（如 9.0 版本）。

c. 安装 MMClassification 库

### 稳定版本

我们推荐使用 [MIM](https://github.com/open-mmlab/mim) 进行 MMClassification 的安装。

```shell
pip install git+https://github.com/open-mmlab/mim.git
mim install mmcls
```

MIM 工具可以自动安装 OpenMMLab 旗下的各个项目及其依赖，同时可以协助进行训练、调参和预训练模型下载等。

或者，可以直接通过 pip 进行 MMClassification 的安装：

```shell
pip install mmcls
```

### 开发版本

首先，克隆最新的 MMClassification 仓库：

```shell
git clone https://github.com/open-mmlab/mmclassification.git
cd mmclassification
```

之后，安装依赖包和 MMClassification：

```shell
pip install -e .  # 或者 "python setup.py develop"
```

```{note}
按照以上步骤，MMClassification 是以 `dev` 模式安装的，任何本地的代码修改都可以直接生效，无需重新安装（除非提交了一些 commit，并且希望提升版本号）
```

### 利用 Docker 镜像安装 MMClassification

MMClassification 提供 [Dockerfile](https://github.com/open-mmlab/mmclassification/blob/master/docker/Dockerfile) ，可以通过以下命令创建 docker 镜像。

```shell
# 创建基于 PyTorch 1.6.0, CUDA 10.1, CUDNN 7 的镜像。
docker build -f ./docker/Dockerfile --rm -t mmcls:torch1.6.0-cuda10.1-cudnn7 .
```

```{important}
请确保已经安装了 [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).
```

运行一个基于上述镜像的容器：

```shell
docker run --gpus all --shm-size=8g -it -v {DATA_DIR}:/workspace/mmclassification/data mmcls:torch1.6.0-cuda10.1-cudnn7 /bin/bash
```

## 在多个 MMClassification 版本下进行开发

MMClassification 的训练和测试脚本已经修改了 `PYTHONPATH` 变量，以确保其能够运行当前目录下的 MMClassification。

如果想要运行环境下默认的 MMClassification，用户需要在训练和测试脚本中去除这一行：

```shell
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
```
