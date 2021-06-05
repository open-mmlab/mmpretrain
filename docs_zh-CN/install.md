## 安装

### 安装依赖包

- Python 3.6+
- PyTorch 1.3+
- [MMCV](https://github.com/open-mmlab/mmcv)

MMClassification 和 MMCV 的适配关系如下，请安装正确版本的 MMCV 以避免安装问题

| MMClassification 版本 |  MMCV 版本  |
|:---------------------:|:-----------:|
|        master         | mmcv>=1.3.1, <=1.5.0 |
|        0.12.0         | mmcv>=1.3.1, <=1.5.0 |
|        0.11.1         | mmcv>=1.3.1, <=1.5.0 |
|        0.11.0         | mmcv>=1.3.0 |
|        0.10.0         | mmcv>=1.3.0 |
|         0.9.0         | mmcv>=1.1.4 |
|         0.8.0         | mmcv>=1.1.4 |
|         0.7.0         | mmcv>=1.1.4 |
|         0.6.0         | mmcv>=1.1.4 |

### 安装 MMClassification 步骤

a. 创建 conda 虚拟环境，并激活

```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab
```

b. 按照 [官方指南](https://pytorch.org/) 安装 PyTorch 和 TorchVision，如：

```shell
conda install pytorch torchvision -c pytorch
```

**注**：请确保 CUDA 编译版本和运行版本相匹配
用户可以参照 [PyTorch 官网](https://pytorch.org/) 对预编译包所支持的 CUDA 版本进行核对。

`例 1`：如果用户的 `/usr/local/cuda` 文件夹下已安装 CUDA 10.1 版本，并且想要安装 PyTorch 1.5 版本，
则需要安装 CUDA 10.1 下预编译的 PyTorch。

```shell
conda install pytorch cudatoolkit=10.1 torchvision -c pytorch
```

`例 2`：如果用户的 `/usr/local/cuda` 文件夹下已安装 CUDA 9.2 版本，并且想要安装 PyTorch 1.3.1 版本，
则需要安装 CUDA 9.2 下预编译的 PyTorch。

```shell
conda install pytorch=1.3.1 cudatoolkit=9.2 torchvision=0.4.2 -c pytorch
```

如果 PyTorch 是由源码进行编译安装（而非直接下载预编译好的安装包），则可以使用更多的 CUDA 版本（如 9.0 版本）。

c. 克隆 mmclassification 库

```shell
git clone https://github.com/open-mmlab/mmclassification.git
cd mmclassification
```

d. 安装依赖包和 MMClassification

```shell
pip install -e .  # or "python setup.py develop"
```

提示：

1. 按照以上步骤，MMClassification 是以 `dev` 模式安装的，任何本地的代码修改都可以直接生效，无需重新安装（除非提交了一些 commit，并且希望提升版本号）

2. 如果希望使用 `opencv-python-headless` 而不是 `opencv-python`，可以在安装 [mmcv](https://github.com/open-mmlab/mmcv) 之前提前安装。

### 在多个 MMClassification 版本下进行开发

MMClassification 的训练和测试脚本已经修改了 `PYTHONPATH` 变量，以确保其能够运行当前目录下的 MMClassification。

如果想要运行环境下默认的 MMClassification，用户需要在训练和测试脚本中去除这一行：

```shell
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
```
