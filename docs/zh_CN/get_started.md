# 依赖环境

在本节中，我们将演示如何准备 PyTorch 相关的依赖环境。

MMPretrain 适用于 Linux、Windows 和 macOS。它需要 Python 3.7+、CUDA 10.2+ 和 PyTorch 1.8+。

```{note}
如果你对配置 PyTorch 环境已经很熟悉，并且已经完成了配置，可以直接进入[下一节](#安装)。
否则的话，请依照以下步骤完成配置。
```

**第 1 步** 从[官网](https://docs.conda.io/en/latest/miniconda.html)下载并安装 Miniconda。

**第 2 步** 创建一个 conda 虚拟环境并激活它。

```shell
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```

**第 3 步** 按照[官方指南](https://pytorch.org/get-started/locally/)安装 PyTorch。例如：

在 GPU 平台：

```shell
conda install pytorch torchvision -c pytorch
```

```{warning}
以上命令会自动安装最新版的 PyTorch 与对应的 cudatoolkit，请检查它们是否与你的环境匹配。
```

在 CPU 平台：

```shell
conda install pytorch torchvision cpuonly -c pytorch
```

# 安装

我们推荐用户按照我们的最佳实践来安装 MMPretrain。但除此之外，如果你想根据
你的习惯完成安装流程，也可以参见[自定义安装](#自定义安装)一节来获取更多信息。

## 最佳实践

根据具体需求，我们支持两种安装模式：

- [从源码安装（推荐）](#从源码安装)：希望基于 MMPretrain 框架开发自己的预训练任务，需要添加新的功能，比如新的模型或是数据集，或者使用我们提供的各种工具。
- [作为 Python 包安装](#作为-python-包安装)：只是希望调用 MMPretrain 的 API 接口，或者在自己的项目中导入 MMPretrain 中的模块。

### 从源码安装

这种情况下，从源码按如下方式安装 mmpretrain：

```shell
git clone https://github.com/open-mmlab/mmpretrain.git
cd mmpretrain
pip install -U openmim && mim install -e .
```

```{note}
`"-e"` 表示以可编辑形式安装，这样可以在不重新安装的情况下，让本地修改直接生效
```

### 作为 Python 包安装

直接使用 mim 安装即可。

```shell
pip install -U openmim && mim install "mmpretrain>=1.0.0rc7"
```

```{note}
`mim` 是一个轻量级的命令行工具，可以根据 PyTorch 和 CUDA 版本为 OpenMMLab 算法库配置合适的环境。同时它也提供了一些对于深度学习实验很有帮助的功能。
```

## 验证安装

为了验证 MMPretrain 的安装是否正确，我们提供了一些示例代码来执行模型推理。

如果你是**从源码安装**的 mmpretrain，那么直接运行以下命令进行验证：

```shell
python demo/image_demo.py demo/demo.JPEG resnet18_8xb32_in1k --device cpu
```

你可以看到命令行中输出了结果字典，包括 `pred_label`，`pred_score` 和 `pred_class` 三个字段。

如果你是**作为 Python 包安装**，那么可以打开你的 Python 解释器，并粘贴如下代码：

```python
from mmpretrain import get_model, inference_model

model = get_model('resnet18_8xb32_in1k', device='cpu')  # 或者 device='cuda:0'
inference_model(model, 'demo/demo.JPEG')
```

你会看到输出一个字典，包含预测的标签、得分及类别名。

```{note}
以上示例中，`resnet18_8xb32_in1k` 是模型名称。你可以使用 [`mmpretrain.list_models`](mmpretrain.apis.list_models) 接口来
浏览所有的模型，或者在[模型汇总](./modelzoo_statistics.md)页面进行查找。
```

## 自定义安装

### CUDA 版本

安装 PyTorch 时，需要指定 CUDA 版本。如果您不清楚选择哪个，请遵循我们的建议：

- 对于 Ampere 架构的 NVIDIA GPU，例如 GeForce 30 series 以及 NVIDIA A100，CUDA 11 是必需的。
- 对于更早的 NVIDIA GPU，CUDA 11 是向前兼容的，但 CUDA 10.2 能够提供更好的兼容性，也更加轻量。

请确保你的 GPU 驱动版本满足最低的版本需求，参阅[这张表](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions__table-cuda-toolkit-driver-versions)。

```{note}
如果按照我们的最佳实践进行安装，CUDA 运行时库就足够了，因为我们提供相关 CUDA 代码的预编译，你不需要进行本地编译。
但如果你希望从源码进行 MMCV 的编译，或是进行其他 CUDA 算子的开发，那么就必须安装完整的 CUDA 工具链，参见
[NVIDIA 官网](https://developer.nvidia.com/cuda-downloads)，另外还需要确保该 CUDA 工具链的版本与 PyTorch 安装时
的配置相匹配（如用 `conda install` 安装 PyTorch 时指定的 cudatoolkit 版本）。
```

### 在 CPU 环境中安装

MMPretrain 可以仅在 CPU 环境中安装，在 CPU 模式下，你可以完成训练、测试和模型推理等所有操作。

### 在 Google Colab 中安装

参考 [Colab 教程](https://colab.research.google.com/github/mzr1996/mmclassification-tutorial/blob/master/1.x/MMClassification_tools.ipynb) 安装即可。

### 通过 Docker 使用 MMPretrain

MMPretrain 提供 [Dockerfile](https://github.com/open-mmlab/mmpretrain/blob/main/docker/Dockerfile)
用于构建镜像。请确保你的 [Docker 版本](https://docs.docker.com/engine/install/) >=19.03。

```shell
# 构建默认的 PyTorch 1.12.1，CUDA 11.3 版本镜像
# 如果你希望使用其他版本，请修改 Dockerfile
docker build -t mmpretrain docker/
```

用以下命令运行 Docker 镜像：

```shell
docker run --gpus all --shm-size=8g -it -v {DATA_DIR}:/mmpretrain/data mmpretrain
```

## 故障解决

如果你在安装过程中遇到了什么问题，请先查阅[常见问题](./notes/faq.md)。如果没有找到解决方法，可以在 GitHub
上[提出 issue](https://github.com/open-mmlab/mmpretrain/issues/new/choose)。
