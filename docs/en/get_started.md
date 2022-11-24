# Prerequisites

In this section we demonstrate how to prepare an environment with PyTorch.

MMClassification works on Linux, Windows and macOS. It requires Python 3.6+, CUDA 9.2+ and PyTorch 1.6+.

```{note}
If you are experienced with PyTorch and have already installed it, just skip this part and jump to the [next section](#installation). Otherwise, you can follow these steps for the preparation.
```

**Step 1.** Download and install Miniconda from the [official website](https://docs.conda.io/en/latest/miniconda.html).

**Step 2.** Create a conda environment and activate it.

```shell
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```

**Step 3.** Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/), e.g.

On GPU platforms:

```shell
conda install pytorch torchvision -c pytorch
```

```{warning}
This command will automatically install the latest version PyTorch and cudatoolkit, please check whether they match your environment.
```

On CPU platforms:

```shell
conda install pytorch torchvision cpuonly -c pytorch
```

# Installation

We recommend that users follow our best practices to install MMClassification. However, the whole process is highly customizable. See [Customize Installation](#customize-installation) section for more information.

## Best Practices

**Step 1.** Install [MIM](https://github.com/open-mmlab/mim).

> *mim is a light-weight command-line tool to setup appropriate environment for OpenMMLab repositories according to PyTorch and CUDA version. It also has some useful functions for deep-learning experiments.*

```shell
pip install -U openmim
```

**Step 2.** Install MMClassification.

According to your needs, we support two install modes:

- [Install from source (Recommended)](#install-from-source): You want to develop your own image classification task or new features based on MMClassification framework. For example, adding new dataset or new models. And you can use all tools we provided.
- [Install as a Python package](#install-as-a-python-package): You just want to call MMClassification's APIs or import MMClassification's modules in your project.

### Install from source

In this case, install mmcls from source:

```shell
git clone -b 1.x https://github.com/open-mmlab/mmclassification.git
cd mmclassification
mim install -e .
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```

Optionally, if you want to contribute to MMClassification or experience experimental functions, please checkout to the `dev-1.x` branch:

```shell
git checkout dev-1.x
```

### Install as a Python package

Just install with mim.

```shell
mim install "mmcls>=1.0.0rc0"
```

## Verify the installation

To verify whether MMClassification is installed correctly, we provide some sample codes to run an inference demo.

**Step 1.** We need to download config and checkpoint files.

```shell
mim download mmcls --config resnet50_8xb32_in1k --dest .
```

**Step 2.** Verify the inference demo.

Option (a). If you install mmcls from the source, just run the following command:

```shell
python demo/image_demo.py demo/demo.JPEG resnet50_8xb32_in1k.py resnet50_8xb32_in1k_20210831-ea4938fc.pth --device cpu
```

You will see the output result dict including `pred_label`, `pred_score` and `pred_class` in your terminal.

Option (b). If you install mmcls as a python package, open your python interpreter and copy&paste the following codes.

```python
from mmcls.apis import init_model, inference_model
from mmcls.utils import register_all_modules

config_file = 'resnet50_8xb32_in1k.py'
checkpoint_file = 'resnet50_8xb32_in1k_20210831-ea4938fc.pth'
register_all_modules()  # register all modules and set mmcls as the default scope.
model = init_model(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'
inference_model(model, 'demo/demo.JPEG')
```

You will see a dict printed, including the predicted label, score and category name.

## Customize Installation

### CUDA versions

When installing PyTorch, you need to specify the version of CUDA. If you are
not clear on which to choose, follow our recommendations:

- For Ampere-based NVIDIA GPUs, such as GeForce 30 series and NVIDIA A100, CUDA 11 is a must.
- For older NVIDIA GPUs, CUDA 11 is backward compatible, but CUDA 10.2 offers better compatibility and is more lightweight.

Please make sure the GPU driver satisfies the minimum version requirements. See [this table](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions__table-cuda-toolkit-driver-versions) for more information.

```{note}
Installing CUDA runtime libraries is enough if you follow our best practices,
because no CUDA code will be compiled locally. However if you hope to compile
MMCV from source or develop other CUDA operators, you need to install the
complete CUDA toolkit from NVIDIA's [website](https://developer.nvidia.com/cuda-downloads),
and its version should match the CUDA version of PyTorch. i.e., the specified
version of cudatoolkit in `conda install` command.
```

### Install on CPU-only platforms

MMClassification can be built for CPU only environment. In CPU mode you can train, test or inference a model.

### Install on Google Colab

See [the Colab tutorial](https://colab.research.google.com/github/mzr1996/mmclassification-tutorial/blob/master/1.x/MMClassification_tools.ipynb).

### Using MMClassification with Docker

We provide a [Dockerfile](https://github.com/open-mmlab/mmclassification/blob/1.x/docker/Dockerfile)
to build an image. Ensure that your [docker version](https://docs.docker.com/engine/install/) >=19.03.

```shell
# build an image with PyTorch 1.8.1, CUDA 10.2
# If you prefer other versions, just modified the Dockerfile
docker build -t mmclassification docker/
```

Run it with

```shell
docker run --gpus all --shm-size=8g -it -v {DATA_DIR}:/mmclassification/data mmclassification
```

## Trouble shooting

If you have some issues during the installation, please first view the [FAQ](./notes/faq.md) page.
You may [open an issue](https://github.com/open-mmlab/mmclassification/issues/new/choose)
on GitHub if no solution is found.
