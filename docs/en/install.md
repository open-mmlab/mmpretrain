# Prerequisites

In this section we demonstrate how to prepare an environment with PyTorch.

MMClassification works on Linux, Windows and macOS. It requires Python 3.6+, CUDA 9.2+ and PyTorch 1.5+.

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
This command will automatically install the latest version PyTorch and cudatoolkit, please check whether they matches your environment.
```

On CPU platforms:

```shell
conda install pytorch torchvision cpuonly -c pytorch
```

# Installation

We recommend that users follow our best practices to install MMClassification. However, the whole process is highly customizable. See [Customize Installation](#customize-installation) section for more information.

## Best Practices

**Step 0.** Install [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim).

```shell
pip install -U openmim
mim install mmcv-full
```

**Step 1.** Install MMClassification.

According to your needs, we support two install modes:

- [Install from source (Recommended)](#install-from-source): You want to develop your own image classification task or new features based on MMClassification framework. For example, you want to add new dataset or new models. And you can use all tools we provided.
- [Install as a Python package](#install-as-a-python-package): You just want to call MMClassification's APIs or import MMClassification's modules in your project.

### Install from source

In this case, install mmcls from source:

```shell
git clone https://github.com/open-mmlab/mmclassification.git
cd mmclassification
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```

Optionally, if you want to contribute to MMClassification or experience experimental functions, please checkout to the dev branch:

```shell
git checkout dev
```

### Install as a Python package

Just install with pip.

```shell
pip install mmcls
```

## Verify the installation

To verify whether MMClassification is installed correctly, we provide some sample codes to run an inference demo.

**Step 1.** We need to download config and checkpoint files.

```shell
mim download mmcls --config resnet50_8xb32_in1k --dest .
```

**Step 2.** Verify the inference demo.

Option (a). If you install mmcls from source, just run the following command:

```shell
python demo/image_demo.py demo/demo.JPEG resnet50_8xb32_in1k.py resnet50_8xb32_in1k_20210831-ea4938fc.pth --device cpu
```

You will see the output result dict including `pred_label`, `pred_score` and `pred_class` in your terminal.
And if you have graphical interface (instead of remote terminal etc.), you can enable `--show` option to show
the demo image with these predictions in a window.

Option (b). If you install mmcls as a python package, open you python interpreter and copy&paste the following codes.

```python
from mmcls.apis import init_model, inference_model

config_file = 'resnet50_8xb32_in1k.py'
checkpoint_file = 'resnet50_8xb32_in1k_20210831-ea4938fc.pth'
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

### Install MMCV without MIM

MMCV contains C++ and CUDA extensions, thus depending on PyTorch in a complex
way. MIM solves such dependencies automatically and makes the installation
easier. However, it is not a must.

To install MMCV with pip instead of MIM, please follow
[MMCV installation guides](https://mmcv.readthedocs.io/en/latest/get_started/installation.html).
This requires manually specifying a find-url based on PyTorch version and its CUDA version.

For example, the following command install mmcv-full built for PyTorch 1.10.x and CUDA 11.3.

```shell
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html
```

### Install on CPU-only platforms

MMClassification can be built for CPU only environment. In CPU mode you can
train (requires MMCV version >= 1.4.4), test or inference a model.

Some functionalities are gone in this mode, usually GPU-compiled ops. But don't
worry, almost all models in MMClassification don't depends on these ops.

### Install on Google Colab

[Google Colab](https://research.google.com/) usually has PyTorch installed,
thus we only need to install MMCV and MMClassification with the following
commands.

**Step 1.** Install [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim).

```shell
!pip3 install openmim
!mim install mmcv-full
```

**Step 2.** Install MMClassification from the source.

```shell
!git clone https://github.com/open-mmlab/mmclassification.git
%cd mmclassification
!pip install -e .
```

**Step 3.** Verification.

```python
import mmcls
print(mmcls.__version__)
# Example output: 0.23.0 or newer
```

```{note}
Within Jupyter, the exclamation mark `!` is used to call external executables and `%cd` is a [magic command](https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-cd) to change the current working directory of Python.
```

### Using MMClassification with Docker

We provide a [Dockerfile](https://github.com/open-mmlab/mmclassification/blob/master/docker/Dockerfile)
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

If you have some issues during the installation, please first view the [FAQ](faq.md) page.
You may [open an issue](https://github.com/open-mmlab/mmclassification/issues/new/choose)
on GitHub if no solution is found.
