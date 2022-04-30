# Installation

## Requirements

- Python 3.6+
- PyTorch 1.5+
- [MMCV](https://github.com/open-mmlab/mmcv)

The compatible MMClassification and MMCV versions are as below. Please install the correct version of MMCV to avoid installation issues.

| MMClassification version |     MMCV version      |
|:------------------------:|:---------------------:|
| dev                      | mmcv>=1.5.0, <1.6.0   |
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
Since the `dev` branch is under frequent development, the `mmcv`
version dependency may be inaccurate. If you encounter problems when using
the `dev` branch, please try to update `mmcv` to the latest version.
```

## Install Dependencies

1. Create a conda virtual environment and activate it.

   ```shell
   conda create -n open-mmlab python=3.8 -y
   conda activate open-mmlab
   ```

2. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,

   ```shell
   conda install pytorch torchvision -c pytorch
   ```

   ```{note}
   Make sure that your compilation CUDA version and runtime CUDA version match.
   You can check the supported CUDA version for precompiled packages on the
   [PyTorch website](https://pytorch.org/).
   ```

   *E.g.1* If you have CUDA 10.1 installed under `/usr/local/cuda` and would like to install
   PyTorch 1.5.1, you need to install the prebuilt PyTorch with CUDA 10.1.

   ```shell
   conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=10.1 -c pytorch
   ```

   *E.g.2* If you have CUDA 11.3 installed under `/usr/local/cuda` and would like to install
   PyTorch 1.10.1, you need to install the prebuilt PyTorch with CUDA 11.3.

   ```shell
   conda install pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch
   ```

   If you build PyTorch from source instead of installing the prebuilt package,
   you can use more CUDA versions such as 9.0.

3. Install MMCV

   MMCV is a foundational library for MMClassification. And there are two versions of MMCV.

   - **mmcv**: lite, without CUDA ops but all other features, similar to mmcv<1.0.0. It is useful when you do not need those CUDA ops.
   - **mmcv-full**: comprehensive, with full features and various CUDA ops out of box. It takes longer time to build.

   If you want to install mmcv-full, you can install/compile it according to the [instructions](https://mmcv.readthedocs.io/en/latest/get_started/installation.html).

   A better choice is to use [MIM](https://github.com/open-mmlab/mim) to automatically select the mmcv-full version. MIM will automatically install mmcv-full when you use it to install MMClassification in the next section.

   ```shell
   pip install openmim
   ```

## Install MMClassification repository

According to your needs, we support two install modes.

- [Use as a Python package](#use-as-a-python-package): In this mode, you just want to call MMClassification's APIs or import MMClassification's modules in your project.
- [Develop based on MMClassification (Recommended)](#develop-based-on-mmclassification): In this mode, you want to develop your own image classification task or new features based on MMClassification framework. For example, you want to add new dataset or new models. And you can use all tools we provided.

### Use as a Python package

If you have installed MIM, simply use `mim install mmcls` to install
MMClassification. MIM will automatically install the mmcv-full which fits your
environment. In addition, MIM also has some other functions to help to do
training, parameter searching and model filtering, etc.

Or, you can use pip to install MMClassification with `pip install mmcls`. In
this situation, if you want to use mmcv-full, please install it manually in
advance.

### Develop based on MMClassification

In this mode, any local modifications made to the code will take effect without
the need to reinstall it (unless you submit some commits and want to update the
version number).

1. Clone the MMClassification repository.

   ```shell
   git clone https://github.com/open-mmlab/mmclassification.git
   cd mmclassification
   ```

2. [Optional] Checkout to the `dev` branch.

   ```shell
   git checkout dev
   ```

   *Do I need to do this?* The `dev` branch is in development and includes some experimental functions. If you want these functions or want to contribute to MMClassification, do it.

3. Install requirements and MMClassification.

   Use MIM, and MIM will automatically install the mmcv-full which fits your environment.
   ```shell
   mim install -e .
   ```

   Or use pip, and if you want to use mmcv-full, you need to install it manually in advance.

   ```shell
   pip install -e .
   ```

## Another option: Docker Image

We provide a [Dockerfile](https://github.com/open-mmlab/mmclassification/blob/master/docker/Dockerfile) to build an image.

```shell
# build an image with PyTorch 1.8.1, CUDA 10.2, CUDNN 7 and MMCV-full latest version released.
docker build -f ./docker/Dockerfile --rm -t mmcls:latest .
```

```{important}
Make sure you've installed the [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).
```

Run a container built from mmcls image with command:

```shell
docker run --gpus all --shm-size=8g -it -v {DATA_DIR}:/workspace/mmclassification/data mmcls:latest /bin/bash
```

## Using multiple MMClassification versions

The train and test scripts already modify the `PYTHONPATH` to ensure the script use the MMClassification in the current directory.

To use the default MMClassification installed in the environment rather than that you are working with, you can remove the following line in those scripts

```shell
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
```
