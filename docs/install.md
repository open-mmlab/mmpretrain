## Installation

### Requirements

- Python 3.6+
- PyTorch 1.3+
- [MMCV](https://github.com/open-mmlab/mmcv)

The compatible MMClassification and MMCV versions are as below. Please install the correct version of MMCV to avoid installation issues.

| MMClassification version |     MMCV version     |
|:------------------------:|:--------------------:|
| master                   | mmcv>=1.3.8, <=1.5.0 |
| 0.13.0                   | mmcv>=1.3.8, <=1.5.0 |
| 0.12.0                   | mmcv>=1.3.1, <=1.5.0 |
| 0.11.1                   | mmcv>=1.3.1, <=1.5.0 |
| 0.11.0                   | mmcv>=1.3.0          |
| 0.10.0                   | mmcv>=1.3.0          |
| 0.9.0                    | mmcv>=1.1.4          |
| 0.8.0                    | mmcv>=1.1.4          |
| 0.7.0                    | mmcv>=1.1.4          |
| 0.6.0                    | mmcv>=1.1.4          |

Note: Since the `master` branch is under frequent development, the `mmcv`
version dependency may be inaccurate. If you encounter problems when using
the `master` branch, please try to update `mmcv` to the latest version.

### Install MMClassification

a. Create a conda virtual environment and activate it.

```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab
```

b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,

```shell
conda install pytorch torchvision -c pytorch
```

Note: Make sure that your compilation CUDA version and runtime CUDA version match.
You can check the supported CUDA version for precompiled packages on the [PyTorch website](https://pytorch.org/).

`E.g.1` If you have CUDA 10.1 installed under `/usr/local/cuda` and would like to install
PyTorch 1.5, you need to install the prebuilt PyTorch with CUDA 10.1.

```shell
conda install pytorch cudatoolkit=10.1 torchvision -c pytorch
```

`E.g.2` If you have CUDA 9.2 installed under `/usr/local/cuda` and would like to install
PyTorch 1.3.1., you need to install the prebuilt PyTorch with CUDA 9.2.

```shell
conda install pytorch=1.3.1 cudatoolkit=9.2 torchvision=0.4.2 -c pytorch
```

If you build PyTorch from source instead of installing the prebuilt pacakge,
you can use more CUDA versions such as 9.0.

c. Install MMClassification repository.

#### Release version

We recommend you to install MMClassification with [MIM](https://github.com/open-mmlab/mim).

```shell
pip install git+https://github.com/open-mmlab/mim.git
mim install mmcls
```

MIM can automatically install OpenMMLab projects and their requirements,
and it can also help us to train, parameter search and pretrain model download.

Or, you can install MMClassification with pip:

```shell
pip install mmcls
```

#### Develop version

First, clone the MMClassification repository.

```shell
git clone https://github.com/open-mmlab/mmclassification.git
cd mmclassification
```

And then, install build requirements and install MMClassification.

```shell
pip install -e .  # or "python setup.py develop"
```

Note:

1. Following the above instructions, MMClassification is installed on `dev` mode, any local modifications made to the code will take effect without the need to reinstall it (unless you submit some commits and want to update the version number).

2. If you would like to use `opencv-python-headless` instead of `opencv-python`,

you can install it before installing [mmcv](https://github.com/open-mmlab/mmcv).

### Using multiple MMClassification versions

The train and test scripts already modify the `PYTHONPATH` to ensure the script use the MMClassification in the current directory.

To use the default MMClassification installed in the environment rather than that you are working with, you can remove the following line in those scripts

```shell
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
```
