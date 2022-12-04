FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-devel

# Avoiding user interaction with libopencv-dev
ENV DEBIAN_FRONTEND=noninteractive

# fetch the key refer to https://forums.developer.nvidia.com/t/18-04-cuda-docker-image-is-broken/212892/9
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub 32
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="(dirname(which conda))/../"
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MMCV
RUN pip install openmim
RUN mim install mmengine "mmcv>=2.0rc0"

# Install MMClassification
RUN conda clean --all
RUN git clone https://github.com/open-mmlab/mmclassification.git
WORKDIR ./mmclassification
RUN git checkout dev-1.x
RUN pip install --no-cache-dir -e .

# Install MMSelfSup
RUN git clone https://github.com/YuanLiuuuuuu/mmselfsup.git
WORKDIR ./mmselfsup
RUN git checkout accv
RUN pip install --no-cache-dir -e .
