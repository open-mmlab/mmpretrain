FROM openmmlab/mmdeploy:ubuntu20.04-cuda11.8-mmdeploy1.3.0

RUN pip install -U openmim && mim install git+https://github.com/logivations/mmpretrain.git

RUN git clone https://github.com/logivations/mmpretrain.git /code/mmpretrain