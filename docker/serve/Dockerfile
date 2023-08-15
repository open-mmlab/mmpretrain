ARG PYTORCH="1.12.1"
ARG CUDA="11.3"
ARG CUDNN="8"
FROM pytorch/torchserve:latest-gpu

ARG MMPRE="1.0.2"

ENV PYTHONUNBUFFERED TRUE

ENV HOME="/home/model-server"
ENV PATH="/opt/conda/bin:$HOME/.local/bin:$PATH"
RUN export FORCE_CUDA=1

# TORCHSEVER
RUN pip install torchserve torch-model-archiver
RUN pip install nvgpu

# OPEN-MMLAB
ARG PYTORCH
ARG CUDA
RUN pip install openmim
RUN mim install mmpretrain==${MMPRE}
RUN mkdir -p $HOME/tmp

COPY --chown=model-server entrypoint.sh $HOME/.local/bin/entrypoint.sh

RUN chmod +x $HOME/.local/bin/entrypoint.sh

COPY --chown=model-server config.properties $HOME/config.properties

EXPOSE 8080 8081 8082

USER model-server
WORKDIR $HOME
ENV TEMP=$HOME/tmp
ENTRYPOINT ["/home/model-server/.local/bin/entrypoint.sh"]
CMD ["serve"]
