docker run \
    --gpus all \
    --shm-size=8g \
    --rm \
    -it \
    -v /data:/data \
    -w /data/mmpretrain \
    mmpretrain