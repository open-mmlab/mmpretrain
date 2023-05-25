docker run \
    --gpus all \
    --shm-size=8g \
    --rm \
    -it \
    -v /data/FST/221130_person_crops:/mmclassification/data \
    -v /data/mmclassification/:/mmclassification \
    -v /data:/data \
    -w /mmclassification \
    mmclassification