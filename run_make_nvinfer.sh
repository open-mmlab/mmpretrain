#!/bin/bash

MODEL_DIR=$1
MODEL_DIR=${MODEL_DIR%%/}
shift

CLASSES=( "$@" )
CLASSES=$(IFS=';' ; echo "${CLASSES[*]}")

echo "[property]

onnx-file=$MODEL_DIR/end2end.onnx
model-engine-file=$MODEL_DIR/end2end.onnx_b1_gpu0_fp16.engine

gie-unique-id=3
net-scale-factor=0.01742919389
offsets=123.675;116.128;103.53
scaling-filter=1 # 0=Nearest, 1=Bilinear

network-mode=2 # 0=FP32, 1=INT8, 2=FP16
batch-size=1

infer-dims=3;128;128
maintain-aspect-ratio=0
model-color-format=0

batch-size=8
network-type=100 # >3 disables post-processing
cluster-mode=4 # 1=DBSCAN 4=No Clustering
process-mode=2 # 1=Primary, 2=Secondary

output-tensor-meta=1

operate-on-class-ids=1;
output-blob-names=output

[custom]
labels=$CLASSES
report_labels=$CLASSES
" > "$MODEL_DIR/nvinfer-state-classifier-config.txt"


