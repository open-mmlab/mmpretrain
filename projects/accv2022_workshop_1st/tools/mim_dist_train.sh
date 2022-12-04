#!/usr/bin/env bash

set -x

CFG=$1
PRETRAIN=$2
GPUS=${GPUS:-8}
PY_ARGS=${@:3}


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
mim train mmcls $CFG \
    --launcher pytorch -G $GPUS \
    --cfg-options model.backbone.init_cfg.type=Pretrained \
    model.backbone.init_cfg.checkpoint=$PRETRAIN \
    model.backbone.init_cfg.prefix="backbone." \
    $PY_ARGS
