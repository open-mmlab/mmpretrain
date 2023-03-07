#!/usr/bin/env bash

set -x

CFG=$1
CHECKPOINT=$2
GPUS=$3
PY_ARGS=${@:4}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
mim test mmseg \
    $CFG \
    --checkpoint $CHECKPOINT \
    --launcher pytorch \
    -G $GPUS \
    $PY_ARGS
