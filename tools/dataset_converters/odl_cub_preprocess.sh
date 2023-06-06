#!/usr/bin/env bash

set -x

DOWNLOAD_DIR=$1
DATA_ROOT=$2

cat $DOWNLOAD_DIR/CUB-200-2011/raw/*.tar.gz | tar -xvz -C $(dirname $DATA_ROOT)
mv -f $DATA_ROOT/CUB-200-2011/* $DATA_ROOT/
rm -R $DATA_ROOT/CUB-200-2011/ $DATA_ROOT/raw/
