#!/usr/bin/env bash

set -x

DOWNLOAD_DIR=$1
DATA_ROOT=$2

# unzip all of data
cat $DOWNLOAD_DIR/CUB-200-2011/raw/*.tar.gz | tar -xvz -C $DOWNLOAD_DIR

# move data into DATA_ROOT
mv -f $DOWNLOAD_DIR/CUB-200-2011/CUB-200-2011/* $DATA_ROOT/

# remove useless data file
rm -R $DOWNLOAD_DIR/CUB-200-2011/
