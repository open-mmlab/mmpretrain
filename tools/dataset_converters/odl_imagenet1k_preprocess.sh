#!/usr/bin/env bash

set -x

DOWNLOAD_DIR=$1
DATA_ROOT=$2

# unzip all of data
cat $DOWNLOAD_DIR/ImageNet-1K/raw/*.tar.gz.* | tar -xvz -C $DOWNLOAD_DIR

# move images into data/imagenet
mv $DOWNLOAD_DIR/ImageNet-1K/{train,val,test} $DATA_ROOT

# download the mate ann_files file
wget -P $DATA_ROOT  https://download.openmmlab.com/mmclassification/datasets/imagenet/meta/caffe_ilsvrc12.tar.gz

# unzip mate ann_files file and put it into 'meta' folder
mkdir $DATA_ROOT/meta
tar -xzvf $DATA_ROOT/caffe_ilsvrc12.tar.gz -C $DATA_ROOT/meta

# remove useless data files
rm -R $DOWNLOAD_DIR/ImageNet-1K
