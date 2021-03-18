# Getting Started

This page provides basic tutorials about the usage of MMClassification.

## Prepare datasets

It is recommended to symlink the dataset root to `$MMCLASSIFICATION/data`.
If your folder structure is different, you may need to change the corresponding paths in config files.

```
mmclassification
├── mmcls
├── tools
├── configs
├── docs
├── data
│   ├── imagenet
│   │   ├── meta
│   │   ├── train
│   │   ├── val
│   ├── cifar
│   │   ├── cifar-10-batches-py
│   ├── mnist
│   │   ├── train-images-idx3-ubyte
│   │   ├── train-labels-idx1-ubyte
│   │   ├── t10k-images-idx3-ubyte
│   │   ├── t10k-labels-idx1-ubyte

```

For ImageNet, it has multiple versions, but the most commonly used one is [ILSVRC 2012](http://www.image-net.org/challenges/LSVRC/2012/). It can be accessed with the following steps.

1. Register an account and login to the [download page](http://www.image-net.org/download-images).
2. Find download links for ILSVRC2012 and download the following two files
    - ILSVRC2012_img_train.tar (~138GB)
    - ILSVRC2012_img_val.tar (~6.3GB)
3. Untar the downloaded files
4. Download meta data using this [script](https://github.com/BVLC/caffe/blob/master/data/ilsvrc12/get_ilsvrc_aux.sh)

For MNIST, CIFAR10 and CIFAR100, the datasets will be downloaded and unzipped automatically if they are not found.

For using custom datasets, please refer to [Tutorials 2: Adding New Dataset](tutorials/new_dataset.md).

## Inference with pretrained models

We provide scripts to inference a single image, inference a dataset and test a dataset (e.g., ImageNet).

### Inference a single image

```shell
python demo/image_demo.py ${IMAGE_FILE} ${CONFIG_FILE} ${CHECKPOINT_FILE}
```

### Inference a dataset

- single GPU
- single node multiple GPU
- multiple node

You can use the following commands to infer a dataset.

```shell
# single-gpu inference
python tools/inference.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}]

# multi-gpu inference
./tools/dist_inference.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [--out ${RESULT_FILE}]
```

Optional arguments:

- `RESULT_FILE`: Filename of the output results in pickle format. If not specified, the results will not be saved to a file.

Examples:

Assume that you have already downloaded the checkpoints to the directory `checkpoints/`.
Infer ResNet-50 on ImageNet validation set to get predicted labels and their corresponding predicted scores.

```shell
python tools/inference.py configs/imagenet/resnet50_batch256.py \
    checkpoints/xxx.pth
```

### Test a dataset

- single GPU
- single node multiple GPU
- multiple node

You can use the following commands to test a dataset.

```shell
# single-gpu testing
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}]

# multi-gpu testing
./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [--out ${RESULT_FILE}]
```

Optional arguments:

- `RESULT_FILE`: Filename of the output results in pickle format. If not specified, the results will not be saved to a file.

Examples:

Assume that you have already downloaded the checkpoints to the directory `checkpoints/`.
Test ResNet-50 on ImageNet validation and evaluate the top-1 and top-5.

```shell
python tools/test.py configs/imagenet/resnet50_b32x8.py \
    checkpoints/xxx.pth
```

## Train a model

MMClassification implements distributed training and non-distributed training,
which uses `MMDistributedDataParallel` and `MMDataParallel` respectively.

All outputs (log files and checkpoints) will be saved to the working directory,
which is specified by `work_dir` in the config file.

By default we evaluate the model on the validation set after each epoch, you can change the evaluation interval by adding the interval argument in the training config.

```python
evaluation = dict(interval=12)  # This evaluate the model per 12 epoch.
```

### Train with a single GPU

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

If you want to specify the working directory in the command, you can add an argument `--work_dir ${YOUR_WORK_DIR}`.

### Train with multiple GPUs

```shell
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

Optional arguments are:

- `--no-validate` (**not suggested**): By default, the codebase will perform evaluation at every k (default value is 1) epochs during the training. To disable this behavior, use `--no-validate`.
- `--work-dir ${WORK_DIR}`: Override the working directory specified in the config file.
- `--resume-from ${CHECKPOINT_FILE}`: Resume from a previous checkpoint file.

Difference between `resume-from` and `load-from`:
`resume-from` loads both the model weights and optimizer status, and the epoch is also inherited from the specified checkpoint. It is usually used for resuming the training process that is interrupted accidentally.
`load-from` only loads the model weights and the training epoch starts from 0. It is usually used for finetuning.

### Train with multiple machines

If you run MMClassification on a cluster managed with [slurm](https://slurm.schedmd.com/), you can use the script `slurm_train.sh`. (This script also supports single machine training.)

```shell
[GPUS=${GPUS}] ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${WORK_DIR}
```

You can check [slurm_train.sh](https://github.com/open-mmlab/mmclassification/blob/master/tools/slurm_train.sh) for full arguments and environment variables.

If you have just multiple machines connected with ethernet, you can refer to
PyTorch [launch utility](https://pytorch.org/docs/stable/distributed_deprecated.html#launch-utility).
Usually it is slow if you do not have high speed networking like InfiniBand.

### Launch multiple jobs on a single machine

If you launch multiple jobs on a single machine, e.g., 2 jobs of 4-GPU training on a machine with 8 GPUs,
you need to specify different ports (29500 by default) for each job to avoid communication conflict.

If you use `dist_train.sh` to launch training jobs, you can set the port in commands.

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh ${CONFIG_FILE} 4
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 ./tools/dist_train.sh ${CONFIG_FILE} 4
```

If you use launch training jobs with Slurm, you need to modify the config files (usually the 6th line from the bottom in config files) to set different communication ports.

In `config1.py`,

```python
dist_params = dict(backend='nccl', port=29500)
```

In `config2.py`,

```python
dist_params = dict(backend='nccl', port=29501)
```

Then you can launch two jobs with `config1.py` ang `config2.py`.

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config1.py ${WORK_DIR}
CUDA_VISIBLE_DEVICES=4,5,6,7 GPUS=4 ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config2.py ${WORK_DIR}
```

## Useful tools

We provide lots of useful tools under `tools/` directory.

### Get the FLOPs and params (experimental)

We provide a script adapted from [flops-counter.pytorch](https://github.com/sovrasov/flops-counter.pytorch) to compute the FLOPs and params of a given model.

```shell
python tools/get_flops.py ${CONFIG_FILE} [--shape ${INPUT_SHAPE}]
```

You will get the result like this.

```
==============================
Input shape: (3, 224, 224)
Flops: 4.12 GFLOPs
Params: 25.56 M
==============================
```

**Note**: This tool is still experimental and we do not guarantee that the number is correct. You may well use the result for simple comparisons, but double check it before you adopt it in technical reports or papers.

(1) FLOPs are related to the input shape while parameters are not. The default input shape is (1, 3, 224, 224).
(2) Some operators are not counted into FLOPs like GN and custom operators. Refer to [`mmcv.cnn.get_model_complexity_info()`](https://github.com/open-mmlab/mmcv/blob/master/mmcv/cnn/utils/flops_counter.py) for details.

### Publish a model

Before you upload a model to AWS, you may want to
(1) convert model weights to CPU tensors, (2) delete the optimizer states and
(3) compute the hash of the checkpoint file and append the hash id to the filename.

```shell
python tools/publish_model.py ${INPUT_FILENAME} ${OUTPUT_FILENAME}
```

E.g.,

```shell
python tools/publish_model.py work_dirs/resnet50/latest.pth imagenet_resnet50_20200708.pth
```

The final output filename will be `imagenet_resnet50_20200708-{hash id}.pth`.

## Tutorials

Currently, we provide five tutorials for users to [finetune models](tutorials/finetune.md), [add new dataset](tutorials/new_dataset.md), [design data pipeline](tutorials/data_pipeline.md) and [add new modules](tutorials/new_modules.md).
