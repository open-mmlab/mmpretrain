# Implementation for DINO

**NOTE**: We only guarantee correctness of the forward pass, not responsible for full reimplementation.

First, ensure you are in the root directory of MMPretrain, then you have two choices
to play with DINO in MMPretrain:

## Slurm

If you are using a cluster managed by Slurm, you can use the following command to
start your job:

```shell
GPUS_PER_NODE=8 GPUS=8 CPUS_PER_TASK=16 bash projects/dino/tools/slurm_train.sh mm_model dino projects/dino/config/dino_vit-base-p16_8xb64-amp-coslr-100e_in1k.py --amp
```

The above command will pre-train the model on a single node with 8 GPUs.

## PyTorch

If you are using a single machine, without any cluster management software, you can use the following command

```shell
NNODES=1 bash projects/dino/tools/dist_train.sh projects/dino/config/dino_vit-base-p16_8xb64-amp-coslr-100e_in1k.py 8
--amp
```
