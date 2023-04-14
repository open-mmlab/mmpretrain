# Solution of FGIA ACCV 2022(1st Place)

This is fine-tuning part of the 1st Place Solution for Webly-supervised Fine-grained Recognition, refer to the ACCV workshop competition in https://www.cvmart.net/race/10412/base.

## Result

<details>

<summary>Show the result</summary>

<br>

**Leaderboard A**

![LB-A](https://user-images.githubusercontent.com/18586273/205498131-5728e470-b4f6-43b7-82a5-5f8e3bd5168e.png)

**Leaderboard B**

![LB-B](https://user-images.githubusercontent.com/18586273/205498171-5a3a3055-370a-4a8b-9779-b686254ebc94.png)

</br>

</details>

## Reproduce

For detailed self-supervised pretrain code, please refer to [Self-spervised Pre-training](#self-supervised-pre-training).
For detailed finetuning and inference code, please refer to [this repo](https://github.com/Ezra-Yu/ACCV2022_FGIA_1st).

## Description

### Overview of Our Solution

![image](https://user-images.githubusercontent.com/18586273/205498371-31dbc1f4-5814-44bc-904a-f0d32515c7dd.png)

### Our Model

- ViT(MAE-pre-train)   # Pretrained with [MAE](https://github.com/open-mmlab/mmppretrain/tree/main/projects/fgia_accv2022_1st/config/mae_vit-large-p16_8xb512-amp-coslr-1600e_in1k.py)
- Swin-v2(SimMIM-pre-train)   # From [MMPretrain-swin_transformer_v2](https://github.com/open-mmlab/mmppretrain/tree/main/configs/swin_transformer_v2).

\*\*The architectures we use \*\*

- ViT + CE-loss + post-LongTail-Adjusment
- ViT + SubCenterArcFaceWithAdvMargin(CE)
- Swin-B + SubCenterArcFaceWithAdvMargin(SoftMax-EQL)
- Swin-L + SubCenterArcFaceWithAdvMargin(SoftMAx-EQL)

## Self-supervised Pre-training

### Requirements

```shell
PyTorch 1.11.0
torchvision 0.12.0
CUDA 11.3
MMEngine >= 0.1.0
MMCV >= 2.0.0rc0
```

### Preparing the dataset

First you should refactor the folder of your dataset in the following format:

```text
mmpretrain
|
|── data
|    |── WebiNat5000
|    |       |── meta
|    |       |    |── train.txt
|    |       |── train
|    |       |── testa
|    |       |── testb
```

The `train`, `testa`, and `testb` folders contain the same content with
those provided by the official website of the competition.

### Start pre-training

First, you should install all these requirements, following this [page](https://mmpretrain.readthedocs.io/en/latest/get_started.html).
Then change your current directory to the root of MMPretrain

```shell
cd $MMPretrain
```

Then you have the following two choices to start pre-training

#### Slurm

If you have a cluster managed by Slurm, you can use the following command:

```shell
## we use 16 NVIDIA 80G A100 GPUs for pre-training
GPUS_PER_NODE=8 GPUS=16 SRUN_ARGS=${SRUN_ARGS} bash tools/slurm_train.sh ${PARTITION} ${JOB_NAME} projects/fgia_accv2022_1st/config/mae_vit-large-p16_8xb512-amp-coslr-1600e_in1k.py [optional arguments]
```

#### Pytorch

Or you can use the following two commands to start distributed training on two separate nodes:

```shell
# node 1
NNODES=2 NODE_RANK=0 PORT=${MASTER_PORT} MASTER_ADDR=${MASTER_ADDR} bash tools/dist_train.sh projects/fgia_accv2022_1st/config/mae_vit-large-p16_8xb512-amp-coslr-1600e_in1k.py 8
```

```shell
# node 2
NNODES=2 NODE_RANK=1 PORT=${MASTER_PORT} MASTER_ADDR=${MASTER_ADDR} bash tools/dist_train.sh projects/fgia_accv2022_1st/config/mae_vit-large-p16_8xb512-amp-coslr-1600e_in1k.py 8
```

All these logs and checkpoints will be saved under the folder `work_dirs`in the root.

## Fine-tuning with bag of tricks

- [MAE](https://github.com/open-mmlab/mmpretrain/tree/main/configs/mae) |  [Config](https://github.com/Ezra-Yu/ACCV_workshop/tree/master/configs/vit)
- [Swinv2](https://github.com/open-mmlab/mmpretrain/tree/main/configs/swin_transformer_v2) | [Config](https://github.com/Ezra-Yu/ACCV_workshop/tree/master/configs/swin)
- [ArcFace](https://arxiv.org/abs/1801.07698)   |   [Code](https://github.com/Ezra-Yu/ACCV_workshop/blob/master/src/models/arcface_head.py)
- [SubCenterArcFaceWithAdvMargin](https://paperswithcode.com/paper/sub-center-arcface-boosting-face-recognition)   |   [Code](https://github.com/Ezra-Yu/ACCV_workshop/blob/master/src/models/arcface_head.py)
- [Post-LT-adjusment](https://paperswithcode.com/paper/long-tail-learning-via-logit-adjustment)   |   [Code](https://github.com/Ezra-Yu/ACCV_workshop/blob/master/src/models/linear_head_lt.py)
- [SoftMaxEQL](https://paperswithcode.com/paper/the-equalization-losses-gradient-driven)   |   [Code](https://github.com/Ezra-Yu/ACCV_workshop/blob/master/src/models/eql.py)
- FlipTTA [Code](https://github.com/Ezra-Yu/ACCV_workshop/blob/master/src/models/tta_classifier.py)
- clean dataset
- self-emsemble: [Uniform-model-soup](https://arxiv.org/abs/2203.05482) | [code](https://github.com/Ezra-Yu/ACCV_workshop/blob/master/tools/model_soup.py)
- [pseudo](https://lilianweng.github.io/posts/2021-12-05-semi-supervised/)  | [Code](https://github.com/Ezra-Yu/ACCV_workshop/blob/master/tools/creat_pseudo.py)
- bagging-emsemble [Code](https://github.com/Ezra-Yu/ACCV_workshop/blob/master/tools/emsemble.py),
- post-process: [re-distribute-label](https://github.com/Ezra-Yu/ACCV_workshop/blob/master/tools/re-distribute-label.py);

![Overview](https://user-images.githubusercontent.com/18586273/205498258-e5720d83-7006-4aea-86b5-aab1a8998c6c.png)

![image](https://user-images.githubusercontent.com/18586273/205498027-def99b0d-a99a-470b-b292-8d5fc83111fc.png)

#### Used but no improvements

1. Using retrieval paradigm to solve this classification task;
2. Using EfficientNetv2 backbone.

#### Not used but worth to do

1. Try [DiVE](https://arxiv.org/abs/2103.15042) algorithm to improve performance in long tail dataset;
2. Use SimMIM to pre-train Swin-v2 on the competition dataset;
3. refine the re-distribute-label tool.
