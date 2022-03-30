# RepMLP

> [RepMLP: Re-parameterizing Convolutions into Fully-connected Layers forImage Recognition](https://arxiv.org/abs/2105.01883)
<!-- [ALGORITHM] -->

## Abstract

We propose RepMLP, a multi-layer-perceptron-style neural network building block for image recognition, which is composed of a series of fully-connected (FC) layers. Compared to convolutional layers, FC layers are more efficient, better at modeling the long-range dependencies and positional patterns, but worse at capturing the local structures, hence usually less favored for image recognition. We propose a structural re-parameterization technique that adds local prior into an FC to make it powerful for image recognition. Specifically, we construct convolutional layers inside a RepMLP during training and merge them into the FC for inference. On CIFAR, a simple pure-MLP model shows performance very close to CNN. By inserting RepMLP in traditional CNN, we improve ResNets by 1.8% accuracy on ImageNet, 2.9% for face recognition, and 2.3% mIoU on Cityscapes with lower FLOPs. Our intriguing findings highlight that combining the global representational capacity and positional perception of FC with the local prior of convolution can improve the performance of neural network with faster speed on both the tasks with translation invariance (e.g., semantic segmentation) and those with aligned images and positional patterns (e.g., face recognition).

<div align=center>
<img src="https://user-images.githubusercontent.com/18586273/155455288-a17a5c48-11af-4b74-995a-cf7183f0e2d2.png" width="80%"/>
</div>


## Results and models

### ImageNet-1k

|      Model     | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) | Config | Download |
|:--------------:|:---------:|:--------:|:---------:|:---------:|:------:|:--------:|
|  RepMLP-B224\*  |  68.24   |  6.71    |  80.41    |  95.12    | [train_cfg](https://github.com/open-mmlab/mmclassification/blob/master/configs/repmlp/repmlp-base_8xb64_in1k.py) \| [deploy_cfg](https://github.com/open-mmlab/mmclassification/blob/master/configs/repmlp/repmlp-base_delopy_8xb64_in1k.py)   | [model](https://download.openmmlab.com/mmclassification/v0/repmlp/repmlp-base_3rdparty_8xb64_in1k_20220330-1cb1f11b.pth) |
|  RepMLP-B256\*  |  96.45   |  9.69    |  81.11    |  95.5    | [train_cfg](https://github.com/open-mmlab/mmclassification/blob/master/configs/repmlp/repmlp-base_8xb64_in1k-256px.py) \| [deploy_cfg](https://github.com/open-mmlab/mmclassification/blob/master/configs/repmlp/repmlp-b256_deploy_8xb64_in1k.py)  | [model](https://download.openmmlab.com/mmclassification/v0/repmlp/repmlp-base_3rdparty_8xb64_in1k-256px_20220330-7c5a91ce.pth) |

*Models with \* are converted from [the official repo.](https://github.com/DingXiaoH/RepMLP). The config files of these models are only for validation. We don't ensure these config files' training accuracy and welcome you to contribute your reproduction results.*


## How to use

The checkpoints provided are all `training-time` models. Use the reparameterize tool to switch them to more efficient `inference-time` architecture, which not only has fewer parameters but also less calculations.

### Use tool

Use provided tool to reparameterize the given model and save the checkpoint:

```bash
python tools/convert_models/reparameterize_model.py ${CFG_PATH} ${SRC_CKPT_PATH} ${TARGET_CKPT_PATH}
```

`${CFG_PATH}` is the config file, `${SRC_CKPT_PATH}` is the source chenpoint file, `${TARGET_CKPT_PATH}` is the target deploy weight file path.

To use reparameterized weights, the config file must switch to the deploy config files.

```bash
python tools/test.py ${Deploy_CFG} ${Deploy_Checkpoint} --metrics accuracy
```

### In the code

Use `backbone.switch_to_deploy()` or `classificer.backbone.switch_to_deploy()` to switch to the deploy mode. For example:

```python
from mmcls.models import build_backbone

backbone_cfg=dict(type='RepMLPNet', arch='B', img_size=224, reparam_conv_kernels=(1, 3), deploy=False)
backbone = build_backbone(backbone_cfg)
backbone.switch_to_deploy()
```

or

```python
from mmcls.models import build_classifier

cfg = dict(
    type='ImageClassifier',
    backbone=dict(
        type='RepMLPNet',
        arch='B',
        img_size=224,
        reparam_conv_kernels=(1, 3),
        deploy=False),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=768,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

classifier = build_classifier(cfg)
classifier.backbone.switch_to_deploy()
```


## Citation

```
@article{ding2021repmlp,
  title={Repmlp: Re-parameterizing convolutions into fully-connected layers for image recognition},
  author={Ding, Xiaohan and Xia, Chunlong and Zhang, Xiangyu and Chu, Xiaojie and Han, Jungong and Ding, Guiguang},
  journal={arXiv preprint arXiv:2105.01883},
  year={2021}
}
```
