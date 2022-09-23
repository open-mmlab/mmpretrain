# RepVGG

> [Repvgg: Making vgg-style convnets great again](https://arxiv.org/abs/2101.03697)

<!-- [ALGORITHM] -->

## Abstract

We present a simple but powerful architecture of convolutional neural network, which has a VGG-like inference-time body composed of nothing but a stack of 3x3 convolution and ReLU, while the training-time model has a multi-branch topology. Such decoupling of the training-time and inference-time architecture is realized by a structural re-parameterization technique so that the model is named RepVGG. On ImageNet, RepVGG reaches over 80% top-1 accuracy, which is the first time for a plain model, to the best of our knowledge. On NVIDIA 1080Ti GPU, RepVGG models run 83% faster than ResNet-50 or 101% faster than ResNet-101 with higher accuracy and show favorable accuracy-speed trade-off compared to the state-of-the-art models like EfficientNet and RegNet.

<div align=center>
<img src="https://user-images.githubusercontent.com/26739999/142573223-f7f14d32-ea08-43a1-81ad-5a6a83ee0122.png" width="60%"/>
</div>

## Results and models

### ImageNet-1k

|     Model     | Epochs |             Params(M)             |            Flops(G)             | Top-1 (%) | Top-5 (%) |                      Config                      |                      Download                       |
| :-----------: | :----: | :-------------------------------: | :-----------------------------: | :-------: | :-------: | :----------------------------------------------: | :-------------------------------------------------: |
|  RepVGG-A0\*  |  120   |   9.11（train) \| 8.31 (deploy)   |  1.52 (train) \| 1.36 (deploy)  |   72.41   |   90.50   | [config (train)](https://github.com/open-mmlab/mmclassification/blob/master/configs/repvgg/repvgg-A0_4xb64-coslr-120e_in1k.py) \| [config (deploy)](https://github.com/open-mmlab/mmclassification/blob/master/configs/repvgg/deploy/repvgg-A0_deploy_4xb64-coslr-120e_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-A0_3rdparty_4xb64-coslr-120e_in1k_20210909-883ab98c.pth) |
|  RepVGG-A1\*  |  120   |  14.09 (train) \| 12.79 (deploy)  |  2.64 (train) \| 2.37 (deploy)  |   74.47   |   91.85   | [config (train)](https://github.com/open-mmlab/mmclassification/blob/master/configs/repvgg/repvgg-A1_4xb64-coslr-120e_in1k.py) \| [config (deploy)](https://github.com/open-mmlab/mmclassification/blob/master/configs/repvgg/deploy/repvgg-A1_deploy_4xb64-coslr-120e_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-A1_3rdparty_4xb64-coslr-120e_in1k_20210909-24003a24.pth) |
|  RepVGG-A2\*  |  120   |  28.21 (train) \| 25.5 (deploy)   |  5.7 (train)  \| 5.12 (deploy)  |   76.48   |   93.01   | [config (train)](https://github.com/open-mmlab/mmclassification/blob/master/configs/repvgg/repvgg-A2_4xb64-coslr-120e_in1k.py) \|[config (deploy)](https://github.com/open-mmlab/mmclassification/blob/master/configs/repvgg/deploy/repvgg-A2_deploy_4xb64-coslr-120e_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-A2_3rdparty_4xb64-coslr-120e_in1k_20210909-97d7695a.pth) |
|  RepVGG-B0\*  |  120   |  15.82 (train) \| 14.34 (deploy)  |  3.42 (train) \| 3.06 (deploy)  |   75.14   |   92.42   | [config (train)](https://github.com/open-mmlab/mmclassification/blob/master/configs/repvgg/repvgg-B0_4xb64-coslr-120e_in1k.py) \|[config (deploy)](https://github.com/open-mmlab/mmclassification/blob/master/configs/repvgg/deploy/repvgg-B0_deploy_4xb64-coslr-120e_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-B0_3rdparty_4xb64-coslr-120e_in1k_20210909-446375f4.pth) |
|  RepVGG-B1\*  |  120   |  57.42 (train) \| 51.83 (deploy)  | 13.16 (train) \| 11.82 (deploy) |   78.37   |   94.11   | [config (train)](https://github.com/open-mmlab/mmclassification/blob/master/configs/repvgg/repvgg-B1_4xb64-coslr-120e_in1k.py) \|[config (deploy)](https://github.com/open-mmlab/mmclassification/blob/master/configs/repvgg/deploy/repvgg-B1_deploy_4xb64-coslr-120e_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-B1_3rdparty_4xb64-coslr-120e_in1k_20210909-750cdf67.pth) |
| RepVGG-B1g2\* |  120   |  45.78 (train) \| 41.36 (deploy)  |  9.82 (train) \| 8.82 (deploy)  |   77.79   |   93.88   | [config (train)](https://github.com/open-mmlab/mmclassification/blob/master/configs/repvgg/repvgg-B1g2_4xb64-coslr-120e_in1k.py) \|[config (deploy)](https://github.com/open-mmlab/mmclassification/blob/master/configs/repvgg/deploy/repvgg-B1g2_deploy_4xb64-coslr-120e_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-B1g2_3rdparty_4xb64-coslr-120e_in1k_20210909-344f6422.pth) |
| RepVGG-B1g4\* |  120   |  39.97 (train) \| 36.13 (deploy)  |  8.15 (train) \| 7.32 (deploy)  |   77.58   |   93.84   | [config (train)](https://github.com/open-mmlab/mmclassification/blob/master/configs/repvgg/repvgg-B1g4_4xb64-coslr-120e_in1k.py) \|[config (deploy)](https://github.com/open-mmlab/mmclassification/blob/master/configs/repvgg/deploy/repvgg-B1g4_deploy_4xb64-coslr-120e_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-B1g4_3rdparty_4xb64-coslr-120e_in1k_20210909-d4c1a642.pth) |
|  RepVGG-B2\*  |  120   |  89.02 (train) \| 80.32 (deploy)  | 20.46 (train) \| 18.39 (deploy) |   78.78   |   94.42   | [config (train)](https://github.com/open-mmlab/mmclassification/blob/master/configs/repvgg/repvgg-B2_4xb64-coslr-120e_in1k.py) \|[config (deploy)](https://github.com/open-mmlab/mmclassification/blob/master/configs/repvgg/deploy/repvgg-B2_deploy_4xb64-coslr-120e_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-B2_3rdparty_4xb64-coslr-120e_in1k_20210909-bd6b937c.pth) |
| RepVGG-B2g4\* |  200   |  61.76 (train) \| 55.78 (deploy)  | 12.63 (train) \| 11.34 (deploy) |   79.38   |   94.68   | [config (train)](https://github.com/open-mmlab/mmclassification/blob/master/configs/repvgg/repvgg-B2g4_4xb64-autoaug-lbs-mixup-coslr-200e_in1k.py) \|[config (deploy)](https://github.com/open-mmlab/mmclassification/blob/master/configs/repvgg/deploy/repvgg-B2g4_deploy_4xb64-autoaug-lbs-mixup-coslr-200e_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-B2g4_3rdparty_4xb64-autoaug-lbs-mixup-coslr-200e_in1k_20210909-7b7955f0.pth) |
|  RepVGG-B3\*  |  200   | 123.09 (train) \| 110.96 (deploy) | 29.17 (train) \| 26.22 (deploy) |   80.52   |   95.26   | [config (train)](https://github.com/open-mmlab/mmclassification/blob/master/configs/repvgg/repvgg-B3_4xb64-autoaug-lbs-mixup-coslr-200e_in1k.py) \|[config (deploy)](https://github.com/open-mmlab/mmclassification/blob/master/configs/repvgg/deploy/repvgg-B3_deploy_4xb64-autoaug-lbs-mixup-coslr-200e_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-B3_3rdparty_4xb64-autoaug-lbs-mixup-coslr-200e_in1k_20210909-dda968bf.pth) |
| RepVGG-B3g4\* |  200   |  83.83 (train) \| 75.63 (deploy)  | 17.9 (train) \| 16.08 (deploy)  |   80.22   |   95.10   | [config (train)](https://github.com/open-mmlab/mmclassification/blob/master/configs/repvgg/repvgg-B3g4_4xb64-autoaug-lbs-mixup-coslr-200e_in1k.py) \|[config (deploy)](https://github.com/open-mmlab/mmclassification/blob/master/configs/repvgg/deploy/repvgg-B3g4_deploy_4xb64-autoaug-lbs-mixup-coslr-200e_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-B3g4_3rdparty_4xb64-autoaug-lbs-mixup-coslr-200e_in1k_20210909-4e54846a.pth) |
| RepVGG-D2se\* |  200   | 133.33 (train) \| 120.39 (deploy) | 36.56 (train) \| 32.85 (deploy) |   81.81   |   95.94   | [config (train)](https://github.com/open-mmlab/mmclassification/blob/master/configs/repvgg/repvgg-D2se_4xb64-autoaug-lbs-mixup-coslr-200e_in1k.py) \|[config (deploy)](https://github.com/open-mmlab/mmclassification/blob/master/configs/repvgg/deploy/repvgg-D2se_deploy_4xb64-autoaug-lbs-mixup-coslr-200e_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-D2se_3rdparty_4xb64-autoaug-lbs-mixup-coslr-200e_in1k_20210909-cf3139b7.pth) |

*Models with * are converted from the [official repo](https://github.com/DingXiaoH/RepVGG). The config files of these models are only for validation. We don't ensure these config files' training accuracy and welcome you to contribute your reproduction results.*

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

backbone_cfg=dict(type='RepVGG',arch='A0'),
backbone = build_backbone(backbone_cfg)
backbone.switch_to_deploy()
```

or

```python
from mmcls.models import build_classifier

cfg = dict(
    type='ImageClassifier',
    backbone=dict(
        type='RepVGG',
        arch='A0'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1280,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

classifier = build_classifier(cfg)
classifier.backbone.switch_to_deploy()
```

## Citation

```
@inproceedings{ding2021repvgg,
  title={Repvgg: Making vgg-style convnets great again},
  author={Ding, Xiaohan and Zhang, Xiangyu and Ma, Ningning and Han, Jungong and Ding, Guiguang and Sun, Jian},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={13733--13742},
  year={2021}
}
```
