# MobileOne

> [An Improved One millisecond Mobile Backbone](https://arxiv.org/abs/2206.04040)

<!-- [ALGORITHM] -->

## Abstract

Efficient neural network backbones for mobile devices are often optimized for metrics such as FLOPs or parameter count. However, these metrics may not correlate well with latency of the network when deployed on a mobile device. Therefore, we perform extensive analysis of different metrics by deploying several mobile-friendly networks on a mobile device. We identify and analyze architectural and optimization bottlenecks in recent efficient neural networks and provide ways to mitigate these bottlenecks. To this end, we design an efficient backbone MobileOne, with variants achieving an inference time under 1 ms on an iPhone12 with 75.9% top-1 accuracy on ImageNet. We show that MobileOne achieves state-of-the-art performance within the efficient architectures while being many times faster on mobile. Our best model obtains similar performance on ImageNet as MobileFormer while being 38x faster. Our model obtains 2.3% better top-1 accuracy on ImageNet than EfficientNet at similar latency. Furthermore, we show that our model generalizes to multiple tasks - image classification, object detection, and semantic segmentation with significant improvements in latency and accuracy as compared to existing efficient architectures when deployed on a mobile device.

<div align=center>
<img src="https://user-images.githubusercontent.com/18586273/183552452-74657532-f461-48f7-9aa7-c23f006cdb07.png" width="40%"/>
</div>

## Results and models

### ImageNet-1k

|     Model      |            Params(M)            |           Flops(G)            | Top-1 (%) | Top-5 (%) |                        Config                         |                        Download                         |
| :------------: | :-----------------------------: | :---------------------------: | :-------: | :-------: | :---------------------------------------------------: | :-----------------------------------------------------: |
| MobileOne-s0\* |  9.11（train) \| 8.31 (deploy)  | 1.52 (train) \| 1.36 (deploy) |   71.36   |   89.87   | [config (train)](https://github.com/open-mmlab/mmclassification/blob/master/configs/repvgg/repvgg-A0_4xb64-coslr-120e_in1k.py) \| [config (deploy)](https://github.com/open-mmlab/mmclassification/blob/master/configs/repvgg/deploy/repvgg-A0_deploy_4xb64-coslr-120e_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-A0_3rdparty_4xb64-coslr-120e_in1k_20210909-883ab98c.pth) |
| MobileOne-s1\* | 14.09 (train) \| 12.79 (deploy) | 2.64 (train) \| 2.37 (deploy) |   75.76   |   92.77   | [config (train)](https://github.com/open-mmlab/mmclassification/blob/master/configs/repvgg/repvgg-A1_4xb64-coslr-120e_in1k.py) \| [config (deploy)](https://github.com/open-mmlab/mmclassification/blob/master/configs/repvgg/deploy/repvgg-A1_deploy_4xb64-coslr-120e_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-A1_3rdparty_4xb64-coslr-120e_in1k_20210909-24003a24.pth) |
| MobileOne-s2\* | 28.21 (train) \| 25.5 (deploy)  | 5.7 (train)  \| 5.12 (deploy) |   77.39   |   93.63   | [config (train)](https://github.com/open-mmlab/mmclassification/blob/master/configs/repvgg/repvgg-A2_4xb64-coslr-120e_in1k.py) \|[config (deploy)](https://github.com/open-mmlab/mmclassification/blob/master/configs/repvgg/deploy/repvgg-A2_deploy_4xb64-coslr-120e_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-A2_3rdparty_4xb64-coslr-120e_in1k_20210909-97d7695a.pth) |
| MobileOne-s3\* | 28.21 (train) \| 25.5 (deploy)  | 5.7 (train)  \| 5.12 (deploy) |   77.93   |   93.89   | [config (train)](https://github.com/open-mmlab/mmclassification/blob/master/configs/repvgg/repvgg-A2_4xb64-coslr-120e_in1k.py) \|[config (deploy)](https://github.com/open-mmlab/mmclassification/blob/master/configs/repvgg/deploy/repvgg-A2_deploy_4xb64-coslr-120e_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-A2_3rdparty_4xb64-coslr-120e_in1k_20210909-97d7695a.pth) |
| MobileOne-s4\* | 15.82 (train) \| 14.34 (deploy) | 3.42 (train) \| 3.06 (deploy) |   79.30   |   94.37   | [config (train)](https://github.com/open-mmlab/mmclassification/blob/master/configs/repvgg/repvgg-B0_4xb64-coslr-120e_in1k.py) \|[config (deploy)](https://github.com/open-mmlab/mmclassification/blob/master/configs/repvgg/deploy/repvgg-B0_deploy_4xb64-coslr-120e_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-B0_3rdparty_4xb64-coslr-120e_in1k_20210909-446375f4.pth) |

*Models with * are converted from the [official repo](https://github.com/apple/ml-mobileone). The config files of these models are only for validation. We don't ensure these config files' training accuracy and welcome you to contribute your reproduction results.*

## How to use

The checkpoints provided are all `training-time` models. Use the reparameterize tool to switch them to more efficient `inference-time` architecture, which not only has fewer parameters but also less calculations.

### Use tool

Use provided tool to reparameterize the given model and save the checkpoint:

```bash
python tools/convert_models/reparameterize_model.py ${CFG_PATH} ${SRC_CKPT_PATH} ${TARGET_CKPT_PATH}
```

`${CFG_PATH}` is the config file, `${SRC_CKPT_PATH}` is the source chenpoint file, `${TARGET_CKPT_PATH}` is the target deploy weight file path.

For example:

```shell


```

To use reparameterized weights, the config file must switch to the deploy config files.

```bash
python tools/test.py ${Deploy_CFG} ${Deploy_Checkpoint} --metrics accuracy
```

### In the code

Use `backbone.switch_to_deploy()` or `classificer.backbone.switch_to_deploy()` to switch to the deploy mode. For example:

```python
from mmcls.models import build_backbone

backbone_cfg=dict(type='MobileOne',arch='s0'),
backbone = build_backbone(backbone_cfg)
backbone.switch_to_deploy()
```

or

```python
from mmcls.models import build_classifier

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='MobileOne',
        arch='s0',
        out_indices=(3, ),
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1024,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

classifier = build_classifier(cfg)
classifier.backbone.switch_to_deploy()
```

## Citation

```bibtex
@article{mobileone2022,
  title={An Improved One millisecond Mobile Backbone},
  author={Vasu, Pavan Kumar Anasosalu and Gabriel, James and Zhu, Jeff and Tuzel, Oncel and Ranjan, Anurag},
  journal={arXiv preprint arXiv:2206.04040},
  year={2022}
}
```
