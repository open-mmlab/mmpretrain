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

|     Model      |            Params(M)            |            Flops(G)            | Top-1 (%) | Top-5 (%) |                        Config                        |                        Download                         |
| :------------: | :-----------------------------: | :----------------------------: | :-------: | :-------: | :--------------------------------------------------: | :-----------------------------------------------------: |
| MobileOne-s0\* |  5.29（train) \| 2.08 (deploy)  | 1.09 (train) \| 0.28 (deploy)  |   71.36   |   89.87   | [config (train)](./mobileone-s0_8xb128_in1k.py) \| [config (deploy)](./deploy/mobileone-s0_deploy_8xb128_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/mobileone/mobileone-s0_3rdparty_in1k_20220915-007ae971.pth) |
| MobileOne-s1\* |  4.83 (train) \| 4.76 (deploy)  | 0.86 (train) \| 0.84 (deploy)  |   75.76   |   92.77   | [config (train)](./mobileone-s1_8xb128_in1k.py) \| [config (deploy)](./deploy/mobileone-s1_deploy_8xb128_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/mobileone/mobileone-s1_3rdparty_in1k_20220915-473c8469.pth) |
| MobileOne-s2\* |  7.88 (train) \| 7.88 (deploy)  | 1.34 (train)  \| 1.31 (deploy) |   77.39   |   93.63   | [config (train)](./mobileone-s2_8xb128_in1k.py) \|[config (deploy)](./deploy/mobileone-s2_deploy_8xb128_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/mobileone/mobileone-s2_3rdparty_in1k_20220915-ed2e4c30.pth) |
| MobileOne-s3\* | 10.17 (train) \| 10.08 (deploy) | 1.95 (train)  \| 1.91 (deploy) |   77.93   |   93.89   | [config (train)](./mobileone-s3_8xb128_in1k.py) \|[config (deploy)](./deploy/mobileone-s3_deploy_8xb128_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/mobileone/mobileone-s3_3rdparty_in1k_20220915-84d6a02c.pth) |
| MobileOne-s4\* | 14.95 (train) \| 14.84 (deploy) | 3.05 (train) \| 3.00 (deploy)  |   79.30   |   94.37   | [config (train)](./mobileone-s4_8xb128_in1k.py) \|[config (deploy)](./deploy/mobileone-s4_deploy_8xb128_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/mobileone/mobileone-s4_3rdparty_in1k_20220915-ce9509ee.pth) |

*Models with * are converted from the [official repo](https://github.com/apple/ml-mobileone). The config files of these models are only for validation. We don't ensure these config files' training accuracy and welcome you to contribute your reproduction results.*

*Because the [official repo.](https://github.com/apple/ml-mobileone) does not give a strategy for training and testing, the test data pipline of [RepVGG](https://github.com/open-mmlab/mmclassification/tree/master/configs/repvgg) is used here, and the result is about 0.1 lower than that in the paper. Refer to [this issue](https://github.com/apple/ml-mobileone/issues/2).*

## How to use

The checkpoints provided are all `training-time` models. Use the reparameterize tool to switch them to more efficient `inference-time` architecture, which not only has fewer parameters but also less calculations.

### Use tool

Use provided tool to reparameterize the given model and save the checkpoint:

```bash
python tools/convert_models/reparameterize_model.py ${CFG_PATH} ${SRC_CKPT_PATH} ${TARGET_CKPT_PATH}
```

`${CFG_PATH}` is the config file path, `${SRC_CKPT_PATH}` is the source chenpoint file path, `${TARGET_CKPT_PATH}` is the target deploy weight file path.

For example:

```shell
python ./tools/convert_models/reparameterize_model.py ./configs/mobileone/mobileone-s0_8xb128_in1k.py https://download.openmmlab.com/mmclassification/v0/mobileone/mobileone-s0_3rdparty_in1k_20220811-db5ce29b.pth ./mobileone_s0_deploy.pth
```

To use reparameterized weights, the config file must switch to **the deploy config files**.

```bash
python tools/test.py ${Deploy_CFG} ${Deploy_Checkpoint} --metrics accuracy
```

For example of using the reparameterized weights above:

```shell
python ./tools/test.py ./configs/mobileone/deploy/mobileone-s0_deploy_8xb128_in1k.py  mobileone_s0_deploy.pth --metrics accuracy
```

### In the code

Use the API `switch_to_deploy` of `MobileOne` backbone to to switch to the deploy mode. Usually called like `backbone.switch_to_deploy()` or `classificer.backbone.switch_to_deploy()`.

For Backbones:

```python
from mmcls.models import build_backbone
import torch

x = torch.randn( (1, 3, 224, 224) )
backbone_cfg=dict(type='MobileOne', arch='s0')
backbone = build_backbone(backbone_cfg)
backbone.init_weights()
backbone.eval()
outs_ori = backbone(x)

backbone.switch_to_deploy()
outs_dep = backbone(x)

for out1, out2 in zip(outs_ori, outs_dep):
    assert torch.allclose(out1, out2)
```

For ImageClassifiers:

```python
from mmcls.models import build_classifier
import torch
import numpy as np

cfg = dict(
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

x = torch.randn( (1, 3, 224, 224) )
classifier = build_classifier(cfg)
classifier.init_weights()
classifier.eval()
y_ori = classifier(x, return_loss=False)

classifier.backbone.switch_to_deploy()
y_dep = classifier(x, return_loss=False)

for y1, y2 in zip(y_ori, y_dep):
    assert np.allclose(y1, y2)
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
