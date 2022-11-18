# RepLKNet

> [Scaling Up Your Kernels to 31x31: Revisiting Large Kernel Design in CNNs](https://arxiv.org/abs/2203.06717)

<!-- [ALGORITHM] -->

## Abstract

We revisit large kernel design in modern convolutional neural networks (CNNs). Inspired by recent advances in vision transformers (ViTs), in this paper, we demonstrate that using a few large convolutional kernels instead of a stack of small kernels could be a more powerful paradigm. We suggested five guidelines, e.g., applying re-parameterized large depth-wise convolutions, to design efficient highperformance large-kernel CNNs. Following the guidelines, we propose RepLKNet, a pure CNN architecture whose kernel size is as large as 31×31, in contrast to commonly used 3×3. RepLKNet greatly closes the performance gap between CNNs and ViTs, e.g., achieving comparable or superior results than Swin Transformer on ImageNet and a few typical downstream tasks, with lower latency. RepLKNet also shows nice scalability to big data and large models, obtaining 87.8% top-1 accuracy on ImageNet and 56.0% mIoU on ADE20K, which is very competitive among the state-of-the-arts with similar model sizes. Our study further reveals that, in contrast to small-kernel CNNs, large kernel CNNs have much larger effective receptive fields and higher shape bias rather than texture bias.

<div align=center>
<img src="https://user-images.githubusercontent.com/48375204/197546040-cdf078c3-7fbd-400f-8b27-01668c8dfebf.png" width="60%"/>
</div>

## Results and models

### ImageNet-1k

|     Model      | Resolution | Pretrained Dataset |            Params(M)            |            Flops(G)             | Top-1 (%) | Top-5 (%) |                 Config                 |                 Download                 |
| :------------: | :--------: | :----------------: | :-----------------------------: | :-----------------------------: | :-------: | :-------: | :------------------------------------: | :--------------------------------------: |
| RepLKNet-31B\* |  224x224   |    From Scratch    |  79.9（train) \| 79.5 (deploy)  |  15.6 (train) \| 15.4 (deploy)  |   83.48   |   96.57   | [config (train)](./replknet-31B_32xb64_in1k.py) \| [config (deploy)](./deploy/replknet-31B-deploy_32xb64_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/replknet/replknet-31B_3rdparty_in1k_20221118-fd08e268.pth) |
| RepLKNet-31B\* |  384x384   |    From Scratch    |  79.9（train) \| 79.5 (deploy)  |  46.0 (train) \| 45.3 (deploy)  |   84.84   |   97.34   | [config (train)](./replknet-31B_32xb64_in1k-384px.py) \| [config (deploy)](./deploy/replknet-31B-deploy_32xb64_in1k-384px.py) | [model](https://download.openmmlab.com/mmclassification/v0/replknet/replknet-31B_3rdparty_in1k-384px_20221118-03a170ce.pth) |
| RepLKNet-31B\* |  224x224   |    ImageNet-21K    |  79.9（train) \| 79.5 (deploy)  |  15.6 (train) \| 15.4 (deploy)  |   85.20   |   97.56   | [config (train)](./replknet-31B_32xb64_in1k.py) \| [config (deploy)](./deploy/replknet-31B-deploy_32xb64_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/replknet/replknet-31B_in21k-pre_3rdparty_in1k_20221118-54ed5c46.pth) |
| RepLKNet-31B\* |  384x384   |    ImageNet-21K    |  79.9（train) \| 79.5 (deploy)  |  46.0 (train) \| 45.3 (deploy)  |   85.99   |   97.75   | [config (train)](./replknet-31B_32xb64_in1k-384px.py) \| [config (deploy)](./deploy/replknet-31B-deploy_32xb64_in1k-384px.py) | [model](https://download.openmmlab.com/mmclassification/v0/replknet/replknet-31B_in21k-pre_3rdparty_in1k-384px_20221118-76c92b24.pth) |
| RepLKNet-31L\* |  384x384   |    ImageNet-21K    | 172.7（train) \| 172.0 (deploy) |  97.2 (train) \| 97.0 (deploy)  |   86.63   |   98.00   | [config (train)](./replknet-31L_32xb64_in1k-384px.py) \| [config (deploy)](./deploy/replknet-31L-deploy_32xb64_in1k-384px.py) | [model](https://download.openmmlab.com/mmclassification/v0/replknet/replknet-31L_in21k-pre_3rdparty_in1k-384px_20221118-dc3fc07c.pth) |
| RepLKNet-XL\*  |  320x320   |    MegData-73M     | 335.4（train) \| 335.0 (deploy) | 129.6 (train) \| 129.0 (deploy) |   87.57   |   98.39   | [config (train)](./replknet-XL_32xb64_in1k-320px.py) \| [config (deploy)](./deploy/replknet-XL-deploy_32xb64_in1k-320px.py) | [model](https://download.openmmlab.com/mmclassification/v0/replknet/replknet-XL_meg73m-pre_3rdparty_in1k-320px_20221118-88259b1d.pth) |

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

backbone_cfg=dict(type='RepLKNet',arch='31B'),
backbone = build_backbone(backbone_cfg)
backbone.switch_to_deploy()
```

or

```python
from mmcls.models import build_classifier

cfg = dict(
    type='ImageClassifier',
    backbone=dict(
        type='RepLKNet',
        arch='31B'),
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

```
@inproceedings{ding2022scaling,
  title={Scaling up your kernels to 31x31: Revisiting large kernel design in cnns},
  author={Ding, Xiaohan and Zhang, Xiangyu and Han, Jungong and Ding, Guiguang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11963--11975},
  year={2022}
}
```
