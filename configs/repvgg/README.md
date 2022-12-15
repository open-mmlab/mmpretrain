# RepVGG

> [RepVGG: Making VGG-style ConvNets Great Again](https://arxiv.org/abs/2101.03697)

<!-- [ALGORITHM] -->

## Introduction

RepVGG is a VGG-style convolutional architecture. It has the following advantages:

1. The model has a VGG-like plain (a.k.a. feed-forward) topology 1 without any branches. I.e., every layer takes the output of its only preceding layer as input and feeds the output into its only following layer.
2. The model’s body uses only 3 × 3 conv and ReLU.
3. The concrete architecture (including the specific depth and layer widths) is instantiated with no automatic search, manual refinement, compound scaling, nor other heavy designs.

<div align=center>
<img src="https://user-images.githubusercontent.com/26739999/142573223-f7f14d32-ea08-43a1-81ad-5a6a83ee0122.png" width="60%"/>
</div>

## Abstract

<details>

<summary>Show the paper's abstract</summary>

<br>
We present a simple but powerful architecture of convolutional neural network, which has a VGG-like inference-time body composed of nothing but a stack of 3x3 convolution and ReLU, while the training-time model has a multi-branch topology. Such decoupling of the training-time and inference-time architecture is realized by a structural re-parameterization technique so that the model is named RepVGG. On ImageNet, RepVGG reaches over 80% top-1 accuracy, which is the first time for a plain model, to the best of our knowledge. On NVIDIA 1080Ti GPU, RepVGG models run 83% faster than ResNet-50 or 101% faster than ResNet-101 with higher accuracy and show favorable accuracy-speed trade-off compared to the state-of-the-art models like EfficientNet and RegNet.
</br>

</details>

## How to use

The checkpoints provided are all `training-time` models. Use the reparameterize tool or `switch_to_deploy` interface to switch them to more efficient `inference-time` architecture, which not only has fewer parameters but also less calculations.

<!-- [TABS-BEGIN] -->

**Predict image**

Use `classifier.backbone.switch_to_deploy()` interface to switch the RepVGG models into inference mode.

```python
>>> import torch
>>> from mmcls.apis import init_model, inference_model
>>>
>>> model = init_model('configs/repvgg/repvgg-A0_8xb32_in1k.py', 'https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-A0_8xb32_in1k_20221213-60ae8e23.pth')
>>> results = inference_model(model, 'demo/demo.JPEG')
>>> print( (results['pred_class'], results['pred_score']) )
('sea snake' 0.8338906168937683)
>>>
>>> # switch to deploy mode
>>> model.backbone.switch_to_deploy()
>>> results = inference_model(model, 'demo/demo.JPEG')
>>> print( (results['pred_class'], results['pred_score']) )
('sea snake', 0.7883061170578003)
```

**Use the model**

```python
>>> import torch
>>> from mmcls.apis import get_model
>>>
>>> model = get_model("repvgg-a0_8xb32_in1k", pretrained=True)
>>> model.eval()
>>> inputs = torch.rand(1, 3, 224, 224).to(model.data_preprocessor.device)
>>> # To get classification scores.
>>> out = model(inputs)
>>> print(out.shape)
torch.Size([1, 1000])
>>> # To extract features.
>>> outs = model.extract_feat(inputs)
>>> print(outs[0].shape)
torch.Size([1, 1280])
>>>
>>> # switch to deploy mode
>>> model.backbone.switch_to_deploy()
>>> out_deploy = model(inputs)
>>> print(out.shape)
torch.Size([1, 1000])
>>> assert torch.allclose(out, out_deploy, rtol=1e-4, atol=1e-5) # pass without error
```

**Train/Test Command**

Place the ImageNet dataset to the `data/imagenet/` directory, or prepare datasets according to the [docs](https://mmclassification.readthedocs.io/en/1.x/user_guides/dataset_prepare.html#prepare-dataset).

Train:

```shell
python tools/train.py configs/repvgg/repvgg-a0_8xb32_in1k.py
```

Download Checkpoint:

```shell
wget https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-A0_8xb32_in1k_20221213-60ae8e23.pth
```

Test use unfused model:

```shell
python tools/test.py configs/repvgg/repvgg-a0_8xb32_in1k.py repvgg-A0_8xb32_in1k_20221213-60ae8e23.pth
```

Reparameterize checkpoint:

```shell
python ./tools/convert_models/reparameterize_model.py configs/repvgg/repvgg-a0_8xb32_in1k.py repvgg-A0_8xb32_in1k_20221213-60ae8e23.pth repvgg_A0_deploy.pth
```

Test use fused model:

```shell
python tools/test.py configs/repvgg/repvgg-A0_8xb32_in1k.py repvgg_A0_deploy.pth --cfg-options model.backbone.deploy=True
```

or

```shell
python tools/test.py configs/repvgg/repvgg-A0_deploy_in1k.py repvgg_A0_deploy.pth
```

<!-- [TABS-END] -->

For more configurable parameters, please refer to the [API](https://mmclassification.readthedocs.io/en/1.x/api/generated/mmcls.models.backbones.RepVGG.html#mmcls.models.backbones.RepVGG).

<details>

<summary><b>How to use the reparameterisation tool</b>(click to show)</summary>

<br>

Use provided tool to reparameterize the given model and save the checkpoint:

```bash
python tools/convert_models/reparameterize_model.py ${CFG_PATH} ${SRC_CKPT_PATH} ${TARGET_CKPT_PATH}
```

`${CFG_PATH}` is the config file path, `${SRC_CKPT_PATH}` is the source chenpoint file path, `${TARGET_CKPT_PATH}` is the target deploy weight file path.

For example:

```shell
# download the weight
wget https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-A0_8xb32_in1k_20221213-60ae8e23.pth
# reparameterize unfused weight to fused weight
python ./tools/convert_models/reparameterize_model.py configs/repvgg/repvgg-a0_8xb32_in1k.py repvgg-A0_8xb32_in1k_20221213-60ae8e23.pth repvgg-A0_deploy.pth
```

To use reparameterized weights, the config file must switch to **the deploy config files** as [the deploy_A0 example](./repvgg-A0_deploy_in1k.py) or add `--cfg-options model.backbone.deploy=True` in command.

For example of using the reparameterized weights above:

```shell
python ./tools/test.py ./configs/repvgg/repvgg-A0_deploy_in1k.py  repvgg-A0_deploy.pth
```

You can get other deploy configs by modifying the [A0_deploy example](./repvgg-A0_deploy_in1k.py):

```text
# in repvgg-A0_deploy_in1k.py
_base_ = '../repvgg-A0_8xb32_in1k.py'  # basic A0 config

model = dict(backbone=dict(deploy=True))  # switch model into deploy mode
```

or add `--cfg-options model.backbone.deploy=True` in command as following：

```shell
python tools/test.py configs/repvgg/repvgg-A0_8xb32_in1k.py repvgg_A0_deploy.pth --cfg-options model.backbone.deploy=True
```

</br>

</details>

## Results and models

### ImageNet-1k

|            Model            |   Pretrain   | <p> Params(M) <br>（train\|deploy) </p> | <p> Flops(G)  <br>（train\|deploy)  </p> | Top-1 (%) | Top-5 (%) |             Config              |             Download              |
| :-------------------------: | :----------: | :-------------------------------------: | :--------------------------------------: | :-------: | :-------: | :-----------------------------: | :-------------------------------: |
|    repvgg-A0_8xb32_in1k     | From scratch |              9.11 \| 8.31               |               1.53 \| 1.36               |   72.37   |   90.56   | [config](./repvgg-A0_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-A0_8xb32_in1k_20221213-60ae8e23.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-A0_8xb32_in1k_20221213-60ae8e23.log) |
|    repvgg-A1_8xb32_in1k     | From scratch |             14.09  \| 12.79             |               2.65 \| 2.37               |   74.47   |   91.85   | [config](./repvgg-A1_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-A1_8xb32_in1k_20221213-f81bf3df.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-A1_8xb32_in1k_20221213-f81bf3df.log) |
|    repvgg-A2_8xb32_in1k     | From scratch |             28.21   \| 25.5             |             5.72    \| 5.12              |   76.49   |   93.09   | [config](./repvgg-A2_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-A2_8xb32_in1k_20221213-a8767caf.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-A2_8xb32_in1k_20221213-a8767caf.log) |
|    repvgg-B0_8xb32_in1k     | From scratch |            15.82   \| 14.34             |              3.43   \| 3.06              |   75.27   |   92.21   | [config](./repvgg-B0_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-B0_8xb32_in1k_20221213-5091ecc7.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-B0_8xb32_in1k_20221213-5091ecc7.log) |
|    repvgg-B1_8xb32_in1k     | From scratch |            57.42   \| 51.83             |             13.20   \| 11.81             |   78.19   |   94.04   | [config](./repvgg-B1_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-B1_8xb32_in1k_20221213-d17c45e7.pth)   \| [log](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-B1_8xb32_in1k_20221213-d17c45e7.log) |
|   repvgg-B1g2_8xb32_in1k    | From scratch |            45.78   \| 41.36             |              9.86   \| 8.80              |   77.87   |   93.99   | [config](./repvgg-B1g2_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-B1g2_8xb32_in1k_20221213-ae6428fd.pth)  \| [log](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-B1g2_8xb32_in1k_20221213-ae6428fd.log) |
|   repvgg-B1g4_8xb32_in1k    | From scratch |            39.97   \| 36.13             |              8.19   \| 7.30              |   77.81   |   93.77   | [config](./repvgg-B1g4_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-B1g4_8xb32_in1k_20221213-a7a4aaea.pth)  \| [log](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-B1g4_8xb32_in1k_20221213-a7a4aaea.log) |
|    repvgg-B2_8xb32_in1k     | From scratch |            89.02   \| 80.32             |              20.5   \| 18.4              |   78.58   |   94.23   | [config](./repvgg-B2_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-B2_8xb32_in1k_20221213-d8b420ef.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-B2_8xb32_in1k_20221213-d8b420ef.log) |
|   repvgg-B2g4_8xb32_in1k    | From scratch |            61.76   \| 55.78             |              12.7   \| 11.3              |   79.44   |   94.72   | [config](./repvgg-B2g4_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-B2g4_8xb32_in1k_20221213-0c1990eb.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-B2g4_8xb32_in1k_20221213-0c1990eb.log) |
|    repvgg-B3_8xb32_in1k     | From scratch |           123.09   \| 110.96            |              29.2   \| 26.2              |   80.58   |   95.33   | [config](./repvgg-B3_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-B3_8xb32_in1k_20221213-927a329a.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-B3_8xb32_in1k_20221213-927a329a.log) |
|   repvgg-B3g4_8xb32_in1k    | From scratch |            83.83   \| 75.63             |              18.0   \| 16.1              |   80.26   |   95.15   | [config](./repvgg-B3g4_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-B3g4_8xb32_in1k_20221213-e01cb280.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-B3g4_8xb32_in1k_20221213-e01cb280.log) |
| repvgg-D2se_3rdparty_in1k\* | From scratch |           133.33   \| 120.39            |              36.6   \| 32.8              |   81.81   |   95.94   | [config](./repvgg-D2se_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-D2se_3rdparty_4xb64-autoaug-lbs-mixup-coslr-200e_in1k_20210909-cf3139b7.pth) |

*Models with * are converted from the [official repo](https://github.com/DingXiaoH/RepVGG/blob/9f272318abfc47a2b702cd0e916fca8d25d683e7/repvgg.py#L250). The config files of these models are only for inference. We don't ensure these config files' training accuracy and welcome you to contribute your reproduction results.*

## Citation

```bibtex
@inproceedings{ding2021repvgg,
  title={Repvgg: Making vgg-style convnets great again},
  author={Ding, Xiaohan and Zhang, Xiangyu and Ma, Ningning and Han, Jungong and Ding, Guiguang and Sun, Jian},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={13733--13742},
  year={2021}
}
```
