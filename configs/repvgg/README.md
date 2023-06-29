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

## How to use it?

<!-- [TABS-BEGIN] -->

**Predict image**

```python
from mmpretrain import inference_model, get_model

model = get_model('repvgg-A0_8xb32_in1k', pretrained=True)
model.backbone.switch_to_deploy()
predict = inference_model(model, 'demo/bird.JPEG')
print(predict['pred_class'])
print(predict['pred_score'])
```

**Use the model**

```python
import torch
from mmpretrain import get_model

model = get_model('repvgg-A0_8xb32_in1k', pretrained=True)
inputs = torch.rand(1, 3, 224, 224)
out = model(inputs)
print(type(out))
# To extract features.
feats = model.extract_feat(inputs)
print(type(feats))
```

**Train/Test Command**

Prepare your dataset according to the [docs](https://mmpretrain.readthedocs.io/en/latest/user_guides/dataset_prepare.html#prepare-dataset).

Train:

```shell
python tools/train.py configs/repvgg/repvgg-A0_8xb32_in1k.py
```

Test:

```shell
python tools/test.py configs/repvgg/repvgg-A0_8xb32_in1k.py https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-A0_8xb32_in1k_20221213-60ae8e23.pth
```

Test with reparameterized model:

```shell
python tools/test.py configs/repvgg/repvgg-A0_8xb32_in1k.py repvgg_A0_deploy.pth --cfg-options model.backbone.deploy=True
```

**Reparameterization**

The checkpoints provided are all `training-time` models. Use the reparameterize tool to switch them to more efficient `inference-time` architecture, which not only has fewer parameters but also less calculations.

```bash
python tools/convert_models/reparameterize_model.py ${CFG_PATH} ${SRC_CKPT_PATH} ${TARGET_CKPT_PATH}
```

`${CFG_PATH}` is the config file, `${SRC_CKPT_PATH}` is the source chenpoint file, `${TARGET_CKPT_PATH}` is the target deploy weight file path.

To use reparameterized weights, the config file must switch to the deploy config files.

```bash
python tools/test.py ${deploy_cfg} ${deploy_checkpoint} --metrics accuracy
```

You can also use `backbone.switch_to_deploy()` to switch to the deploy mode in Python code. For example:

```python
from mmpretrain.models import RepVGG

backbone = RepVGG(arch='A0')
backbone.switch_to_deploy()
```

<!-- [TABS-END] -->

## Models and results

### Image Classification on ImageNet-1k

| Model                         |   Pretrain   | Params (M) | Flops (G) | Top-1 (%) | Top-5 (%) |               Config                |                                        Download                                         |
| :---------------------------- | :----------: | :--------: | :-------: | :-------: | :-------: | :---------------------------------: | :-------------------------------------------------------------------------------------: |
| `repvgg-A0_8xb32_in1k`        | From scratch |    8.31    |   1.36    |   72.37   |   90.56   |  [config](repvgg-A0_8xb32_in1k.py)  | [model](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-A0_8xb32_in1k_20221213-60ae8e23.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-A0_8xb32_in1k_20221213-60ae8e23.log) |
| `repvgg-A1_8xb32_in1k`        | From scratch |   12.79    |   2.36    |   74.23   |   91.80   |  [config](repvgg-A1_8xb32_in1k.py)  | [model](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-A1_8xb32_in1k_20221213-f81bf3df.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-A1_8xb32_in1k_20221213-f81bf3df.log) |
| `repvgg-A2_8xb32_in1k`        | From scratch |   25.50    |   5.12    |   76.49   |   93.09   |  [config](repvgg-A2_8xb32_in1k.py)  | [model](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-A2_8xb32_in1k_20221213-a8767caf.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-A2_8xb32_in1k_20221213-a8767caf.log) |
| `repvgg-B0_8xb32_in1k`        | From scratch |    3.42    |   15.82   |   75.27   |   92.21   |  [config](repvgg-B0_8xb32_in1k.py)  | [model](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-B0_8xb32_in1k_20221213-5091ecc7.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-B0_8xb32_in1k_20221213-5091ecc7.log) |
| `repvgg-B1_8xb32_in1k`        | From scratch |   51.83    |   11.81   |   78.19   |   94.04   |  [config](repvgg-B1_8xb32_in1k.py)  | [model](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-B1_8xb32_in1k_20221213-d17c45e7.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-B1_8xb32_in1k_20221213-d17c45e7.log) |
| `repvgg-B1g2_8xb32_in1k`      | From scratch |   41.36    |   8.81    |   77.87   |   93.99   | [config](repvgg-B1g2_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-B1g2_8xb32_in1k_20221213-ae6428fd.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-B1g2_8xb32_in1k_20221213-ae6428fd.log) |
| `repvgg-B1g4_8xb32_in1k`      | From scratch |   36.13    |   7.30    |   77.81   |   93.77   | [config](repvgg-B1g4_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-B1g4_8xb32_in1k_20221213-a7a4aaea.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-B1g4_8xb32_in1k_20221213-a7a4aaea.log) |
| `repvgg-B2_8xb32_in1k`        | From scratch |   80.32    |   18.37   |   78.58   |   94.23   |  [config](repvgg-B2_8xb32_in1k.py)  | [model](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-B2_8xb32_in1k_20221213-d8b420ef.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-B2_8xb32_in1k_20221213-d8b420ef.log) |
| `repvgg-B2g4_8xb32_in1k`      | From scratch |   55.78    |   11.33   |   79.44   |   94.72   | [config](repvgg-B2g4_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-B2g4_8xb32_in1k_20221213-0c1990eb.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-B2g4_8xb32_in1k_20221213-0c1990eb.log) |
| `repvgg-B3_8xb32_in1k`        | From scratch |   110.96   |   26.21   |   80.58   |   95.33   |  [config](repvgg-B3_8xb32_in1k.py)  | [model](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-B3_8xb32_in1k_20221213-927a329a.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-B3_8xb32_in1k_20221213-927a329a.log) |
| `repvgg-B3g4_8xb32_in1k`      | From scratch |   75.63    |   16.06   |   80.26   |   95.15   | [config](repvgg-B3g4_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-B3g4_8xb32_in1k_20221213-e01cb280.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-B3g4_8xb32_in1k_20221213-e01cb280.log) |
| `repvgg-D2se_3rdparty_in1k`\* | From scratch |   120.39   |   32.84   |   81.81   |   95.94   | [config](repvgg-D2se_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-D2se_3rdparty_4xb64-autoaug-lbs-mixup-coslr-200e_in1k_20210909-cf3139b7.pth) |

*Models with * are converted from the [official repo](https://github.com/DingXiaoH/RepVGG/blob/9f272318abfc47a2b702cd0e916fca8d25d683e7/repvgg.py#L250). The config files of these models are only for inference. We haven't reproduce the training results.*

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
