# ResNet

> [Deep Residual Learning for Image Recognition](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)

<!-- [ALGORITHM] -->

## Introduction

**Residual Networks**, or **ResNets**, learn residual functions with reference to the layer inputs, instead of
learning unreferenced functions. In the mainstream previous works, like VGG, the neural networks are a stack
of layers and every layer attempts to fit a desired underlying mapping. In ResNets, a few stacked layers are
grouped as a block, and the layers in a block attempts to learn a residual mapping.

Formally, denoting the desired underlying mapping of a block as $\mathcal{H}(x)$, split the underlying mapping
into the sum of the identity and the residual mapping as $\mathcal{H}(x) = x + \mathcal{F}(x)$, and let the
stacked non-linear layers fit the residual mapping $\mathcal{F}(x)$.

Many works proved this method makes deep neural networks easier to optimize, and can gain accuracy from
considerably increased depth. Recently, the residual structure is widely used in various models.

<div align=center>
<img src="https://user-images.githubusercontent.com/26739999/142574068-60cfdeea-c4ec-4c49-abb2-5dc2facafc3b.png" width="40%"/>
</div>

## Abstract

<details>

<summary>Show the paper's abstract</summary>

<br>
Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions. We provide comprehensive empirical evidence showing that these residual networks are easier to optimize, and can gain accuracy from considerably increased depth. On the ImageNet dataset we evaluate residual nets with a depth of up to 152 layers---8x deeper than VGG nets but still having lower complexity. An ensemble of these residual nets achieves 3.57% error on the ImageNet test set. This result won the 1st place on the ILSVRC 2015 classification task. We also present analysis on CIFAR-10 with 100 and 1000 layers.

The depth of representations is of central importance for many visual recognition tasks. Solely due to our extremely deep representations, we obtain a 28% relative improvement on the COCO object detection dataset. Deep residual nets are foundations of our submissions to ILSVRC & COCO 2015 competitions, where we also won the 1st places on the tasks of ImageNet detection, ImageNet localization, COCO detection, and COCO segmentation.
</br>

</details>

## How to use it?

<!-- [TABS-BEGIN] -->

**Predict image**

```python
from mmpretrain import inference_model

predict = inference_model('resnet18_8xb16_cifar10', 'demo/bird.JPEG')
print(predict['pred_class'])
print(predict['pred_score'])
```

**Use the model**

```python
import torch
from mmpretrain import get_model

model = get_model('resnet18_8xb16_cifar10', pretrained=True)
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
python tools/train.py configs/resnet/resnet18_8xb16_cifar10.py
```

Test:

```shell
python tools/test.py configs/resnet/resnet18_8xb16_cifar10.py https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_b16x8_cifar10_20210528-bd6371c8.pth
```

<!-- [TABS-END] -->

## Models and results

### Image Classification on ImageNet-1k

| Model                              |   Pretrain   | Params (M) | Flops (G) | Top-1 (%) | Top-5 (%) |                    Config                     |                                 Download                                 |
| :--------------------------------- | :----------: | :--------: | :-------: | :-------: | :-------: | :-------------------------------------------: | :----------------------------------------------------------------------: |
| `resnet18_8xb32_in1k`              | From scratch |   11.69    |   1.82    |   69.90   |   89.43   |       [config](resnet18_8xb32_in1k.py)        | [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_8xb32_in1k_20210831-fbbb1da6.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_8xb32_in1k_20210831-fbbb1da6.json) |
| `resnet34_8xb32_in1k`              | From scratch |    2.18    |   3.68    |   73.62   |   91.59   |       [config](resnet34_8xb32_in1k.py)        | [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnet34_8xb32_in1k_20210831-f257d4e6.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/resnet/resnet34_8xb32_in1k_20210831-f257d4e6.json) |
| `resnet50_8xb32_in1k`              | From scratch |   25.56    |   4.12    |   76.55   |   93.06   |       [config](resnet50_8xb32_in1k.py)        | [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.json) |
| `resnet101_8xb32_in1k`             | From scratch |   44.55    |   7.85    |   77.97   |   94.06   |       [config](resnet101_8xb32_in1k.py)       | [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnet101_8xb32_in1k_20210831-539c63f8.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/resnet/resnet101_8xb32_in1k_20210831-539c63f8.json) |
| `resnet152_8xb32_in1k`             | From scratch |   60.19    |   11.58   |   78.48   |   94.13   |       [config](resnet152_8xb32_in1k.py)       | [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnet152_8xb32_in1k_20210901-4d7582fa.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/resnet/resnet152_8xb32_in1k_20210901-4d7582fa.json) |
| `resnetv1d50_8xb32_in1k`           | From scratch |   25.58    |   4.36    |   77.54   |   93.57   |      [config](resnetv1d50_8xb32_in1k.py)      | [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnetv1d50_b32x8_imagenet_20210531-db14775a.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/resnet/resnetv1d50_b32x8_imagenet_20210531-db14775a.json) |
| `resnetv1d101_8xb32_in1k`          | From scratch |   44.57    |   8.09    |   78.93   |   94.48   |     [config](resnetv1d101_8xb32_in1k.py)      | [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnetv1d101_b32x8_imagenet_20210531-6e13bcd3.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/resnet/resnetv1d101_b32x8_imagenet_20210531-6e13bcd3.json) |
| `resnetv1d152_8xb32_in1k`          | From scratch |   60.21    |   11.82   |   79.41   |   94.70   |     [config](resnetv1d152_8xb32_in1k.py)      | [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnetv1d152_b32x8_imagenet_20210531-278cf22a.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/resnet/resnetv1d152_b32x8_imagenet_20210531-278cf22a.json) |
| `resnet50_8xb32-fp16_in1k`         | From scratch |   25.56    |   4.12    |   76.30   |   93.07   |     [config](resnet50_8xb32-fp16_in1k.py)     | [model](https://download.openmmlab.com/mmclassification/v0/fp16/resnet50_batch256_fp16_imagenet_20210320-b3964210.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/fp16/resnet50_batch256_fp16_imagenet_20210320-b3964210.json) |
| `resnet50_8xb256-rsb-a1-600e_in1k` | From scratch |   25.56    |   4.12    |   80.12   |   94.78   | [config](resnet50_8xb256-rsb-a1-600e_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb256-rsb-a1-600e_in1k_20211228-20e21305.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb256-rsb-a1-600e_in1k_20211228-20e21305.json) |
| `resnet50_8xb256-rsb-a2-300e_in1k` | From scratch |   25.56    |   4.12    |   79.55   |   94.37   | [config](resnet50_8xb256-rsb-a2-300e_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb256-rsb-a2-300e_in1k_20211228-0fd8be6e.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb256-rsb-a2-300e_in1k_20211228-0fd8be6e.json) |
| `resnet50_8xb256-rsb-a3-100e_in1k` | From scratch |   25.56    |   4.12    |   78.30   |   93.80   | [config](resnet50_8xb256-rsb-a3-100e_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb256-rsb-a3-100e_in1k_20211228-3493673c.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb256-rsb-a3-100e_in1k_20211228-3493673c.json) |
| `resnetv1c50_8xb32_in1k`           | From scratch |   25.58    |   4.36    |   77.01   |   93.58   |      [config](resnetv1c50_8xb32_in1k.py)      | [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnetv1c50_8xb32_in1k_20220214-3343eccd.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/resnet/resnetv1c50_8xb32_in1k_20220214-3343eccd.json) |
| `resnetv1c101_8xb32_in1k`          | From scratch |   44.57    |   8.09    |   78.30   |   94.27   |     [config](resnetv1c101_8xb32_in1k.py)      | [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnetv1c101_8xb32_in1k_20220214-434fe45f.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/resnet/resnetv1c101_8xb32_in1k_20220214-434fe45f.json) |
| `resnetv1c152_8xb32_in1k`          | From scratch |   60.21    |   11.82   |   78.76   |   94.41   |     [config](resnetv1c152_8xb32_in1k.py)      | [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnetv1c152_8xb32_in1k_20220214-c013291f.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/resnet/resnetv1c152_8xb32_in1k_20220214-c013291f.json) |

### Image Classification on CIFAR-10

| Model                     |   Pretrain   | Params (M) | Flops (G) | Top-1 (%) |                Config                |                                              Download                                               |
| :------------------------ | :----------: | :--------: | :-------: | :-------: | :----------------------------------: | :-------------------------------------------------------------------------------------------------: |
| `resnet18_8xb16_cifar10`  | From scratch |   11.17    |   0.56    |   94.82   | [config](resnet18_8xb16_cifar10.py)  | [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_b16x8_cifar10_20210528-bd6371c8.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_b16x8_cifar10_20210528-bd6371c8.json) |
| `resnet34_8xb16_cifar10`  | From scratch |   21.28    |   1.16    |   95.34   | [config](resnet34_8xb16_cifar10.py)  | [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnet34_b16x8_cifar10_20210528-a8aa36a6.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/resnet/resnet34_b16x8_cifar10_20210528-a8aa36a6.json) |
| `resnet50_8xb16_cifar10`  | From scratch |   23.52    |   1.31    |   95.55   | [config](resnet50_8xb16_cifar10.py)  | [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_b16x8_cifar10_20210528-f54bfad9.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_b16x8_cifar10_20210528-f54bfad9.json) |
| `resnet101_8xb16_cifar10` | From scratch |   42.51    |   2.52    |   95.58   | [config](resnet101_8xb16_cifar10.py) | [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnet101_b16x8_cifar10_20210528-2d29e936.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/resnet/resnet101_b16x8_cifar10_20210528-2d29e936.json) |
| `resnet152_8xb16_cifar10` | From scratch |   58.16    |   3.74    |   95.76   | [config](resnet152_8xb16_cifar10.py) | [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnet152_b16x8_cifar10_20210528-3e8e9178.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/resnet/resnet152_b16x8_cifar10_20210528-3e8e9178.json) |

### Image Classification on CIFAR-100

| Model                     |   Pretrain   | Params (M) | Flops (G) | Top-1 (%) | Top-5 (%) |                Config                |                                          Download                                          |
| :------------------------ | :----------: | :--------: | :-------: | :-------: | :-------: | :----------------------------------: | :----------------------------------------------------------------------------------------: |
| `resnet50_8xb16_cifar100` | From scratch |   23.71    |   1.31    |   79.90   |   95.19   | [config](resnet50_8xb16_cifar100.py) | [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_b16x8_cifar100_20210528-67b58a1b.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_b16x8_cifar100_20210528-67b58a1b.json) |

### Image Classification on CUB-200-2011

| Model               |   Pretrain   | Params (M) | Flops (G) | Top-1 (%) |             Config             |                                                    Download                                                     |
| :------------------ | :----------: | :--------: | :-------: | :-------: | :----------------------------: | :-------------------------------------------------------------------------------------------------------------: |
| `resnet50_8xb8_cub` | From scratch |   23.92    |   16.48   |   88.45   | [config](resnet50_8xb8_cub.py) | [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb8_cub_20220307-57840e60.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb8_cub_20220307-57840e60.json) |

## Citation

```bibtex
@inproceedings{he2016deep,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={770--778},
  year={2016}
}
```
