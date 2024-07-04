# ConvNeXt

> [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545v1)

<!-- [ALGORITHM] -->

## Introduction

**ConvNeXt** is initially described in [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545v1), which is a pure convolutional model (ConvNet), inspired by the design of Vision Transformers. The ConvNeXt has the pyramid structure and achieve competitive  performance on various vision tasks, with simplicity and efficiency.

<div align=center>
<img src="https://user-images.githubusercontent.com/8370623/148624004-e9581042-ea4d-4e10-b3bd-42c92b02053b.png" width="100%"/>
</div>

## Abstract

<details>

<summary>Show the paper's abstract</summary>

<br>
The "Roaring 20s" of visual recognition began with the introduction of Vision Transformers (ViTs), which quickly superseded ConvNets as the state-of-the-art image classification model. A vanilla ViT, on the other hand, faces difficulties when applied to general computer vision tasks such as object detection and semantic segmentation. It is the hierarchical Transformers (e.g., Swin Transformers) that reintroduced several ConvNet priors, making Transformers practically viable as a generic vision backbone and demonstrating remarkable performance on a wide variety of vision tasks. However, the effectiveness of such hybrid approaches is still largely credited to the intrinsic superiority of Transformers, rather than the inherent inductive biases of convolutions. In this work, we reexamine the design spaces and test the limits of what a pure ConvNet can achieve. We gradually "modernize" a standard ResNet toward the design of a vision Transformer, and discover several key components that contribute to the performance difference along the way. The outcome of this exploration is a family of pure ConvNet models dubbed ConvNeXt. Constructed entirely from standard ConvNet modules, ConvNeXts compete favorably with Transformers in terms of accuracy and scalability, achieving 87.8% ImageNet top-1 accuracy and outperforming Swin Transformers on COCO detection and ADE20K segmentation, while maintaining the simplicity and efficiency of standard ConvNets.
</br>

</details>

## How to use it?

<!-- [TABS-BEGIN] -->

**Predict image**

```python
from mmpretrain import inference_model

predict = inference_model('convnext-tiny_32xb128_in1k', 'demo/bird.JPEG')
print(predict['pred_class'])
print(predict['pred_score'])
```

**Use the model**

```python
import torch
from mmpretrain import get_model

model = get_model('convnext-tiny_32xb128_in1k', pretrained=True)
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
python tools/train.py configs/convnext/convnext-tiny_32xb128_in1k.py
```

Test:

```shell
python tools/test.py configs/convnext/convnext-tiny_32xb128_in1k.py https://download.openmmlab.com/mmclassification/v0/convnext/convnext-tiny_32xb128_in1k_20221207-998cf3e9.pth
```

<!-- [TABS-END] -->

## Models and results

### Pretrained models

| Model                              | Params (M) | Flops (G) |                  Config                   |                                                  Download                                                  |
| :--------------------------------- | :--------: | :-------: | :---------------------------------------: | :--------------------------------------------------------------------------------------------------------: |
| `convnext-base_3rdparty_in21k`\*   |   88.59    |   15.36   | [config](convnext-base_32xb128_in21k.py)  | [model](https://download.openmmlab.com/mmclassification/v0/convnext/convnext-base_3rdparty_in21k_20220124-13b83eec.pth) |
| `convnext-large_3rdparty_in21k`\*  |   197.77   |   34.37   | [config](convnext-large_64xb64_in21k.py)  | [model](https://download.openmmlab.com/mmclassification/v0/convnext/convnext-large_3rdparty_in21k_20220124-41b5a79f.pth) |
| `convnext-xlarge_3rdparty_in21k`\* |   350.20   |   60.93   | [config](convnext-xlarge_64xb64_in21k.py) | [model](https://download.openmmlab.com/mmclassification/v0/convnext/convnext-xlarge_3rdparty_in21k_20220124-f909bad7.pth) |

*Models with * are converted from the [official repo](https://github.com/facebookresearch/ConvNeXt). The config files of these models are only for inference. We haven't reproduce the training results.*

### Image Classification on ImageNet-1k

| Model                                             |   Pretrain   | Params (M) | Flops (G) | Top-1 (%) | Top-5 (%) |                     Config                     |                         Download                         |
| :------------------------------------------------ | :----------: | :--------: | :-------: | :-------: | :-------: | :--------------------------------------------: | :------------------------------------------------------: |
| `convnext-tiny_32xb128_in1k`                      | From scratch |   28.59    |   4.46    |   82.14   |   96.06   |    [config](convnext-tiny_32xb128_in1k.py)     | [model](https://download.openmmlab.com/mmclassification/v0/convnext/convnext-tiny_32xb128_in1k_20221207-998cf3e9.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/convnext/convnext-tiny_32xb128_in1k_20221207-998cf3e9.json) |
| `convnext-tiny_32xb128-noema_in1k`                | From scratch |   28.59    |   4.46    |   81.95   |   95.89   |    [config](convnext-tiny_32xb128_in1k.py)     | [model](https://download.openmmlab.com/mmclassification/v0/convnext/convnext-tiny_32xb128-noema_in1k_20221208-5d4509c7.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/convnext/convnext-tiny_32xb128_in1k_20221207-998cf3e9.json) |
| `convnext-tiny_in21k-pre_3rdparty_in1k`\*         | ImageNet-21k |   28.59    |   4.46    |   82.90   |   96.62   |    [config](convnext-tiny_32xb128_in1k.py)     | [model](https://download.openmmlab.com/mmclassification/v0/convnext/convnext-tiny_in21k-pre_3rdparty_in1k_20221219-7501e534.pth) |
| `convnext-tiny_in21k-pre_3rdparty_in1k-384px`\*   | ImageNet-21k |   28.59    |   13.14   |   84.11   |   97.14   | [config](convnext-tiny_32xb128_in1k-384px.py)  | [model](https://download.openmmlab.com/mmclassification/v0/convnext/convnext-tiny_in21k-pre_3rdparty_in1k-384px_20221219-c1182362.pth) |
| `convnext-small_32xb128_in1k`                     | From scratch |   50.22    |   8.69    |   83.16   |   96.56   |    [config](convnext-small_32xb128_in1k.py)    | [model](https://download.openmmlab.com/mmclassification/v0/convnext/convnext-small_32xb128_in1k_20221207-4ab7052c.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/convnext/convnext-small_32xb128_in1k_20221207-4ab7052c.json) |
| `convnext-small_32xb128-noema_in1k`               | From scratch |   50.22    |   8.69    |   83.21   |   96.48   |    [config](convnext-small_32xb128_in1k.py)    | [model](https://download.openmmlab.com/mmclassification/v0/convnext/convnext-small_32xb128-noema_in1k_20221208-4a618995.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/convnext/convnext-small_32xb128_in1k_20221207-4ab7052c.json) |
| `convnext-small_in21k-pre_3rdparty_in1k`\*        | ImageNet-21k |   50.22    |   8.69    |   84.59   |   97.41   |    [config](convnext-small_32xb128_in1k.py)    | [model](https://download.openmmlab.com/mmclassification/v0/convnext/convnext-small_in21k-pre_3rdparty_in1k_20221219-aeca4c93.pth) |
| `convnext-small_in21k-pre_3rdparty_in1k-384px`\*  | ImageNet-21k |   50.22    |   25.58   |   85.75   |   97.88   | [config](convnext-small_32xb128_in1k-384px.py) | [model](https://download.openmmlab.com/mmclassification/v0/convnext/convnext-small_in21k-pre_3rdparty_in1k-384px_20221219-96f0bb87.pth) |
| `convnext-base_32xb128_in1k`                      | From scratch |   88.59    |   15.36   |   83.66   |   96.74   |    [config](convnext-base_32xb128_in1k.py)     | [model](https://download.openmmlab.com/mmclassification/v0/convnext/convnext-base_32xb128_in1k_20221207-fbdb5eb9.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/convnext/convnext-base_32xb128_in1k_20221207-fbdb5eb9.json) |
| `convnext-base_32xb128-noema_in1k`                | From scratch |   88.59    |   15.36   |   83.64   |   96.61   |    [config](convnext-base_32xb128_in1k.py)     | [model](https://download.openmmlab.com/mmclassification/v0/convnext/convnext-base_32xb128-noema_in1k_20221208-f8182678.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/convnext/convnext-base_32xb128_in1k_20221207-fbdb5eb9.json) |
| `convnext-base_3rdparty_in1k`\*                   | From scratch |   88.59    |   15.36   |   83.85   |   96.74   |    [config](convnext-base_32xb128_in1k.py)     | [model](https://download.openmmlab.com/mmclassification/v0/convnext/convnext-base_3rdparty_32xb128_in1k_20220124-d0915162.pth) |
| `convnext-base_3rdparty-noema_in1k`\*             | From scratch |   88.59    |   15.36   |   83.71   |   96.60   |    [config](convnext-base_32xb128_in1k.py)     | [model](https://download.openmmlab.com/mmclassification/v0/convnext/convnext-base_3rdparty_32xb128-noema_in1k_20220222-dba4f95f.pth) |
| `convnext-base_3rdparty_in1k-384px`\*             | From scratch |   88.59    |   45.21   |   85.10   |   97.34   | [config](convnext-base_32xb128_in1k-384px.py)  | [model](https://download.openmmlab.com/mmclassification/v0/convnext/convnext-base_3rdparty_in1k-384px_20221219-c8f1dc2b.pth) |
| `convnext-base_in21k-pre_3rdparty_in1k`\*         | ImageNet-21k |   88.59    |   15.36   |   85.81   |   97.86   |    [config](convnext-base_32xb128_in1k.py)     | [model](https://download.openmmlab.com/mmclassification/v0/convnext/convnext-base_in21k-pre-3rdparty_32xb128_in1k_20220124-eb2d6ada.pth) |
| `convnext-base_in21k-pre-3rdparty_in1k-384px`\*   | From scratch |   88.59    |   45.21   |   86.82   |   98.25   | [config](convnext-base_32xb128_in1k-384px.py)  | [model](https://download.openmmlab.com/mmclassification/v0/convnext/convnext-base_in21k-pre-3rdparty_in1k-384px_20221219-4570f792.pth) |
| `convnext-large_3rdparty_in1k`\*                  | From scratch |   197.77   |   34.37   |   84.30   |   96.89   |    [config](convnext-large_64xb64_in1k.py)     | [model](https://download.openmmlab.com/mmclassification/v0/convnext/convnext-large_3rdparty_64xb64_in1k_20220124-f8a0ded0.pth) |
| `convnext-large_3rdparty_in1k-384px`\*            | From scratch |   197.77   |  101.10   |   85.50   |   97.59   | [config](convnext-large_64xb64_in1k-384px.py)  | [model](https://download.openmmlab.com/mmclassification/v0/convnext/convnext-large_3rdparty_in1k-384px_20221219-6dd29d10.pth) |
| `convnext-large_in21k-pre_3rdparty_in1k`\*        | ImageNet-21k |   197.77   |   34.37   |   86.61   |   98.04   |    [config](convnext-large_64xb64_in1k.py)     | [model](https://download.openmmlab.com/mmclassification/v0/convnext/convnext-large_in21k-pre-3rdparty_64xb64_in1k_20220124-2412403d.pth) |
| `convnext-large_in21k-pre-3rdparty_in1k-384px`\*  | From scratch |   197.77   |  101.10   |   87.46   |   98.37   | [config](convnext-large_64xb64_in1k-384px.py)  | [model](https://download.openmmlab.com/mmclassification/v0/convnext/convnext-large_in21k-pre-3rdparty_in1k-384px_20221219-6d38dd66.pth) |
| `convnext-xlarge_in21k-pre_3rdparty_in1k`\*       | ImageNet-21k |   350.20   |   60.93   |   86.97   |   98.20   |    [config](convnext-xlarge_64xb64_in1k.py)    | [model](https://download.openmmlab.com/mmclassification/v0/convnext/convnext-xlarge_in21k-pre-3rdparty_64xb64_in1k_20220124-76b6863d.pth) |
| `convnext-xlarge_in21k-pre-3rdparty_in1k-384px`\* | From scratch |   350.20   |  179.20   |   87.76   |   98.55   | [config](convnext-xlarge_64xb64_in1k-384px.py) | [model](https://download.openmmlab.com/mmclassification/v0/convnext/convnext-xlarge_in21k-pre-3rdparty_in1k-384px_20221219-b161bc14.pth) |

*Models with * are converted from the [official repo](https://github.com/facebookresearch/ConvNeXt). The config files of these models are only for inference. We haven't reproduce the training results.*

## Citation

```bibtex
@Article{liu2022convnet,
  author  = {Zhuang Liu and Hanzi Mao and Chao-Yuan Wu and Christoph Feichtenhofer and Trevor Darrell and Saining Xie},
  title   = {A ConvNet for the 2020s},
  journal = {arXiv preprint arXiv:2201.03545},
  year    = {2022},
}
```
