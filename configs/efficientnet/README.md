# EfficientNet

> [Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946v5)
<!-- [ALGORITHM] -->

## Abstract

Convolutional Neural Networks (ConvNets) are commonly developed at a fixed resource budget, and then scaled up for better accuracy if more resources are available. In this paper, we systematically study model scaling and identify that carefully balancing network depth, width, and resolution can lead to better performance. Based on this observation, we propose a new scaling method that uniformly scales all dimensions of depth/width/resolution using a simple yet highly effective compound coefficient. We demonstrate the effectiveness of this method on scaling up MobileNets and ResNet.   To go even further, we use neural architecture search to design a new baseline network and scale it up to obtain a family of models, called EfficientNets, which achieve much better accuracy and efficiency than previous ConvNets. In particular, our EfficientNet-B7 achieves state-of-the-art 84.3% top-1 accuracy on ImageNet, while being 8.4x smaller and 6.1x faster on inference than the best existing ConvNet. Our EfficientNets also transfer well and achieve state-of-the-art accuracy on CIFAR-100 (91.7%), Flowers (98.8%), and 3 other transfer learning datasets, with an order of magnitude fewer parameters.

<div align=center>
<img src="https://user-images.githubusercontent.com/26739999/150078232-d28c91fc-d0e8-43e3-9d20-b5162f0fb463.png" width="60%"/>
</div>

## Results and models

### ImageNet-1k

In the result table, AA means trained with AutoAugment pre-processing, more details can be found in the [paper](https://arxiv.org/abs/1805.09501), and AdvProp is a method to train with adversarial examples, more details can be found in the [paper](https://arxiv.org/abs/1911.09665).

Note: In MMClassification, we support training with AutoAugment, don't support AdvProp by now.

|         Model         | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) | Config | Download |
|:---------------------:|:---------:|:--------:|:---------:|:---------:|:------:|:--------:|
| EfficientNet-B0\* | 5.29 | 0.02 | 76.74 | 93.17 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet-b0_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b0_3rdparty_8xb32_in1k_20220119-a7e2a0b1.pth) |
| EfficientNet-B0 (AA)\* | 5.29 | 0.02 | 77.26 | 93.41 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet-b0_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b0_3rdparty_8xb32-aa_in1k_20220119-8d939117.pth) |
| EfficientNet-B0 (AA + AdvProp)\* | 5.29 | 0.02 | 77.53 | 93.61 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet-b0_8xb32-01norm_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b0_3rdparty_8xb32-aa-advprop_in1k_20220119-26434485.pth) |
| EfficientNet-B1\* | 7.79 | 0.03 | 78.68 | 94.28 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet-b1_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b1_3rdparty_8xb32_in1k_20220119-002556d9.pth) |
| EfficientNet-B1 (AA)\* | 7.79 | 0.03 | 79.20 | 94.42 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet-b1_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b1_3rdparty_8xb32-aa_in1k_20220119-619d8ae3.pth) |
| EfficientNet-B1 (AA + AdvProp)\* | 7.79 | 0.03 | 79.52 | 94.43 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet-b1_8xb32-01norm_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b1_3rdparty_8xb32-aa-advprop_in1k_20220119-5715267d.pth) |
| EfficientNet-B2\* | 9.11 | 0.03 | 79.64 | 94.80 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet-b2_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b2_3rdparty_8xb32_in1k_20220119-ea374a30.pth) |
| EfficientNet-B2 (AA)\* | 9.11 | 0.03 | 80.21 | 94.96 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet-b2_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b2_3rdparty_8xb32-aa_in1k_20220119-dd61e80b.pth) |
| EfficientNet-B2 (AA + AdvProp)\* | 9.11 | 0.03 | 80.45 | 95.07 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet-b2_8xb32-01norm_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b2_3rdparty_8xb32-aa-advprop_in1k_20220119-1655338a.pth) |
| EfficientNet-B3\* | 12.23 | 0.06 | 81.01 | 95.34 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet-b3_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b3_3rdparty_8xb32_in1k_20220119-4b4d7487.pth) |
| EfficientNet-B3 (AA)\* | 12.23 | 0.06 | 81.58 | 95.67 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet-b3_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b3_3rdparty_8xb32-aa_in1k_20220119-5b4887a0.pth) |
| EfficientNet-B3 (AA + AdvProp)\* | 12.23 | 0.06 | 81.81 | 95.69 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet-b3_8xb32-01norm_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b3_3rdparty_8xb32-aa-advprop_in1k_20220119-53b41118.pth) |
| EfficientNet-B4\* | 19.34 | 0.12 | 82.57 | 96.09 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet-b4_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b4_3rdparty_8xb32_in1k_20220119-81fd4077.pth) |
| EfficientNet-B4 (AA)\* | 19.34 | 0.12 | 82.95 | 96.26 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet-b4_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b4_3rdparty_8xb32-aa_in1k_20220119-45b8bd2b.pth) |
| EfficientNet-B4 (AA + AdvProp)\* | 19.34 | 0.12 | 83.25 | 96.44 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet-b4_8xb32-01norm_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b4_3rdparty_8xb32-aa-advprop_in1k_20220119-38c2238c.pth) |
| EfficientNet-B5\* | 30.39 | 0.24 | 83.18 | 96.47 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet-b5_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b5_3rdparty_8xb32_in1k_20220119-e9814430.pth) |
| EfficientNet-B5 (AA)\* | 30.39 | 0.24 | 83.82 | 96.76 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet-b5_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b5_3rdparty_8xb32-aa_in1k_20220119-2cab8b78.pth) |
| EfficientNet-B5 (AA + AdvProp)\* | 30.39 | 0.24 | 84.21 | 96.98 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet-b5_8xb32-01norm_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b5_3rdparty_8xb32-aa-advprop_in1k_20220119-f57a895a.pth) |
| EfficientNet-B6 (AA)\* | 43.04 | 0.41 | 84.05 | 96.82 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet-b6_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b6_3rdparty_8xb32-aa_in1k_20220119-45b03310.pth) |
| EfficientNet-B6 (AA + AdvProp)\* | 43.04 | 0.41 | 84.74 | 97.14 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet-b6_8xb32-01norm_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b6_3rdparty_8xb32-aa-advprop_in1k_20220119-bfe3485e.pth) |
| EfficientNet-B7 (AA)\* | 66.35 | 0.72 | 84.38 | 96.88 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet-b7_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b7_3rdparty_8xb32-aa_in1k_20220119-bf03951c.pth) |
| EfficientNet-B7 (AA + AdvProp)\* | 66.35 | 0.72 | 85.14 | 97.23 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet-b7_8xb32-01norm_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b7_3rdparty_8xb32-aa-advprop_in1k_20220119-c6dbff10.pth) |
| EfficientNet-B8 (AA + AdvProp)\* | 87.41 | 1.09 | 85.38 | 97.28 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet-b8_8xb32-01norm_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b8_3rdparty_8xb32-aa-advprop_in1k_20220119-297ce1b7.pth) |

*Models with \* are converted from the [official repo](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet). The config files of these models are only for inference. We don't ensure these config files' training accuracy and welcome you to contribute your reproduction results.*

## Citation

```
@inproceedings{tan2019efficientnet,
  title={Efficientnet: Rethinking model scaling for convolutional neural networks},
  author={Tan, Mingxing and Le, Quoc},
  booktitle={International Conference on Machine Learning},
  pages={6105--6114},
  year={2019},
  organization={PMLR}
}
```
