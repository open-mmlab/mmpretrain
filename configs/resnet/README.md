# Deep Residual Learning for Image Recognition
<!-- {ResNet} -->

## Introduction

<!-- [ALGORITHM] -->

```latex
@inproceedings{he2016deep,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={770--778},
  year={2016}
}
```

## Results and models

## Cifar10

|         Model         | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) | Config | Download |
|:---------------------:|:---------:|:--------:|:---------:|:---------:|:---------:|:--------:|
| ResNet-18-b16x8 | 11.17 | 0.56 | 94.82 | | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet18_b16x8_cifar10.py) | [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_b16x8_cifar10_20210528-bd6371c8.pth) &#124; [log](https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_b16x8_cifar10_20210528-bd6371c8.log.json) |
| ResNet-34-b16x8 | 21.28 | 1.16 | 95.34 | | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet34_b16x8_cifar10.py) | [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnet34_b16x8_cifar10_20210528-a8aa36a6.pth) &#124; [log](https://download.openmmlab.com/mmclassification/v0/resnet/resnet34_b16x8_cifar10_20210528-a8aa36a6.log.json) |
| ResNet-50-b16x8 | 23.52 | 1.31 | 95.55 | | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet50_b16x8_cifar10.py) | [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_b16x8_cifar10_20210528-f54bfad9.pth) &#124; [log](https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_b16x8_cifar10_20210528-f54bfad9.log.json) |
| ResNet-101-b16x8 | 42.51 | 2.52 | 95.58 | | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet101_b16x8_cifar10.py) | [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnet101_b16x8_cifar10_20210528-2d29e936.pth) &#124; [log](https://download.openmmlab.com/mmclassification/v0/resnet/resnet101_b16x8_cifar10_20210528-2d29e936.log.json) |
| ResNet-152-b16x8 | 58.16 | 3.74 | 95.76 | | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet152_b16x8_cifar10.py) | [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnet152_b16x8_cifar10_20210528-3e8e9178.pth) &#124; [log](https://download.openmmlab.com/mmclassification/v0/resnet/resnet152_b16x8_cifar10_20210528-3e8e9178.log.json) |

## Cifar100

|         Model         | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) | Config | Download |
|:---------------------:|:---------:|:--------:|:---------:|:---------:|:---------:|:--------:|
| ResNet-50-b16x8 | 23.71 | 1.31 | 79.90 | 95.19 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet50_b16x8_cifar100.py) | [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_b16x8_cifar100_20210528-67b58a1b.pth) &#124; [log](https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_b16x8_cifar100_20210528-67b58a1b.log.json) |

### ImageNet

|         Model         | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) | Config | Download |
|:---------------------:|:---------:|:--------:|:---------:|:---------:|:---------:|:--------:|
| ResNet-18             | 11.69     | 1.82     | 69.90 | 89.43 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet18_b32x8_imagenet.py) | [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_8xb32_in1k_20210831-fbbb1da6.pth) &#124; [log](https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_8xb32_in1k_20210831-fbbb1da6.log.json) |
| ResNet-34             | 21.8      | 3.68     | 73.62 | 91.59 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet34_b32x8_imagenet.py) | [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnet34_8xb32_in1k_20210831-f257d4e6.pth) &#124; [log](https://download.openmmlab.com/mmclassification/v0/resnet/resnet34_8xb32_in1k_20210831-f257d4e6.log.json) |
| ResNet-50             | 25.56     | 4.12     | 76.55 | 93.06 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet50_b32x8_imagenet.py) | [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth) &#124; [log](https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.log.json) |
| ResNet-101            | 44.55     | 7.85     | 77.97 | 94.06 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet101_b32x8_imagenet.py) | [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnet101_8xb32_in1k_20210831-539c63f8.pth) &#124; [log](https://download.openmmlab.com/mmclassification/v0/resnet/resnet101_8xb32_in1k_20210831-539c63f8.log.json) |
| ResNet-152            | 60.19     | 11.58    | 78.48 | 94.13 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet152_b32x8_imagenet.py) | [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnet152_8xb32_in1k_20210901-4d7582fa.pth) &#124; [log](https://download.openmmlab.com/mmclassification/v0/resnet/resnet152_8xb32_in1k_20210901-4d7582fa.log.json) |
| ResNetV1D-50          | 25.58     | 4.36     | 77.54  | 93.57 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnetv1d50_b32x8_imagenet.py) | [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnetv1d50_b32x8_imagenet_20210531-db14775a.pth) &#124; [log](https://download.openmmlab.com/mmclassification/v0/resnet/resnetv1d50_b32x8_imagenet_20210531-db14775a.log.json) |
| ResNetV1D-101         | 44.57     | 8.09     | 78.93 | 94.48 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnetv1d101_b32x8_imagenet.py) | [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnetv1d101_b32x8_imagenet_20210531-6e13bcd3.pth) &#124; [log](https://download.openmmlab.com/mmclassification/v0/resnet/resnetv1d101_b32x8_imagenet_20210531-6e13bcd3.log.json) |
| ResNetV1D-152         | 60.21     | 11.82    | 79.41 | 94.70 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnetv1d152_b32x8_imagenet.py) | [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnetv1d152_b32x8_imagenet_20210531-278cf22a.pth) &#124; [log](https://download.openmmlab.com/mmclassification/v0/resnet/resnetv1d152_b32x8_imagenet_20210531-278cf22a.log.json) |
