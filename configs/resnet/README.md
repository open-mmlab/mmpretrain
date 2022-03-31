# ResNet

> [Deep Residual Learning for Image Recognition](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)
<!-- [ALGORITHM] -->

## Abstract

Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions. We provide comprehensive empirical evidence showing that these residual networks are easier to optimize, and can gain accuracy from considerably increased depth. On the ImageNet dataset we evaluate residual nets with a depth of up to 152 layers---8x deeper than VGG nets but still having lower complexity. An ensemble of these residual nets achieves 3.57% error on the ImageNet test set. This result won the 1st place on the ILSVRC 2015 classification task. We also present analysis on CIFAR-10 with 100 and 1000 layers.

The depth of representations is of central importance for many visual recognition tasks. Solely due to our extremely deep representations, we obtain a 28% relative improvement on the COCO object detection dataset. Deep residual nets are foundations of our submissions to ILSVRC & COCO 2015 competitions, where we also won the 1st places on the tasks of ImageNet detection, ImageNet localization, COCO detection, and COCO segmentation.

<div align=center>
<img src="https://user-images.githubusercontent.com/26739999/142574068-60cfdeea-c4ec-4c49-abb2-5dc2facafc3b.png" width="40%"/>
</div>

## Results and models

The pre-trained models on ImageNet-21k are used to fine-tune, and therefore don't have evaluation results.

|   Model         | resolution  | Params(M) |  Flops(G) | Download |
|:---------------:|:-----------:|:---------:|:---------:|:--------:|
| ResNet-50-mill  |   224x224   |   86.74   |   15.14   | [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_3rdparty-mill_in21k_20220331-faac000b.pth)|

*The "mill" means using the mutil-label pretrain weight from [ImageNet-21K Pretraining for the Masses](https://github.com/Alibaba-MIIL/ImageNet21K).*

### Cifar10

|         Model         | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) | Config | Download |
|:---------------------:|:---------:|:--------:|:---------:|:---------:|:---------:|:--------:|
| ResNet-18 | 11.17 | 0.56 | 94.82 | 99.87 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet18_8xb16_cifar10.py) | [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_b16x8_cifar10_20210528-bd6371c8.pth) &#124; [log](https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_b16x8_cifar10_20210528-bd6371c8.log.json) |
| ResNet-34 | 21.28 | 1.16 | 95.34 | 99.87 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet34_8xb16_cifar10.py) | [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnet34_b16x8_cifar10_20210528-a8aa36a6.pth) &#124; [log](https://download.openmmlab.com/mmclassification/v0/resnet/resnet34_b16x8_cifar10_20210528-a8aa36a6.log.json) |
| ResNet-50 | 23.52 | 1.31 | 95.55 | 99.91 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet50_8xb16_cifar10.py) | [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_b16x8_cifar10_20210528-f54bfad9.pth) &#124; [log](https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_b16x8_cifar10_20210528-f54bfad9.log.json) |
| ResNet-101 | 42.51 | 2.52 | 95.58 | 99.87 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet101_8xb16_cifar10.py) | [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnet101_b16x8_cifar10_20210528-2d29e936.pth) &#124; [log](https://download.openmmlab.com/mmclassification/v0/resnet/resnet101_b16x8_cifar10_20210528-2d29e936.log.json) |
| ResNet-152 | 58.16 | 3.74 | 95.76 | 99.89 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet152_8xb16_cifar10.py) | [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnet152_b16x8_cifar10_20210528-3e8e9178.pth) &#124; [log](https://download.openmmlab.com/mmclassification/v0/resnet/resnet152_b16x8_cifar10_20210528-3e8e9178.log.json) |

### Cifar100

|         Model         | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) | Config | Download |
|:---------------------:|:---------:|:--------:|:---------:|:---------:|:---------:|:--------:|
| ResNet-50 | 23.71 | 1.31 | 79.90 | 95.19 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet50_8xb16_cifar100.py) | [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_b16x8_cifar100_20210528-67b58a1b.pth) &#124; [log](https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_b16x8_cifar100_20210528-67b58a1b.log.json) |

### ImageNet-1k

|         Model         | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) | Config | Download |
|:---------------------:|:---------:|:--------:|:---------:|:---------:|:---------:|:--------:|
| ResNet-18             | 11.69     | 1.82     | 69.90 | 89.43 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet18_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_8xb32_in1k_20210831-fbbb1da6.pth) &#124; [log](https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_8xb32_in1k_20210831-fbbb1da6.log.json) |
| ResNet-34             | 21.8      | 3.68     | 73.62 | 91.59 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet34_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnet34_8xb32_in1k_20210831-f257d4e6.pth) &#124; [log](https://download.openmmlab.com/mmclassification/v0/resnet/resnet34_8xb32_in1k_20210831-f257d4e6.log.json) |
| ResNet-50             | 25.56     | 4.12     | 76.55 | 93.06 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet50_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth) &#124; [log](https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.log.json) |
| ResNet-101            | 44.55     | 7.85     | 77.97 | 94.06 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet101_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnet101_8xb32_in1k_20210831-539c63f8.pth) &#124; [log](https://download.openmmlab.com/mmclassification/v0/resnet/resnet101_8xb32_in1k_20210831-539c63f8.log.json) |
| ResNet-152            | 60.19     | 11.58    | 78.48 | 94.13 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet152_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnet152_8xb32_in1k_20210901-4d7582fa.pth) &#124; [log](https://download.openmmlab.com/mmclassification/v0/resnet/resnet152_8xb32_in1k_20210901-4d7582fa.log.json) |
| ResNetV1C-50          | 25.58     | 4.36     | 77.01 | 93.58 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnetv1c50_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnetv1c50_8xb32_in1k_20220214-3343eccd.pth) &#124; [log](https://download.openmmlab.com/mmclassification/v0/resnet/resnetv1c50_8xb32_in1k_20220214-3343eccd.log.json) |
| ResNetV1C-101         | 44.57     | 8.09     | 78.30 | 94.27 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnetv1c101_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnetv1c101_8xb32_in1k_20220214-434fe45f.pth) &#124; [log](https://download.openmmlab.com/mmclassification/v0/resnet/resnetv1c101_8xb32_in1k_20220214-434fe45f.log.json) |
| ResNetV1C-152         | 60.21     | 11.82    | 78.76 | 94.41 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnetv1c152_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnetv1c152_8xb32_in1k_20220214-c013291f.pth) &#124; [log](https://download.openmmlab.com/mmclassification/v0/resnet/resnetv1c152_8xb32_in1k_20220214-c013291f.log.json) |
| ResNetV1D-50          | 25.58     | 4.36     | 77.54  | 93.57 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnetv1d50_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnetv1d50_b32x8_imagenet_20210531-db14775a.pth) &#124; [log](https://download.openmmlab.com/mmclassification/v0/resnet/resnetv1d50_b32x8_imagenet_20210531-db14775a.log.json) |
| ResNetV1D-101         | 44.57     | 8.09     | 78.93 | 94.48 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnetv1d101_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnetv1d101_b32x8_imagenet_20210531-6e13bcd3.pth) &#124; [log](https://download.openmmlab.com/mmclassification/v0/resnet/resnetv1d101_b32x8_imagenet_20210531-6e13bcd3.log.json) |
| ResNetV1D-152         | 60.21     | 11.82    | 79.41 | 94.70 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnetv1d152_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnetv1d152_b32x8_imagenet_20210531-278cf22a.pth) &#124; [log](https://download.openmmlab.com/mmclassification/v0/resnet/resnetv1d152_b32x8_imagenet_20210531-278cf22a.log.json) |
| ResNet-50 (fp16)      | 25.56     | 4.12     | 76.30 | 93.07 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet50_8xb32-fp16_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/fp16/resnet50_batch256_fp16_imagenet_20210320-b3964210.pth) &#124; [log](https://download.openmmlab.com/mmclassification/v0/fp16/resnet50_batch256_fp16_imagenet_20210320-b3964210.log.json) |
| Wide-ResNet-50\*      | 68.88     | 11.44     | 78.48 | 94.08 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/wide-resnet50_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/resnet/wide-resnet50_3rdparty_8xb32_in1k_20220304-66678344.pth)  |
| Wide-ResNet-101\*     | 126.89    | 22.81     | 78.84 | 94.28 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet101_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/resnet/wide-resnet101_3rdparty_8xb32_in1k_20220304-8d5f9d61.pth) |
| ResNet-50 (rsb-a1)    | 25.56     | 4.12     | 80.12 | 94.78 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet50_8xb256-rsb-a1-600e_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb256-rsb-a1-600e_in1k_20211228-20e21305.pth) &#124; [log](https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb256-rsb-a1-600e_in1k_20211228-20e21305.log.json) |
| ResNet-50 (rsb-a2)    | 25.56     | 4.12     | 79.55 | 94.37 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet50_8xb256-rsb-a2-300e_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb256-rsb-a2-300e_in1k_20211228-0fd8be6e.pth) &#124; [log](https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb256-rsb-a2-300e_in1k_20211228-0fd8be6e.log.json) |
| ResNet-50 (rsb-a3)    | 25.56     | 4.12     | 78.30 | 93.80 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet50_8xb256-rsb-a3-100e_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb256-rsb-a3-100e_in1k_20211228-3493673c.pth) &#124; [log](https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb256-rsb-a3-100e_in1k_20211228-3493673c.log.json) |

*The "rsb" means using the training settings from [ResNet strikes back: An improved training procedure in timm](https://arxiv.org/abs/2110.00476).*

*Models with \* are converted from the [official repo](https://github.com/pytorch/vision). The config files of these models are only for validation. We don't ensure these config files' training accuracy and welcome you to contribute your reproduction results.*

### CUB-200-2011

|         Model         |   Pretrain   |  resolution  |  Params(M) | Flops(G) | Top-1 (%) |  Config |  Download  |
|:---------------------:|:------------:|:---------:|:---------:|:--------:|:---------:|:---------:|:---------:|
|    ResNet-50          | [ImageNet-21k-mill](https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_3rdparty-mill_in21k_20220331-faac000b.pth)  | 448x448 |   23.92   |   16.48   |   88.45   |  [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet50_8xb8_cub.py) | [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb8_cub_20220307-57840e60.pth) &#124; [log](https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb8_cub_20220307-57840e60.log.json) |


## Citation

```
@inproceedings{he2016deep,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={770--778},
  year={2016}
}
```
