# Model Zoo

## ImageNet

ImageNet has multiple versions, but the most commonly used one is [ILSVRC 2012](http://www.image-net.org/challenges/LSVRC/2012/).
The ResNet family models below are trained by standard data augmentations, i.e., RandomResizedCrop, RandomHorizontalFlip and Normalize.


|         Model         | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) | Download |
|:---------------------:|:---------:|:--------:|:---------:|:---------:|:--------:|
| ResNet-18             | 11.69     | 1.82     | 70.07 | 89.44 | [model](https://openmmlab.oss-accelerate.aliyuncs.com/mmclassification/v0/imagenet/resnet18_batch256_20200708-34ab8f90.pth) &#124; [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmclassification/v0/imagenet/resnet18_batch256_20200708-34ab8f90.log.json) |
| ResNet-34             | 21.8      | 3.68     | 73.85 | 91.53 | [model](https://openmmlab.oss-accelerate.aliyuncs.com/mmclassification/v0/imagenet/resnet34_batch256_20200708-32ffb4f7.pth) &#124; [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmclassification/v0/imagenet/resnet34_batch256_20200708-32ffb4f7.log.json) |
| ResNet-50             | 25.56     | 4.12     | 76.55 | 93.15 | [model](https://openmmlab.oss-accelerate.aliyuncs.com/mmclassification/v0/imagenet/resnet50_batch256_20200708-cfb998bf.pth) &#124; [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmclassification/v0/imagenet/resnet50_batch256_20200708-cfb998bf.log.json) |
| ResNet-101            | 44.55     | 7.85     | 78.18 | 94.03 | [model](https://openmmlab.oss-accelerate.aliyuncs.com/mmclassification/v0/imagenet/resnet101_batch256_20200708-753f3608.pth) &#124; [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmclassification/v0/imagenet/resnet101_batch256_20200708-753f3608.log.json) |
| ResNet-152            | 60.19     | 11.58    | 78.63 | 94.16 | [model](https://openmmlab.oss-accelerate.aliyuncs.com/mmclassification/v0/imagenet/resnet152_batch256_20200708-ec25b1f9.pth) &#124; [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmclassification/v0/imagenet/resnet152_batch256_20200708-ec25b1f9.log.json) |
| ResNetV1D-50          | 25.58     | 4.36     | 77.4  | 93.66 | [model](https://openmmlab.oss-accelerate.aliyuncs.com/mmclassification/v0/imagenet/resnetv1d50_batch256_20200708-1ad0ce94.pth) &#124; [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmclassification/v0/imagenet/resnetv1d50_batch256_20200708-1ad0ce94.log.json) |
| ResNetV1D-101         | 44.57     | 8.09     | 78.85 | 94.38 | [model](https://openmmlab.oss-accelerate.aliyuncs.com/mmclassification/v0/imagenet/resnetv1d101_batch256_20200708-9cb302ef.pth) &#124; [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmclassification/v0/imagenet/resnetv1d101_batch256_20200708-9cb302ef.log.json) |
| ResNetV1D-152         | 60.21     | 11.82    | 79.35 | 94.61 | [model](https://openmmlab.oss-accelerate.aliyuncs.com/mmclassification/v0/imagenet/resnetv1d152_batch256_20200708-e79cb6a2.pth) &#124; [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmclassification/v0/imagenet/resnetv1d152_batch256_20200708-e79cb6a2.log.json) |
| ResNeXt-32x4d-50      | 25.03     | 4.27     | 77.92 | 93.74 | [model](https://openmmlab.oss-accelerate.aliyuncs.com/mmclassification/v0/imagenet/resnext50_32x4d_batch256_20200708-c07adbb7.pth) &#124; [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmclassification/v0/imagenet/resnext50_32x4d_batch256_20200708-c07adbb7.log.json) |
| ResNeXt-32x4d-101     | 44.18     | 8.03     | 78.7  | 94.34 | [model](https://openmmlab.oss-accelerate.aliyuncs.com/mmclassification/v0/imagenet/resnext101_32x4d_batch256_20200708-87f2d1c9.pth) &#124; [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmclassification/v0/imagenet/resnext101_32x4d_batch256_20200708-87f2d1c9.log.json) |
| ResNeXt-32x8d-101     | 88.79     | 16.5     | 79.22 | 94.52 | [model](https://openmmlab.oss-accelerate.aliyuncs.com/mmclassification/v0/imagenet/resnext101_32x8d_batch256_20200708-1ec34aa7.pth) &#124; [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmclassification/v0/imagenet/resnext101_32x8d_batch256_20200708-1ec34aa7.log.json) |
| ResNeXt-32x4d-152     | 59.95     | 11.8     | 79.06 | 94.47 | [model](https://openmmlab.oss-accelerate.aliyuncs.com/mmclassification/v0/imagenet/resnext152_32x4d_batch256_20200708-aab5034c.pth) &#124; [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmclassification/v0/imagenet/resnext152_32x4d_batch256_20200708-aab5034c.log.json) |
| SE-ResNet-50          | 28.09     | 4.13     | 77.74 | 93.84 | [model](https://openmmlab.oss-accelerate.aliyuncs.com/mmclassification/v0/imagenet/se-resnet50_batch256_20200804-ae206104.pth) &#124; [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmclassification/v0/imagenet/se-resnet50_batch256_20200708-657b3c36.log.json) |
| SE-ResNet-101         | 49.33     | 7.86     | 78.26 | 94.07 | [model](https://openmmlab.oss-accelerate.aliyuncs.com/mmclassification/v0/imagenet/se-resnet101_batch256_20200804-ba5b51d4.pth) &#124; [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmclassification/v0/imagenet/se-resnet101_batch256_20200708-038a4d04.log.json) |
| ShuffleNetV1 1.0x (group=3)| 1.87 | 0.146    | 68.13 | 87.81 | [model](https://openmmlab.oss-accelerate.aliyuncs.com/mmclassification/v0/imagenet/shufflenet_v1_batch1024_20200804-5d6cec73.pth) &#124; [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmclassification/v0/imagenet/shufflenet_v1_batch1024_20200804-5d6cec73.log.json) |
| ShuffleNetV2 1.0x     | 2.28      | 0.149    | 69.55 | 88.92 | [model](https://openmmlab.oss-accelerate.aliyuncs.com/mmclassification/v0/imagenet/shufflenet_v2_batch1024_20200812-5bf4721e.pth) &#124; [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmclassification/v0/imagenet/shufflenet_v2_batch1024_20200804-8860eec9.log.json) |
| MobileNet V2          | 3.5       | 0.319    | 71.86 | 90.42 | [model](https://openmmlab.oss-accelerate.aliyuncs.com/mmclassification/v0/imagenet/mobilenet_v2_batch256_20200708-3b2dc3af.pth) &#124; [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmclassification/v0/imagenet/mobilenet_v2_batch256_20200708-3b2dc3af.log.json) |
| EfficientNet_b0               | 5.29 | 0.02   | 76.53 | 93.01 | [model]() &#124; - |
| EfficientNet_b0 (AutoAugment) | 5.29 | 0.02  | 76.84 | 93.23 | [model]() &#124; - |
| EfficientNet_b0 (AdvProp + AA)| 5.29 | 0.02  | 77.10 | 93.26 | [model]() &#124; - |
| EfficientNet_b1               | 7.7  | 0.03  | 78.54 | 94.10 | [model]() &#124; - |
| EfficientNet_b1 (AutoAugment) | 7.79 | 0.03  | 78.84 | 94.19 | [model]() &#124; - |
| EfficientNet_b1 (AdvProp + AA)| 7.79 | 0.03  | 79.27 | 94.31 | [model]() &#124; - |
| EfficientNet_b2               | 9.11 | 0.03  | 79.60 | 94.73 | [model]() &#124; - |
| EfficientNet_b2 (AutoAugment) | 9.1  | 0.03  | 80.07 | 94.90 | [model]() &#124; - |
| EfficientNet_b2 (AdvProp + AA)| 9.1  | 0.03  | 80.30 | 95.03 | [model]() &#124; - |
| EfficientNet_b3               | 12.23| 0.06  | 80.97 | 95.28 | [model]() &#124; - |
| EfficientNet_b3 (AutoAugment) | 12.23| 0.06  | 81.61 | 95.68 | [model]() &#124; - |
| EfficientNet_b3 (AdvProp + AA)| 12.23| 0.06  | 81.77 | 95.60 | [model]() &#124; - |
| EfficientNet_b4               | 19.34| 0.12  | 82.68 | 96.25 | [model]() &#124; - |
| EfficientNet_b4 (AutoAugment) | 19.34| 0.12  | 83.03 | 96.35 | [model]() &#124; - |
| EfficientNet_b4 (AdvProp + AA)| 19.34| 0.12  | 83.12 | 96.41 | [model]() &#124; - |
| EfficientNet_b5               | 30.39| 0.24  | 83.29 | 96.56 | [model]() &#124; - |
| EfficientNet_b5 (AutoAugment) | 30.39| 0.24  | 83.83 | 96.80 | [model]() &#124; - |
| EfficientNet_b5 (AdvProp + AA)| 30.39| 0.24  | 84.34 | 97.00 | [model]() &#124; - |
| EfficientNet_b6 (AutoAugment) | 43.04| 0.41  | 84.14 | 96.95 | [model]() &#124; - |
| EfficientNet_b6 (AdvProp + AA)| 43.04| 0.41  | 84.84 | 97.20 | [model]() &#124; - |
| EfficientNet_b7 (AutoAugment) | 66.35| 0.72  | 84.58 | 97.00 | [model]() &#124; - |
| EfficientNet_b7 (RandAugment) | 66.35| 0.72  | 84.93 | 97.24 | [model]() &#124; - |
| EfficientNet_b7 (AdvProp + AA)| 66.35| 0.72  | 85.25 | 97.29 | [model]() &#124; - |
| EfficientNet_b8 (AdvProp + AA)| 87.41| 1.09  | 85.36 | 97.36 | [model]() &#124; - |
| EfficientNet_edgetpu_S        | 5.44 | 0.03  | 77.28 | 93.61 | [model]() &#124; - |
| EfficientNet_edgetpu_M        | 6.9  | 0.05  | 78.70 | 94.33 | [model]() &#124; - |

Models with * are converted from other repos, others are trained by ourselves.


## CIFAR10

|         Model         | Params(M) | Flops(G) | Top-1 (%) | Download |
|:---------------------:|:---------:|:--------:|:---------:|:--------:|
| ResNet-18-b16x8 | 11.17 | 0.56 | 94.72 | [model](https://openmmlab.oss-accelerate.aliyuncs.com/mmclassification/v0/cifar10/resnet18_b16x8_20200823-f906fa4e.pth) &#124; [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmclassification/v0/cifar10/resnet18_b16x8_20200823-f906fa4e.log.json) |
| ResNet-34-b16x8 | 21.28 | 1.16 | 95.34 | [model](https://openmmlab.oss-accelerate.aliyuncs.com/mmclassification/v0/cifar10/resnet34_b16x8_20200823-52d5d832.pth) &#124; [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmclassification/v0/cifar10/resnet34_b16x8_20200823-52d5d832.log.json) |
| ResNet-50-b16x8 | 23.52 | 1.31 | 95.36 | [model](https://openmmlab.oss-accelerate.aliyuncs.com/mmclassification/v0/cifar10/resnet50_b16x8_20200823-882aa7b1.pth) &#124; [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmclassification/v0/cifar10/resnet50_b16x8_20200823-882aa7b1.log.json) |
| ResNet-101-b16x8 | 42.51 | 2.52 | 95.66 | [model](https://openmmlab.oss-accelerate.aliyuncs.com/mmclassification/v0/cifar10/resnet101_b16x8_20200823-d9501bbc.pth) &#124; [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmclassification/v0/cifar10/resnet101_b16x8_20200823-d9501bbc.log.json) |
| ResNet-152-b16x8 | 58.16 | 3.74 | 95.96 | [model](https://openmmlab.oss-accelerate.aliyuncs.com/mmclassification/v0/cifar10/resnet152_b16x8_20200823-ad4d5d0c.pth) &#124; [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmclassification/v0/cifar10/resnet152_b16x8_20200823-ad4d5d0c.log.json) |
