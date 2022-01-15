# Rethinking Model Scaling for Convolutional Neural Networks

## Introduction

<!-- [ALGORITHM] -->

```latex
@inproceedings{tan2019efficientnet,
  title={Efficientnet: Rethinking model scaling for convolutional neural networks},
  author={Tan, Mingxing and Le, Quoc},
  booktitle={International Conference on Machine Learning},
  pages={6105--6114},
  year={2019},
  organization={PMLR}
}
```

## Results and models

### ImageNet

|         Model         | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) | Config | Download |
|:---------------------:|:---------:|:--------:|:---------:|:---------:|:---------:|:--------:|
| EfficientNet_b0*               | 5.29 | 0.02  | 76.74 | 93.18 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet_b0_b32x8_imagenet.py) |  [model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet_b0_20200902-fbd07c93.pth) &#124; - |
| EfficientNet_b0 (AutoAugment)* | 5.29 | 0.02  | 77.27 | 93.4 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet_b0_b32x8_imagenet.py) |[model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet_b0_autoaugment_20200902-bc21d1cb.pth) &#124; - |
| EfficientNet_b0 (AdvProp + AA)*| 5.29 | 0.02  | 77.52 | 93.62 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet_b0_b32x8_imagenet.py) |[model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet_b0_advprob_20200902-71f75f44.pth) &#124; - |
| EfficientNet_b1*               | 7.79 | 0.03  | 78.69 | 94.29 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet_b1_b32x8_imagenet.py) | [model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet_b1_20200902-c43b8538.pth) &#124; - |
| EfficientNet_b1 (AutoAugment)* | 7.79 | 0.03  | 79.19 | 94.42 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet_b1_b32x8_imagenet.py) | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet_b1_b32x8_imagenet.py) |[model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet_b1_autoaugment_20200902-6057bf77.pth) &#124; - |
| EfficientNet_b1 (AdvProp + AA)*| 7.79 | 0.03  | 79.51 | 94.42 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet_b1_b32x8_imagenet.py) | [model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet_b1_advprob_20200902-7b910c4e.pth) &#124; - |
| EfficientNet_b2*               | 9.11 | 0.03  | 79.63 | 94.79 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet_b2_b32x8_imagenet.py) | [model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet_b2_20200902-28d2d19a.pth) &#124; - |
| EfficientNet_b2 (AutoAugment)* | 9.11 | 0.03  | 80.21 | 94.96 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet_b2_b32x8_imagenet.py) | [model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet_b2_autoaugment_20200902-755b5570.pth) &#124; - |
| EfficientNet_b2 (AdvProp + AA)*| 9.11 | 0.03  | 80.45 | 95.07 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet_b2_b32x8_imagenet.py) | [model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet_b2_advprob_20200902-92aae5db.pth) &#124; - |
| EfficientNet_b3*               | 12.23| 0.06  | 81.02 | 95.34 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet_b3_b32x8_imagenet.py) | [model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet_b3_20200902-6b3b50db.pth) &#124; - |
| EfficientNet_b3 (AutoAugment)* | 12.23| 0.06  | 81.57 | 95.67 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet_b3_b32x8_imagenet.py) | [model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet_b3_autoaugment_20200902-98895895.pth) &#124; - |
| EfficientNet_b3 (AdvProp + AA)*| 12.23| 0.06  | 81.82 | 95.69 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet_b3_b32x8_imagenet.py) | [model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet_b3_advprob_20200902-2e57aa32.pth) &#124; - |
| EfficientNet_b4*               | 19.34| 0.12  | 82.57 | 96.09 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet_b4_b32x8_imagenet.py) | [model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet_b4_20200902-6e724d3d.pth) &#124; - |
| EfficientNet_b4 (AutoAugment)* | 19.34| 0.12  | 82.95 | 96.26 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet_b4_b32x8_imagenet.py) | [model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet_b4_autoaugment_20200902-cb07b99a.pth) &#124; - |
| EfficientNet_b4 (AdvProp + AA)*| 19.34| 0.12  | 83.25 | 96.45 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet_b4_b32x8_imagenet.py) | [model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet_b4_advprob_20200902-d2a17db9.pth) &#124; - |
| EfficientNet_b5*               | 30.39| 0.24  | 83.18 | 96.47 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet_b5_b32x8_imagenet.py) | [model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet_b5_20200902-bfd0f1db.pth) &#124; - |
| EfficientNet_b5 (RandAugment)* | 30.39| 0.24  | 83.82 | 96.77 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet_b5_b32x8_imagenet.py) | [model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet_b5_randaugment_20200902-ea4db767.pth) &#124; - |
| EfficientNet_b5 (AdvProp + AA)*| 30.39| 0.24  | 84.21 | 96.98 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet_b5_b32x8_imagenet.py) | [model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet_b5_advprob_20200902-27015836.pth) &#124; - |
| EfficientNet_b6 (AutoAugment)* | 43.04| 0.41  | 84.06 | 96.82 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet_b6_b32x8_imagenet.py) | [model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet_b6_autoaugment_20200902-e751721d.pth) &#124; - |
| EfficientNet_b6 (AdvProp + AA)*| 43.04| 0.41  | 84.73 | 97.14 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet_b6_b32x8_imagenet.py) | [model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet_b6_advprob_20200902-38908102.pth) &#124; - |
| EfficientNet_b7 (AutoAugment)* | 66.35| 0.72  | 84.39 | 96.88 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet_b7_b32x8_imagenet.py) | [model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet_b7_autoaugment_20200902-848069e8.pth) &#124; - |
| EfficientNet_b7 (RandAugment)* | 66.35| 0.72  | 84.92 | 97.21 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet_b7_b32x8_imagenet.py) |[model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet_b7_randaugment_20200902-584f1258.pth) &#124; - |
| EfficientNet_b7 (AdvProp + AA)*| 66.35| 0.72  | 85.14 | 97.23 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet_b7_b32x8_imagenet.py) |[model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet_b7_advprob_20200902-2269887d.pth) &#124; - |
| EfficientNet_b8 (AdvProp + AA)*| 87.41| 1.09  | 85.39 | 97.28 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet_b8_b32x8_imagenet.py) |[model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet_b8_advprob_20200902-7673a8bf.pth) &#124; - |
| EfficientNet_edgetpu_S*        | 5.44 | 0.03  | 77.63 | 93.80 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet_es_b32x8_imagenet.py) |[model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet_es_20200902-81a6b8fc.pth) &#124; - |
| EfficientNet_edgetpu_M*        | 6.9  | 0.05  | 78.87 | 94.49 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet_em_b32x8_imagenet.py) |[model](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet_em_20200902-d9c295bc.pth) &#124; - |

Models with * are converted from other repos, others are trained by ourselves.
