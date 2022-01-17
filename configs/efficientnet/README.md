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

|                Model                 | Params(M) | Flops(G)  | Top-1 (%)  | Top-5 (%) |                                                                 Config                                                                 |      Download      |
|:------------------------------------:|:---------:|:---------:|:----------:|:---------:|:--------------------------------------------------------------------------------------------------------------------------------------:|:------------------:|
|           EfficientNet_b0*           |   5.29    |   0.02    |   76.74    |   93.17   |      [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet_b0_b32x8_imagenet.py)       | [model]() &#124; - |
|    EfficientNet_b0 (AutoAugment)*    |   5.29    |   0.02    |   77.26    |   93.41   |      [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet_b0_b32x8_imagenet.py)       | [model]() &#124; - |
|   EfficientNet_b0 (AdvProp + AA)*    |   5.29    |   0.02    |   77.53    |   93.61   |  [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet_b0_b32x8_imagenet_advprob.py)   | [model]() &#124; - |
| EfficientNet_b0 (NoisyStudent + RA)* |   5.29    |   0.02    |   77.62    |   94.0    |      [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet_b0_b32x8_imagenet.py)       | [model]() &#124; - |
|           EfficientNet_b1*           |   7.79    |   0.03    |   78.68    |   94.28   |      [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet_b1_b32x8_imagenet.py)       | [model]() &#124; - |
|    EfficientNet_b1 (AutoAugment)*    |   7.79    |   0.03    |   79.20    |   94.42   |      [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet_b1_b32x8_imagenet.py)       | [model]() &#124; - |
|   EfficientNet_b1 (AdvProp + AA)*    |   7.79    |   0.03    |   79.52    |   94.43   |  [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet_b1_b32x8_imagenet_advprob.py)   | [model]() &#124; - |
| EfficientNet_b1 (NoisyStudent + RA)* |   7.79    |   0.03    |   81.46    |   95.84   |      [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet_b1_b32x8_imagenet.py)       | [model]() &#124; - |
|           EfficientNet_b2*           |   9.11    |   0.03    |   79.64    |   94.80   |      [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet_b2_b32x8_imagenet.py)       | [model]() &#124; - |
|    EfficientNet_b2 (AutoAugment)*    |   9.11    |   0.03    |   80.21    |   94.96   |      [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet_b2_b32x8_imagenet.py)       | [model]() &#124; - |
|   EfficientNet_b2 (AdvProp + AA)*    |   9.11    |   0.03    |   80.45    |   95.07   |  [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet_b2_b32x8_imagenet_advprob.py)   | [model]() &#124; - |
| EfficientNet_b2 (NoisyStudent + RA)* |   9.11    |   0.03    |   82.48    |   96.23   |      [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet_b2_b32x8_imagenet.py)       | [model]() &#124; - |
|           EfficientNet_b3*           |   12.23   |   0.06    |   81.01    |   95.34   |      [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet_b3_b32x8_imagenet.py)       | [model]() &#124; - |
|    EfficientNet_b3 (AutoAugment)*    |   12.23   |   0.06    |   81.58    |   95.67   |      [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet_b3_b32x8_imagenet.py)       | [model]() &#124; - |
|   EfficientNet_b3 (AdvProp + AA)*    |   12.23   |   0.06    |   81.81    |   95.69   |  [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet_b3_b32x8_imagenet_advprob.py)   | [model]() &#124; - |
| EfficientNet_b3 (NoisyStudent + RA)* |   12.23   |   0.06    |   84.02    |   96.89   |      [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet_b3_b32x8_imagenet.py)       | [model]() &#124; - |
|           EfficientNet_b4*           |   19.34   |   0.12    |   82.57    |   96.09   |      [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet_b4_b32x8_imagenet.py)       | [model]() &#124; - |
|    EfficientNet_b4 (AutoAugment)*    |   19.34   |   0.12    |   82.95    |   96.26   |      [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet_b4_b32x8_imagenet.py)       | [model]() &#124; - |
|   EfficientNet_b4 (AdvProp + AA)*    |   19.34   |   0.12    |   83.25    |   96.44   |  [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet_b4_b32x8_imagenet_advprob.py)   | [model]() &#124; - |
| EfficientNet_b4 (NoisyStudent + RA)* |   19.34   |   0.12    |   85.24    |   97.53   |      [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet_b4_b32x8_imagenet.py)       | [model]() &#124; - |
|           EfficientNet_b5*           |   30.39   |   0.24    |   83.18    |   96.47   |      [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet_b5_b32x8_imagenet.py)       | [model]() &#124; - |
|    EfficientNet_b5 (RandAugment)*    |   30.39   |   0.24    |   83.82    |   96.76   |      [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet_b5_b32x8_imagenet.py)       | [model]() &#124; - |
|   EfficientNet_b5 (AdvProp + AA)*    |   30.39   |   0.24    |   84.21    |   96.98   |  [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet_b5_b32x8_imagenet_advprob.py)   | [model]() &#124; - |
| EfficientNet_b5 (NoisyStudent + RA)* |   30.39   |   0.24    |   86.09    |   97.75   |      [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet_b5_b32x8_imagenet.py)       | [model]() &#124; - |
|    EfficientNet_b6 (AutoAugment)*    |   43.04   |   0.41    |   84.05    |   96.82   |      [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet_b6_b32x8_imagenet.py)       | [model]() &#124; - |
|   EfficientNet_b6 (AdvProp + AA)*    |   43.04   |   0.41    |   84.74    |   97.14   |  [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet_b6_b32x8_imagenet_advprob.py)   | [model]() &#124; - |
| EfficientNet_b6 (NoisyStudent + RA)* |   43.04   |   0.41    |   86.46    |   97.87   |      [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet_b6_b32x8_imagenet.py)       | [model]() &#124; - |
|    EfficientNet_b7 (AutoAugment)*    |   66.35   |   0.72    |   84.38    |   96.88   |      [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet_b7_b32x8_imagenet.py)       | [model]() &#124; - |
|    EfficientNet_b7 (RandAugment)*    |   66.35   |   0.72    |   84.91    |   97.20   |      [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet_b7_b32x8_imagenet.py)       | [model]() &#124; - |
|   EfficientNet_b7 (AdvProp + AA)*    |   66.35   |   0.72    |   85.14    |   97.23   |  [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet_b7_b32x8_imagenet_advprob.py)   | [model]() &#124; - |
| EfficientNet_b7 (NoisyStudent + RA)* |   66.35   |   0.72    |   86.83    |   98.08   |      [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet_b7_b32x8_imagenet.py)       | [model]() &#124; - |
|   EfficientNet_b8 (AdvProp + AA)*    |   87.41   |   1.09    |   85.38    |   97.28   |  [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet_b8_b32x8_imagenet_advprob.py)   | [model]() &#124; - |
|       EfficientNet_edgetpu_S*        |   5.44    |   0.03    |   76.38    |   93.23   |      [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet_es_b32x8_imagenet.py)       | [model]() &#124; - |
|       EfficientNet_edgetpu_M*        |    6.9    |   0.05    |   77.88    |   93.90   |      [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet_em_b32x8_imagenet.py)       | [model]() &#124; - |

Models with * are converted from other repos, others are trained by ourselves.
