# TinyViT

> [TinyViT: Fast Pretraining Distillation for Small Vision Transformers](https://arxiv.org/abs/2207.10666)

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Vision transformer (ViT) recently has drawn great attention in computer vision due to its remarkable model capability. However, most prevailing ViT models suffer from huge number of parameters, restricting their applicability on devices with limited resources. To alleviate this issue, we propose TinyViT, a new family of tiny and efficient small vision transformers pretrained on large-scale datasets with our proposed fast distillation framework. The central idea is to transfer knowledge from large pretrained models to small ones, while enabling small models to get the dividends of massive pretraining data. More specifically, we apply distillation during pretraining for knowledge transfer. The logits of large teacher models are sparsified and stored in disk in advance to save the memory cost and computation overheads. The tiny student transformers are automatically scaled down from a large pretrained model with computation and parameter constraints. Comprehensive experiments demonstrate the efficacy of TinyViT. It achieves a top-1 accuracy of 84.8% on ImageNet-1k with only 21M parameters, being comparable to SwinB pretrained on ImageNet-21k while using 4.2 times fewer parameters. Moreover, increasing image resolutions, TinyViT can reach 86.5% accuracy, being slightly better than Swin-L while using only 11% parameters. Last but not the least, we demonstrate a good transfer ability of TinyViT on various downstream tasks.

<!-- [IMAGE] -->

<div align=center>
<img src="https://github.com/microsoft/Cream/raw/main/TinyViT/.figure/framework.png" width="100%">
</div>

## Results and models

### ImageNet-1k

|            Model            |   Pretrain   | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) |                             Config                             |                             Download                             |
| :-------------------------: | :----------: | :-------: | :------: | :-------: | :-------: | :------------------------------------------------------------: | :--------------------------------------------------------------: |
|      TinyViT-5M-224\*       | From scratch |   28.59   |   4.46   |   82.05   |   95.86   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/convnext/convnext-tiny_32xb128_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/convnext/convnext-tiny_3rdparty_32xb128_in1k_20220124-18abde00.pth) |
|      TinyViT-11M-224\*      | From scratch |   50.22   |   8.69   |   83.13   |   96.44   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/convnext/convnext-small_32xb128_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/convnext/convnext-small_3rdparty_32xb128_in1k_20220124-d39b5192.pth) |
|      TinyViT-21M-224\*      | From scratch |   88.59   |  15.36   |   83.85   |   96.74   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/convnext/convnext-base_32xb128_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/convnext/convnext-base_3rdparty_32xb128_in1k_20220124-d0915162.pth) |
| TinyViT-5M-224-Distilled\*  | ImageNet-21k |   88.59   |  15.36   |   85.81   |   97.86   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/convnext/convnext-base_32xb128_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/convnext/convnext-base_in21k-pre-3rdparty_32xb128_in1k_20220124-eb2d6ada.pth) |
| TinyViT-11M-224-Distilled\* | From scratch |  197.77   |  34.37   |   84.30   |   96.89   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/convnext/convnext-large_64xb64_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/convnext/convnext-large_3rdparty_64xb64_in1k_20220124-f8a0ded0.pth) |
| TinyViT-21M-224-Distilled\* | ImageNet-21k |  197.77   |  34.37   |   86.61   |   98.04   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/convnext/convnext-large_64xb64_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/convnext/convnext-large_in21k-pre-3rdparty_64xb64_in1k_20220124-2412403d.pth) |
| TinyViT-21M-384-Distilled\* | ImageNet-21k |  350.20   |  60.93   |   86.97   |   98.20   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/convnext/convnext-xlarge_64xb64_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/convnext/convnext-xlarge_in21k-pre-3rdparty_64xb64_in1k_20220124-76b6863d.pth) |
| TinyViT-21M-812-Distilled\* | ImageNet-21k |  350.20   |  60.93   |   86.97   |   98.20   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/convnext/convnext-xlarge_64xb64_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/convnext/convnext-xlarge_in21k-pre-3rdparty_64xb64_in1k_20220124-76b6863d.pth) |

*Models with * are converted from the [official repo](https://github.com/facebookresearch/ConvNeXt). The config files of these models are only for inference. We don't ensure these config files' training accuracy and welcome you to contribute your reproduction results.*
