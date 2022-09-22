# ConvNeXt

> [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545v1)
<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->
The "Roaring 20s" of visual recognition began with the introduction of Vision Transformers (ViTs), which quickly superseded ConvNets as the state-of-the-art image classification model. A vanilla ViT, on the other hand, faces difficulties when applied to general computer vision tasks such as object detection and semantic segmentation. It is the hierarchical Transformers (e.g., Swin Transformers) that reintroduced several ConvNet priors, making Transformers practically viable as a generic vision backbone and demonstrating remarkable performance on a wide variety of vision tasks. However, the effectiveness of such hybrid approaches is still largely credited to the intrinsic superiority of Transformers, rather than the inherent inductive biases of convolutions. In this work, we reexamine the design spaces and test the limits of what a pure ConvNet can achieve. We gradually "modernize" a standard ResNet toward the design of a vision Transformer, and discover several key components that contribute to the performance difference along the way. The outcome of this exploration is a family of pure ConvNet models dubbed ConvNeXt. Constructed entirely from standard ConvNet modules, ConvNeXts compete favorably with Transformers in terms of accuracy and scalability, achieving 87.8% ImageNet top-1 accuracy and outperforming Swin Transformers on COCO detection and ADE20K segmentation, while maintaining the simplicity and efficiency of standard ConvNets.

<!-- [IMAGE] -->
<div align=center>
<img src="https://user-images.githubusercontent.com/8370623/148624004-e9581042-ea4d-4e10-b3bd-42c92b02053b.png" width="100%"/>
</div>

## Results and models

### ImageNet-1k

|      Model      |   Pretrain   | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) | Config | Download |
|:---------------:|:------------:|:---------:|:--------:|:---------:|:---------:|:------:|:--------:|
| ConvNeXt-T\*    | From scratch | 28.59 | 4.46 | 82.05 | 95.86  | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/convnext/convnext-tiny_32xb128_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/convnext/convnext-tiny_3rdparty_32xb128_in1k_20220124-18abde00.pth) |
| ConvNeXt-S\*    | From scratch | 50.22 | 8.69 | 83.13 | 96.44  | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/convnext/convnext-small_32xb128_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/convnext/convnext-small_3rdparty_32xb128_in1k_20220124-d39b5192.pth) |
| ConvNeXt-B\*    | From scratch | 88.59 | 15.36 | 83.85 | 96.74 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/convnext/convnext-base_32xb128_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/convnext/convnext-base_3rdparty_32xb128_in1k_20220124-d0915162.pth) |
| ConvNeXt-B\*    | ImageNet-21k | 88.59 | 15.36 | 85.81 | 97.86 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/convnext/convnext-base_32xb128_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/convnext/convnext-base_in21k-pre-3rdparty_32xb128_in1k_20220124-eb2d6ada.pth) |
| ConvNeXt-L\*    | From scratch | 197.77 | 34.37 | 84.30 | 96.89 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/convnext/convnext-large_64xb64_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/convnext/convnext-large_3rdparty_64xb64_in1k_20220124-f8a0ded0.pth) |
| ConvNeXt-L\*    | ImageNet-21k | 197.77 | 34.37 | 86.61 | 98.04 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/convnext/convnext-large_64xb64_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/convnext/convnext-large_in21k-pre-3rdparty_64xb64_in1k_20220124-2412403d.pth) |
| ConvNeXt-XL\*   | ImageNet-21k | 350.20 | 60.93 | 86.97 | 98.20 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/convnext/convnext-xlarge_64xb64_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/convnext/convnext-xlarge_in21k-pre-3rdparty_64xb64_in1k_20220124-76b6863d.pth) |

*Models with \* are converted from the [official repo](https://github.com/facebookresearch/ConvNeXt). The config files of these models are only for inference. We don't ensure these config files' training accuracy and welcome you to contribute your reproduction results.*

### Pre-trained Models

The pre-trained models on ImageNet-1k or ImageNet-21k are used to fine-tune on the downstream tasks.

|       Model    | Training Data | Params(M) | Flops(G) | Download |
|:--------------:|:-------------:|:---------:|:--------:|:--------:|
| ConvNeXt-T\*   | ImageNet-1k   | 28.59     | 4.46     | [model](https://download.openmmlab.com/mmclassification/v0/convnext/convnext-tiny_3rdparty_32xb128-noema_in1k_20220222-2908964a.pth) |
| ConvNeXt-S\*   | ImageNet-1k   | 50.22     | 8.69     | [model](https://download.openmmlab.com/mmclassification/v0/convnext/convnext-small_3rdparty_32xb128-noema_in1k_20220222-fa001ca5.pth) |
| ConvNeXt-B\*   | ImageNet-1k   | 88.59     | 15.36    | [model](https://download.openmmlab.com/mmclassification/v0/convnext/convnext-base_3rdparty_32xb128-noema_in1k_20220222-dba4f95f.pth) |
| ConvNeXt-B\*   | ImageNet-21k  | 88.59     | 15.36    | [model](https://download.openmmlab.com/mmclassification/v0/convnext/convnext-base_3rdparty_in21k_20220124-13b83eec.pth) |
| ConvNeXt-L\*   | ImageNet-21k  | 197.77    | 34.37    | [model](https://download.openmmlab.com/mmclassification/v0/convnext/convnext-large_3rdparty_in21k_20220124-41b5a79f.pth) |
| ConvNeXt-XL\*  | ImageNet-21k  | 350.20    | 60.93    | [model](https://download.openmmlab.com/mmclassification/v0/convnext/convnext-xlarge_3rdparty_in21k_20220124-f909bad7.pth) |

*Models with \* are converted from the [official repo](https://github.com/facebookresearch/ConvNeXt).*

## Citation

```bibtex
@Article{liu2022convnet,
  author  = {Zhuang Liu and Hanzi Mao and Chao-Yuan Wu and Christoph Feichtenhofer and Trevor Darrell and Saining Xie},
  title   = {A ConvNet for the 2020s},
  journal = {arXiv preprint arXiv:2201.03545},
  year    = {2022},
}
```
