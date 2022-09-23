# Swin Transformer

> [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/pdf/2103.14030.pdf)

<!-- [ALGORITHM] -->

## Abstract

This paper presents a new vision Transformer, called Swin Transformer, that capably serves as a general-purpose backbone for computer vision. Challenges in adapting Transformer from language to vision arise from differences between the two domains, such as large variations in the scale of visual entities and the high resolution of pixels in images compared to words in text. To address these differences, we propose a hierarchical Transformer whose representation is computed with **S**hifted **win**dows. The shifted windowing scheme brings greater efficiency by limiting self-attention computation to non-overlapping local windows while also allowing for cross-window connection. This hierarchical architecture has the flexibility to model at various scales and has linear computational complexity with respect to image size. These qualities of Swin Transformer make it compatible with a broad range of vision tasks, including image classification (87.3 top-1 accuracy on ImageNet-1K) and dense prediction tasks such as object detection (58.7 box AP and 51.1 mask AP on COCO test-dev) and semantic segmentation (53.5 mIoU on ADE20K val). Its performance surpasses the previous state-of-the-art by a large margin of +2.7 box AP and +2.6 mask AP on COCO, and +3.2 mIoU on ADE20K, demonstrating the potential of Transformer-based models as vision backbones. The hierarchical design and the shifted window approach also prove beneficial for all-MLP architectures.

<div align=center>
<img src="https://user-images.githubusercontent.com/26739999/142576715-14668c6b-5cb8-4de8-ac51-419fae773c90.png" width="90%"/>
</div>

## Results and models

### ImageNet-21k

The pre-trained models on ImageNet-21k are used to fine-tune, and therefore don't have evaluation results.

| Model  | resolution | Params(M) | Flops(G) |                                                        Download                                                         |
| :----: | :--------: | :-------: | :------: | :---------------------------------------------------------------------------------------------------------------------: |
| Swin-B |  224x224   |   86.74   |  15.14   |    [model](https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin-base_3rdparty_in21k.pth)    |
| Swin-B |  384x384   |   86.88   |  44.49   | [model](https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin-base_3rdparty_in21k-384px.pth) |
| Swin-L |  224x224   |  195.00   |  34.04   |   [model](https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin-large_3rdparty_in21k.pth)    |
| Swin-L |  384x384   |  195.20   |  100.04  | [model](https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin-base_3rdparty_in21k-384px.pth) |

### ImageNet-1k

|  Model   |   Pretrain   | resolution | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) |                               Config                               |                               Download                                |
| :------: | :----------: | :--------: | :-------: | :------: | :-------: | :-------: | :----------------------------------------------------------------: | :-------------------------------------------------------------------: |
|  Swin-T  | From scratch |  224x224   |   28.29   |   4.36   |   81.18   |   95.61   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/swin_transformer/swin-tiny_16xb64_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_tiny_224_b16x64_300e_imagenet_20210616_090925-66df6be6.pth)  \| [log](https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_tiny_224_b16x64_300e_imagenet_20210616_090925.log.json) |
|  Swin-S  | From scratch |  224x224   |   49.61   |   8.52   |   83.02   |   96.29   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/swin_transformer/swin-small_16xb64_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_small_224_b16x64_300e_imagenet_20210615_110219-7f9d988b.pth)  \| [log](https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_small_224_b16x64_300e_imagenet_20210615_110219.log.json) |
|  Swin-B  | From scratch |  224x224   |   87.77   |  15.14   |   83.36   |   96.44   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/swin_transformer/swin_base_224_b16x64_300e_imagenet.py) | [model](https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_base_224_b16x64_300e_imagenet_20210616_190742-93230b0d.pth)  \| [log](https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_base_224_b16x64_300e_imagenet_20210616_190742.log.json) |
| Swin-S\* | From scratch |  224x224   |   49.61   |   8.52   |   83.21   |   96.25   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/swin_transformer/swin-small_16xb64_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin_small_patch4_window7_224-cc7a01c9.pth) |
| Swin-B\* | From scratch |  224x224   |   87.77   |  15.14   |   83.42   |   96.44   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/swin_transformer/swin-base_16xb64_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin_base_patch4_window7_224-4670dd19.pth) |
| Swin-B\* | From scratch |  384x384   |   87.90   |  44.49   |   84.49   |   96.95   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/swin_transformer/swin-base_16xb64_in1k-384px.py) | [model](https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin_base_patch4_window12_384-02c598a4.pth) |
| Swin-B\* | ImageNet-21k |  224x224   |   87.77   |  15.14   |   85.16   |   97.50   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/swin_transformer/swin-base_16xb64_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin_base_patch4_window7_224_22kto1k-f967f799.pth) |
| Swin-B\* | ImageNet-21k |  384x384   |   87.90   |  44.49   |   86.44   |   98.05   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/swin_transformer/swin-base_16xb64_in1k-384px.py) | [model](https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin_base_patch4_window12_384_22kto1k-d59b0d1d.pth) |
| Swin-L\* | ImageNet-21k |  224x224   |  196.53   |  34.04   |   86.24   |   97.88   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/swin_transformer/swin-large_16xb64_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin_large_patch4_window7_224_22kto1k-5f0996db.pth) |
| Swin-L\* | ImageNet-21k |  384x384   |  196.74   |  100.04  |   87.25   |   98.25   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/swin_transformer/swin-large_16xb64_in1k-384px.py) | [model](https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin_large_patch4_window12_384_22kto1k-0a40944b.pth) |

*Models with * are converted from the [official repo](https://github.com/microsoft/Swin-Transformer#main-results-on-imagenet-with-pretrained-models). The config files of these models are only for validation. We don't ensure these config files' training accuracy and welcome you to contribute your reproduction results.*

### CUB-200-2011

| Model  |                       Pretrain                        | resolution | Params(M) | Flops(G) | Top-1 (%) |                       Config                        |                        Download                        |
| :----: | :---------------------------------------------------: | :--------: | :-------: | :------: | :-------: | :-------------------------------------------------: | :----------------------------------------------------: |
| Swin-L | [ImageNet-21k](https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin-base_3rdparty_in21k-384px.pth) |  384x384   |  195.51   |  100.04  |   91.87   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/swin_transformer/swin-large_8xb8_cub_384px.py) | [model](https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin-large_8xb8_cub_384px_20220307-1bbaee6a.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin-large_8xb8_cub_384px_20220307-1bbaee6a.log.json) |

## Citation

```
@article{liu2021Swin,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  journal={arXiv preprint arXiv:2103.14030},
  year={2021}
}
```
