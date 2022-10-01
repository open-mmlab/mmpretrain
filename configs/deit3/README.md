# DeiT III: Revenge of the ViT

> [DeiT III: Revenge of the ViT](https://arxiv.org/pdf/2204.07118.pdf)

<!-- [ALGORITHM] -->

## Abstract

A Vision Transformer (ViT) is a simple neural architecture amenable to serve several computer vision tasks. It has limited built-in architectural priors, in contrast to more recent architectures that incorporate priors either about the input data or of specific tasks. Recent works show that ViTs benefit from self-supervised pre-training, in particular BerT-like pre-training like BeiT. In this paper, we revisit the supervised training of ViTs. Our procedure builds upon and simplifies a recipe introduced for training ResNet-50. It includes a new simple data-augmentation procedure with only 3 augmentations, closer to the practice in self-supervised learning. Our evaluations on Image classification (ImageNet-1k with and without pre-training on ImageNet-21k), transfer learning and semantic segmentation show that our procedure outperforms by a large margin previous fully supervised training recipes for ViT. It also reveals that the performance of our ViT trained with supervision is comparable to that of more recent architectures. Our results could serve as better baselines for recent self-supervised approaches demonstrated on ViT.

<div align=center>
<img src="https://user-images.githubusercontent.com/24734142/192964480-46726469-21d9-4e45-a06a-87c6a57c3367.png" width="90%"/>
</div>

## Results and models

### ImageNet-1k

|   Model   |   Pretrain   | resolution | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) |                     Config                      |  Download   |
| :-------: | :----------: | :--------: | :-------: | :------: | :-------: | :-------: | :---------------------------------------------: | :---------: |
| DeiT3-S\* | From scratch |  224x224   |   22.06   |   4.61   |   81.35   |   95.31   | [config](./deit3-small-p16-64xb64_in1k-224.py)  | [model](<>) |
| DeiT3-S\* | From scratch |  384x384   |   22.21   |  15.52   |   83.43   |   96.68   | [config](./deit3-small-p16-64xb64_in1k-384.py)  | [model](<>) |
| DeiT3-S\* | ImageNet-21k |  224x224   |   22.06   |   4.61   |   83.06   |   96.77   | [config](./deit3-small-p16-64xb64_in1k-224.py)  | [model](<>) |
| DeiT3-S\* | ImageNet-21k |  384x384   |   22.21   |  15.52   |   84.84   |   97.48   | [config](./deit3-small-p16-64xb64_in1k-384.py)  | [model](<>) |
| DeiT3-M\* | From scratch |  224x224   |   38.85   |   8.00   |   82.99   |   96.22   | [config](./deit3-medium-p16-64xb64_in1k-224.py) | [model](<>) |
| DeiT3-M\* | ImageNet-21k |  224x224   |   38.85   |   8.00   |   84.56   |   97.19   | [config](./deit3-medium-p16-64xb64_in1k-224.py) | [model](<>) |
| DeiT3-B\* | From scratch |  224x224   |   86.59   |  17.58   |   83.80   |   96.55   |  [config](./deit3-base-p16-64xb64_in1k-224.py)  | [model](<>) |
| DeiT3-B\* | From scratch |  384x384   |   88.88   |  55.54   |   85.08   |   97.25   |  [config](./deit3-base-p16-64xb32_in1k-384.py)  | [model](<>) |
| DeiT3-B\* | ImageNet-21k |  224x224   |   86.59   |  17.58   |   85.70   |   97.75   |  [config](./deit3-base-p16-64xb64_in1k-224.py)  | [model](<>) |
| DeiT3-B\* | ImageNet-21k |  384x384   |   88.88   |  55.54   |   86.73   |   98.11   |  [config](./deit3-base-p16-64xb32_in1k-384.py)  | [model](<>) |
| DeiT3-L\* | From scratch |  224x224   |    304    |  61.60   |   84.87   |   97.01   | [config](./deit3-large-p16-64xb64_in1k-224.py)  | [model](<>) |
| DeiT3-L\* | From scratch |  384x384   |    305    |   191    |   85.82   |   97.60   | [config](./deit3-large-p16-64xb16_in1k-384.py)  | [model](<>) |
| DeiT3-L\* | ImageNet-21k |  224x224   |    304    |  61.60   |   86.97   |   98.24   | [config](./deit3-large-p16-64xb64_in1k-224.py)  | [model](<>) |
| DeiT3-L\* | ImageNet-21k |  384x384   |    305    |   191    |   87.73   |   98.51   | [config](./deit3-large-p16-64xb16_in1k-384.py)  | [model](<>) |
| DeiT3-H\* | From scratch |  224x224   |    632    |   167    |   85.21   |   97.36   |  [config](./deit3-huge-p16-64xb32_in1k-224.py)  | [model](<>) |
| DeiT3-H\* | ImageNet-21k |  224x224   |    632    |   167    |   87.19   |   98.26   |  [config](./deit3-huge-p16-64xb32_in1k-224.py)  | [model](<>) |

*Models with * are converted from the [official repo](https://github.com/facebookresearch/deit). The config files of these models are only for validation. We don't ensure these config files' training accuracy and welcome you to contribute your reproduction results.*

## Citation

```
@article{Touvron2022DeiTIR,
  title={DeiT III: Revenge of the ViT},
  author={Hugo Touvron and Matthieu Cord and Herve Jegou},
  journal={arXiv preprint arXiv:2204.07118},
  year={2022},
}
```
