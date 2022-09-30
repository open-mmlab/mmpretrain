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
| DeiT3-S\* | From scratch |  224x224   |   22.06   |   4.61   |   81.31   |   95.35   | [config](./deit3-small-p16-64xb64_in1k-224.py)  | [model](<>) |
| DeiT3-S\* | From scratch |  384x384   |   22.21   |  15.52   |   83.20   |   96.55   | [config](./deit3-small-p16-64xb64_in1k-384.py)  | [model](<>) |
| DeiT3-S\* | ImageNet-21k |  224x224   |   22.06   |   4.61   |   82.76   |   96.70   | [config](./deit3-small-p16-64xb64_in1k-224.py)  | [model](<>) |
| DeiT3-S\* | ImageNet-21k |  384x384   |   22.21   |  15.52   |   84.28   |   97.32   | [config](./deit3-small-p16-64xb64_in1k-384.py)  | [model](<>) |
| DeiT3-M\* | From scratch |  224x224   |   38.85   |   8.00   |   82.90   |   96.26   | [config](./deit3-medium-p16-64xb64_in1k-224.py) | [model](<>) |
| DeiT3-M\* | ImageNet-21k |  224x224   |   38.85   |   8.00   |   84.17   |   97.07   | [config](./deit3-medium-p16-64xb64_in1k-224.py) | [model](<>) |
| DeiT3-B\* | From scratch |  224x224   |   86.59   |  17.58   |   83.68   |   96.55   |  [config](./deit3-base-p16-64xb64_in1k-224.py)  | [model](<>) |
| DeiT3-B\* | From scratch |  384x384   |   88.88   |  55.54   |   85.03   |   97.20   |  [config](./deit3-base-p16-64xb32_in1k-384.py)  | [model](<>) |
| DeiT3-B\* | ImageNet-21k |  224x224   |   86.59   |  17.58   |   85.48   |   97.57   |  [config](./deit3-base-p16-64xb64_in1k-224.py)  | [model](<>) |
| DeiT3-B\* | ImageNet-21k |  384x384   |   88.88   |  55.54   |   86.24   |   97.98   |  [config](./deit3-base-p16-64xb32_in1k-384.py)  | [model](<>) |
| DeiT3-L\* | From scratch |  224x224   |    304    |  61.60   |   84.63   |   96.95   | [config](./deit3-large-p16-64xb64_in1k-224.py)  | [model](<>) |
| DeiT3-L\* | From scratch |  384x384   |    305    |   191    |   85.94   |   97.64   | [config](./deit3-large-p16-64xb16_in1k-384.py)  | [model](<>) |
| DeiT3-L\* | ImageNet-21k |  224x224   |    304    |  61.60   |   86.81   |   98.12   | [config](./deit3-large-p16-64xb64_in1k-224.py)  | [model](<>) |
| DeiT3-L\* | ImageNet-21k |  384x384   |    305    |   191    |   87.22   |   98.40   | [config](./deit3-large-p16-64xb16_in1k-384.py)  | [model](<>) |
| DeiT3-H\* | From scratch |  224x224   |    632    |   167    |   85.15   |   97.24   |  [config](./deit3-huge-p16-64xb32_in1k-224.py)  | [model](<>) |
| DeiT3-H\* | ImageNet-21k |  224x224   |    632    |   167    |   86.90   |   98.14   |  [config](./deit3-huge-p16-64xb32_in1k-224.py)  | [model](<>) |

*Models with * are converted from the [official repo](https://github.com/facebookresearch/deit). The config files of these models are only for validation. We don't ensure these config files' training accuracy and welcome you to contribute your reproduction results.*

## Citation

```
@article{dong2022ict,
  title={Bootstrapped Masked Autoencoders for Vision BERT Pretraining},
  author={Xiaoyi Dong, Jianmin Bao, Ting Zhang, Dongdong Chen, Weiming Zhang, Lu Yuan, Dong Chen, Fang Wen, Nenghai Yu},
  journal={arXiv preprint arXiv:2207.07116},
  year={2022}
}
```
