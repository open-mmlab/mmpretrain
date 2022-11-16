# DaViT

> [DaViT: Dual Attention Vision Transformers](https://arxiv.org/abs/2204.03645v1)

<!-- [ALGORITHM] -->

## Abstract

In this work, we introduce Dual Attention Vision Transformers (DaViT), a simple yet effective vision transformer architecture that is able to capture global context while maintaining computational efficiency. We propose approaching the problem from an orthogonal angle: exploiting self-attention mechanisms with both "spatial tokens" and "channel tokens". With spatial tokens, the spatial dimension defines the token scope, and the channel dimension defines the token feature dimension. With channel tokens, we have the inverse: the channel dimension defines the token scope, and the spatial dimension defines the token feature dimension. We further group tokens along the sequence direction for both spatial and channel tokens to maintain the linear complexity of the entire model. We show that these two self-attentions complement each other: (i) since each channel token contains an abstract representation of the entire image, the channel attention naturally captures global interactions and representations by taking all spatial positions into account when computing attention scores between channels; (ii) the spatial attention refines the local representations by performing fine-grained interactions across spatial locations, which in turn helps the global information modeling in channel attention. Extensive experiments show our DaViT achieves state-of-the-art performance on four different tasks with efficient computations. Without extra data, DaViT-Tiny, DaViT-Small, and DaViT-Base achieve 82.8%, 84.2%, and 84.6% top-1 accuracy on ImageNet-1K with 28.3M, 49.7M, and 87.9M parameters, respectively. When we further scale up DaViT with 1.5B weakly supervised image and text pairs, DaViT-Gaint reaches 90.4% top-1 accuracy on ImageNet-1K.

<div align=center>
<img src="https://user-images.githubusercontent.com/24734142/196125065-e232409b-f710-4729-b657-4e5f9158f2d1.png" width="90%"/>
</div>

## Results and models

### ImageNet-1k

|   Model   |   Pretrain   | resolution | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) |                 Config                 |                                             Download                                             |
| :-------: | :----------: | :--------: | :-------: | :------: | :-------: | :-------: | :------------------------------------: | :----------------------------------------------------------------------------------------------: |
| DaViT-T\* | From scratch |  224x224   |   28.36   |   4.54   |   82.24   |   96.13   | [config](./davit-tiny_4xb256_in1k.py)  | [model](https://download.openmmlab.com/mmclassification/v0/davit/davit-tiny_3rdparty_in1k_20221116-700fdf7d.pth) |
| DaViT-S\* | From scratch |  224x224   |   49.74   |   8.79   |   83.61   |   96.75   | [config](./davit-small_4xb256_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/davit/davit-small_3rdparty_in1k_20221116-51a849a6.pth) |
| DaViT-B\* | From scratch |  224x224   |   87.95   |   15.5   |   84.09   |   96.82   | [config](./davit-base_4xb256_in1k.py)  | [model](https://download.openmmlab.com/mmclassification/v0/davit/davit-base_3rdparty_in1k_20221116-19e0d956.pth) |

*Models with * are converted from the [official repo](https://github.com/dingmyu/davit). The config files of these models are only for validation. We don't ensure these config files' training accuracy and welcome you to contribute your reproduction results.*

Note: Inference accuracy is a bit lower than paper result because of inference code for classification doesn't exist.

## Citation

```
@inproceedings{ding2022davit,
    title={DaViT: Dual Attention Vision Transformer},
    author={Ding, Mingyu and Xiao, Bin and Codella, Noel and Luo, Ping and Wang, Jingdong and Yuan, Lu},
    booktitle={ECCV},
    year={2022},
}
```
