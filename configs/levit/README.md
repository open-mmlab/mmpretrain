# LeViT

> [LeViT: a Vision Transformer in ConvNet’s Clothing for Faster Inference](https://arxiv.org/pdf/2104.01136.pdf)

<!-- [ALGORITHM] -->

## Abstract

We design a family of image classification architectures that optimize the trade-off between accuracy and efficiency in a high-speed regime. Our work exploits recent findings in attention-based architectures, which are competitive on highly parallel processing hardware. We revisit principles from the extensive literature on convolutional neural networks to apply them to transformers, in particular activation maps with decreasing resolutions. We also introduce the attention bias, a new way to integrate positional information in vision transformers. As a result, we propose LeVIT: a hybrid neural network for fast inference image classification. We consider different measures of efficiency on different hardware platforms, so as to best reflect a wide range of application scenarios. Our extensive experiments empirically validate our technical choices and show they are suitable to most architectures. Overall, LeViT significantly outperforms existing convnets and vision transformers with respect to the speed/accuracy tradeoff. For example, at 80% ImageNet top-1 accuracy, LeViT is 5 times faster than EfficientNet on CPU.

<div align=center>
<img src="https://raw.githubusercontent.com/facebookresearch/LeViT/main/.github/levit.png" width="90%"/>
</div>

## Results and models

### ImageNet-1k

|   Model    | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) |                                 Config                                 |
| :--------: | :-------: | :------: | :-------: | :-------: | :--------------------------------------------------------------------: |
| LeViT-128S |   7.39    |  0.310   |   76.24   |   92.84   |                     [config](./levit-128s-p16.py)                      |
| LeViT-128  |   8.82    |  0.413   |   78.43   |   93.86   |                      [config](./levit-128-p16.py)                      |
| LeViT-192  |   10.56   |  0.668   |   79.73   |   94.64   |                      [config](./levit-192-p16.py)                      |
| LeViT-256  |   18.38   |  1.142   |   81.14   |   95.14   | [config](./levit-256-p16_4xb256_autoaug-mixup-lbs-coslr-1000e_in1k.py) |
| LeViT-384  |   38.36   |  2.373   |   82.16   |   95.59   |                      [config](./levit-384-p16.py)                      |

## Citation

```
@InProceedings{Graham_2021_ICCV,
    author    = {Graham, Benjamin and El-Nouby, Alaaeldin and Touvron, Hugo and Stock, Pierre and Joulin, Armand and Jegou, Herve and Douze, Matthijs},
    title     = {LeViT: A Vision Transformer in ConvNet's Clothing for Faster Inference},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {12259-12269}
}
```
