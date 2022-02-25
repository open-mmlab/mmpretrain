# Tokens-to-Token ViT

> [Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet](https://arxiv.org/abs/2101.11986)
<!-- [ALGORITHM] -->

## Abstract

Transformers, which are popular for language modeling, have been explored for solving vision tasks recently, \eg, the Vision Transformer (ViT) for image classification. The ViT model splits each image into a sequence of tokens with fixed length and then applies multiple Transformer layers to model their global relation for classification. However, ViT achieves inferior performance to CNNs when trained from scratch on a midsize dataset like ImageNet. We find it is because: 1) the simple tokenization of input images fails to model the important local structure such as edges and lines among neighboring pixels, leading to low training sample efficiency; 2) the redundant attention backbone design of ViT leads to limited feature richness for fixed computation budgets and limited training samples. To overcome such limitations, we propose a new Tokens-To-Token Vision Transformer (T2T-ViT), which incorporates 1) a layer-wise Tokens-to-Token (T2T) transformation to progressively structurize the image to tokens by recursively aggregating neighboring Tokens into one Token (Tokens-to-Token), such that local structure represented by surrounding tokens can be modeled and tokens length can be reduced; 2) an efficient backbone with a deep-narrow structure for vision transformer motivated by CNN architecture design after empirical study. Notably, T2T-ViT reduces the parameter count and MACs of vanilla ViT by half, while achieving more than 3.0\% improvement when trained from scratch on ImageNet. It also outperforms ResNets and achieves comparable performance with MobileNets by directly training on ImageNet. For example, T2T-ViT with comparable size to ResNet50 (21.5M parameters) can achieve 83.3\% top1 accuracy in image resolution 384×384 on ImageNet.

<div align=center>
<img src="https://user-images.githubusercontent.com/26739999/142578381-e9040610-05d9-457c-8bf5-01c2fa94add2.png" width="60%"/>
</div>

## Results and models

### ImageNet-1k

|      Model     | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) | Config | Download |
|:--------------:|:---------:|:--------:|:---------:|:---------:|:------:|:--------:|
| T2T-ViT_t-14   |   21.47   |  4.34    | 81.83     | 95.84     | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/t2t_vit/t2t-vit-t-14_8xb64_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/t2t-vit/t2t-vit-t-14_8xb64_in1k_20211220-f7378dd5.pth)  &#124; [log](https://download.openmmlab.com/mmclassification/v0/t2t-vit/t2t-vit-t-14_8xb64_in1k_20211220-f7378dd5.log.json)|
| T2T-ViT_t-19   |   39.08   |  7.80    | 82.63     | 96.18     | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/t2t_vit/t2t-vit-t-19_8xb64_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/t2t-vit/t2t-vit-t-19_8xb64_in1k_20211214-7f5e3aaf.pth)  &#124; [log](https://download.openmmlab.com/mmclassification/v0/t2t-vit/t2t-vit-t-19_8xb64_in1k_20211214-7f5e3aaf.log.json)|
| T2T-ViT_t-24   |   64.00   | 12.69    | 82.71     | 96.09     | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/t2t_vit/t2t-vit-t-24_8xb64_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/t2t-vit/t2t-vit-t-24_8xb64_in1k_20211214-b2a68ae3.pth)  &#124; [log](https://download.openmmlab.com/mmclassification/v0/t2t-vit/t2t-vit-t-24_8xb64_in1k_20211214-b2a68ae3.log.json)|

*In consistent with the [official repo](https://github.com/yitu-opensource/T2T-ViT), we adopt the best checkpoints during training.*

## Citation

```
@article{yuan2021tokens,
  title={Tokens-to-token vit: Training vision transformers from scratch on imagenet},
  author={Yuan, Li and Chen, Yunpeng and Wang, Tao and Yu, Weihao and Shi, Yujun and Tay, Francis EH and Feng, Jiashi and Yan, Shuicheng},
  journal={arXiv preprint arXiv:2101.11986},
  year={2021}
}
```
