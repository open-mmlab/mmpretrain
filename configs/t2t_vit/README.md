# Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet
<!-- {Tokens-to-Token ViT} -->

## Introduction

<!-- [ALGORITHM] -->

```latex
@article{yuan2021tokens,
  title={Tokens-to-token vit: Training vision transformers from scratch on imagenet},
  author={Yuan, Li and Chen, Yunpeng and Wang, Tao and Yu, Weihao and Shi, Yujun and Tay, Francis EH and Feng, Jiashi and Yan, Shuicheng},
  journal={arXiv preprint arXiv:2101.11986},
  year={2021}
}
```

## Pretrain model

The pre-trained modles are converted from [official repo](https://github.com/yitu-opensource/T2T-ViT/tree/main#2-t2t-vit-models).

### ImageNet-1k

|      Model     | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) | Config | Download |
|:--------------:|:---------:|:--------:|:---------:|:---------:|:------:|:--------:|
| T2T-ViT_t-14\* |   21.47   |  4.34    | 81.69     | 95.85     | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/t2t_vit/t2t-vit-t-14_8xb64_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/t2t-vit/t2t-vit-t-14_3rdparty_8xb64_in1k_20210928-b7c09b62.pth)  &#124; [log]()|
| T2T-ViT_t-19\* |   39.08   |  7.80    | 82.43     | 96.08     | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/t2t_vit/t2t-vit-t-19_8xb64_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/t2t-vit/t2t-vit-t-19_3rdparty_8xb64_in1k_20210928-7f1478d5.pth)  &#124; [log]()|
| T2T-ViT_t-24\* |   64.00   | 12.69    | 82.55     | 96.06     | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/t2t_vit/t2t-vit-t-24_8xb64_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/t2t-vit/t2t-vit-t-24_3rdparty_8xb64_in1k_20210928-fe95a61b.pth)  &#124; [log]()|

*Models with \* are converted from other repos.*

## Results and models

Waiting for adding.
