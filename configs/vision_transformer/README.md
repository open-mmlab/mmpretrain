# An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale

## Introduction

[ALGORITHM]

```latex
@inproceedings{
  dosovitskiy2021an,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Alexey Dosovitskiy and Lucas Beyer and Alexander Kolesnikov and Dirk Weissenborn and Xiaohua Zhai and Thomas Unterthiner and Mostafa Dehghani and Matthias Minderer and Georg Heigold and Sylvain Gelly and Jakob Uszkoreit and Neil Houlsby},
  booktitle={International Conference on Learning Representations},
  year={2021},
  url={https://openreview.net/forum?id=YicbFdNTTy}
}
```

## Pretrain model

The training step
The pre-trained models are converted from [model zoo of Google Research](https://github.com/google-research/vision_transformer#available-vit-models).

### ImageNet 21k

|   Model    | Params(M) |  Flops(G) | Download |
|:----------:|:---------:|:---------:|:--------:|
|  ViT-B16\* |   86.86   |   33.03   | [model](https://download.openmmlab.com/mmclassification/v0/vit/pretrain/vit-base-p16_in21k-pre-3rdparty_20210820-37e7e71c.pth)|
|  ViT-B32\* |   88.30   |    8.56   | [model](https://download.openmmlab.com/mmclassification/v0/vit/pretrain/vit-base-p32_in21k-pre-3rdparty_20210820-bf66d859.pth)|
|  ViT-L16\* |  304.72   |  116.68   | [model](https://download.openmmlab.com/mmclassification/v0/vit/pretrain/vit-large-p16_in21k-pre-3rdparty_20210820-831bb8ba.pth)|

*Models with \* are converted from other repos.*


## Finetune model

The pre-trained models are converted from [model zoo of Google Research](https://github.com/google-research/vision_transformer#available-vit-models).

### ImageNet 1k
|    Model   |  Pretrain    | resolution  | Params(M) |  Flops(G) | Top-1 (%) | Top-5 (%) |   Config   | Download |
|:----------:|:------------:|:-----------:|:---------:|:---------:|:---------:|:---------:|:----------:|:--------:|
|  ViT-B16\* | ImageNet-21k |   384x384   |   86.86   |   33.03   |   85.43   |   97.77   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/vision_transformer/vit-base-p16_ft-evalonly_in-1k-384.py) | [model](https://download.openmmlab.com/mmclassification/v0/vit/finetune/vit-base-p16_in21k-pre-3rdparty_in1k-384_20210819-65c4bf44.pth)|
|  ViT-B32\* | ImageNet-21k |   384x384   |   88.30   |    8.56   |   84.01   |   97.08   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/vision_transformer/vit-base-p32_ft-evalonly_in-1k-384.py) | [model](https://download.openmmlab.com/mmclassification/v0/vit/finetune/vit-base-p32_in21k-pre-3rdparty_in1k-384_20210819-a56f8886.pth)|
|  ViT-L16\* | ImageNet-21k |   384x384   |  304.72   |  116.68   |   85.63   |   97.63   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/vision_transformer/vit-large-p16_ft-evalonly_in-1k-384.py) | [model](https://download.openmmlab.com/mmclassification/v0/vit/finetune/vit-large-p16_in21k-pre-3rdparty_in1k-384_20210819-0bb8550c.pth)|
*Models with \* are converted from other repos.*
