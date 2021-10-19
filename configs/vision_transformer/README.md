# An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
<!-- {Vision Transformer} -->

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

The training step of Vision Transformers is divided into two steps. The first
step is training the model on a large dataset, like ImageNet-21k, and get the
pretrain model. And the second step is training the model on the target dataset,
like ImageNet-1k, and get the finetune model. Here, we provide both pretrain
models and finetune models.

## Pretrain model

The pre-trained models are converted from [model zoo of Google Research](https://github.com/google-research/vision_transformer#available-vit-models).

### ImageNet 21k

|   Model    | Params(M) |  Flops(G) | Download |
|:----------:|:---------:|:---------:|:--------:|
|  ViT-B16\* |   86.86   |   33.03   | [model](https://download.openmmlab.com/mmclassification/v0/vit/pretrain/vit-base-p16_3rdparty_pt-64xb64_in1k-224_20210928-02284250.pth)|
|  ViT-B32\* |   88.30   |    8.56   | [model](https://download.openmmlab.com/mmclassification/v0/vit/pretrain/vit-base-p32_3rdparty_pt-64xb64_in1k-224_20210928-eee25dd4.pth)|
|  ViT-L16\* |  304.72   |  116.68   | [model](https://download.openmmlab.com/mmclassification/v0/vit/pretrain/vit-large-p16_3rdparty_pt-64xb64_in1k-224_20210928-0001f9a1.pth)|

*Models with \* are converted from other repos.*


## Finetune model

The finetune models are converted from [model zoo of Google Research](https://github.com/google-research/vision_transformer#available-vit-models).

### ImageNet 1k
|    Model   |  Pretrain    | resolution  | Params(M) |  Flops(G) | Top-1 (%) | Top-5 (%) |   Config   | Download |
|:----------:|:------------:|:-----------:|:---------:|:---------:|:---------:|:---------:|:----------:|:--------:|
|  ViT-B16\* | ImageNet-21k |   384x384   |   86.86   |   33.03   |   85.43   |   97.77   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/vision_transformer/vit-base-p16_ft-64xb64_in1k-384.py) | [model](https://download.openmmlab.com/mmclassification/v0/vit/finetune/vit-base-p16_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-98e8652b.pth)|
|  ViT-B32\* | ImageNet-21k |   384x384   |   88.30   |    8.56   |   84.01   |   97.08   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/vision_transformer/vit-base-p32_ft-64xb64_in1k-384.py) | [model](https://download.openmmlab.com/mmclassification/v0/vit/finetune/vit-base-p32_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-9cea8599.pth)|
|  ViT-L16\* | ImageNet-21k |   384x384   |  304.72   |  116.68   |   85.63   |   97.63   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/vision_transformer/vit-large-p16_ft-64xb64_in1k-384.py) | [model](https://download.openmmlab.com/mmclassification/v0/vit/finetune/vit-large-p16_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-b20ba619.pth)|

*Models with \* are converted from other repos.*
