# Swin Transformer: Hierarchical Vision Transformer using Shifted Windows

## Introduction

[ALGORITHM]

```latex
@article{liu2021Swin,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  journal={arXiv preprint arXiv:2103.14030},
  year={2021}
}
```

## Pretrain model

The pre-trained modles are converted from [model zoo of Swin Transformer](https://github.com/microsoft/Swin-Transformer#main-results-on-imagenet-with-pretrained-models).

### ImageNet 1k

|   Model   |  Pretrain    | resolution  | Params(M) |  Flops(G) | Top-1 (%) | Top-5 (%) | Download |
|:---------:|:------------:|:-----------:|:---------:|:---------:|:---------:|:---------:|:--------:|
|  Swin-T   | ImageNet-1k  |   224x224   |   28.29   |    4.36   |   81.18   |   95.52   | [model](https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin_tiny_patch4_window7_224-160bb0a5.pth)|
|  Swin-S   | ImageNet-1k  |   224x224   |   49.61   |    8.52   |   83.21   |   96.25   | [model](https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin_small_patch4_window7_224-cc7a01c9.pth)|
|  Swin-B   | ImageNet-1k  |   224x224   |   87.77   |   15.14   |   83.42   |   96.44   | [model](https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin_base_patch4_window7_224-4670dd19.pth)|
|  Swin-B   | ImageNet-1k  |   384x384   |   87.90   |   44.49   |   84.49   |   96.95   | [model](https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin_base_patch4_window12_384-02c598a4.pth)|
|  Swin-B   | ImageNet-22k |   224x224   |   87.77   |   15.14   |   85.16   |   97.50   | [model](https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin_base_patch4_window7_224_22kto1k-f967f799.pth)|
|  Swin-B   | ImageNet-22k |   384x384   |   87.90   |   44.49   |   86.44   |   98.05   | [model](https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin_base_patch4_window12_384_22kto1k-d59b0d1d.pth)|
|  Swin-L   | ImageNet-22k |   224x224   |  196.53   |   34.04   |   86.24   |   97.88   | [model](https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin_large_patch4_window7_224_22kto1k-5f0996db.pth)|
|  Swin-L   | ImageNet-22k |   384x384   |  196.74   |  100.04   |   87.25   |   98.25   | [model](https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin_large_patch4_window12_384_22kto1k-0a40944b.pth)|


## Results and models

### ImageNet
|   Model   |  Pretrain    | resolution  | Params(M) |  Flops(G) | Top-1 (%) | Top-5 (%) |   Config   | Download |
|:---------:|:------------:|:-----------:|:---------:|:---------:|:---------:|:---------:|:----------:|:--------:|
|  Swin-T   | ImageNet-1k  |   224x224   |   28.29   |    4.36   |   81.18   |   95.61   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/swin_transformer/swin_tiny_224_b16x64_300e_imagenet.py) | [model](https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_tiny_224_b16x64_300e_imagenet_20210616_090925-66df6be6.pth)  &#124; [log](https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_tiny_224_b16x64_300e_imagenet_20210616_090925.log.json)|
|  Swin-S   | ImageNet-1k  |   224x224   |   49.61   |    8.52   |   83.02   |   96.29   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/swin_transformer/swin_small_224_b16x64_300e_imagenet.py) | [model](https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_small_224_b16x64_300e_imagenet_20210615_110219-7f9d988b.pth)  &#124; [log](https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_small_224_b16x64_300e_imagenet_20210615_110219.log.json)|
|  Swin-B   | ImageNet-1k  |   224x224   |   87.77   |   15.14   |   83.36   |   96.44   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/swin_transformer/swin_base_224_b16x64_300e_imagenet.py) | [model](https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_base_224_b16x64_300e_imagenet_20210616_190742-93230b0d.pth)  &#124; [log](https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_base_224_b16x64_300e_imagenet_20210616_190742.log.json)|
