# Aggregated Residual Transformations for Deep Neural Networks

## Introduction

[ALGORITHM]

```latex
@inproceedings{xie2017aggregated,
  title={Aggregated residual transformations for deep neural networks},
  author={Xie, Saining and Girshick, Ross and Doll{\'a}r, Piotr and Tu, Zhuowen and He, Kaiming},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={1492--1500},
  year={2017}
}
```

## Results and models

### ImageNet

|         Model         | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) | Config | Download |
|:---------------------:|:---------:|:--------:|:---------:|:---------:|:---------:|:--------:|
| ResNeXt-32x4d-50      | 25.03     | 4.27     | 77.92 | 93.74 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnext/resnext50_32x4d_b32x8_imagenet.py) | [model](https://download.openmmlab.com/mmclassification/v0/resnext/resnext50_32x4d_batch256_imagenet_20200708-c07adbb7.pth) &#124; [log](https://download.openmmlab.com/mmclassification/v0/resnext/resnext50_32x4d_batch256_imagenet_20200708-c07adbb7.log.json) |
| ResNeXt-32x4d-101     | 44.18     | 8.03     | 78.7  | 94.34 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnext/resnext101_32x4d_b32x8_imagenet.py) | [model](https://download.openmmlab.com/mmclassification/v0/resnext/resnext101_32x4d_batch256_imagenet_20200708-87f2d1c9.pth) &#124; [log](https://download.openmmlab.com/mmclassification/v0/resnext/resnext101_32x4d_batch256_imagenet_20200708-87f2d1c9.log.json) |
| ResNeXt-32x8d-101     | 88.79     | 16.5     | 79.22 | 94.52 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnext/resnext101_32x8d_b32x8_imagenet.py) | [model](https://download.openmmlab.com/mmclassification/v0/resnext/resnext101_32x8d_batch256_imagenet_20200708-1ec34aa7.pth) &#124; [log](https://download.openmmlab.com/mmclassification/v0/resnext/resnext101_32x8d_batch256_imagenet_20200708-1ec34aa7.log.json) |
| ResNeXt-32x4d-152     | 59.95     | 11.8     | 79.06 | 94.47 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnext/resnext152_32x4d_b32x8_imagenet.py) | [model](https://download.openmmlab.com/mmclassification/v0/resnext/resnext152_32x4d_batch256_imagenet_20200708-aab5034c.pth) &#124; [log](https://download.openmmlab.com/mmclassification/v0/resnext/resnext152_32x4d_batch256_imagenet_20200708-aab5034c.log.json) |
