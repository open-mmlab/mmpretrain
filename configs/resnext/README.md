# Aggregated Residual Transformations for Deep Neural Networks

## Introduction

<!-- [ALGORITHM] -->

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
| ResNeXt-32x4d-50      | 25.03     | 4.27     | 77.90 | 93.66 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnext/resnext50_32x4d_b32x8_imagenet.py) | [model](https://download.openmmlab.com/mmclassification/v0/resnext/resnext50_32x4d_b32x8_imagenet_20210429-56066e27.pth) &#124; [log](https://download.openmmlab.com/mmclassification/v0/resnext/resnext50_32x4d_b32x8_imagenet_20210429-56066e27.log.json) |
| ResNeXt-32x4d-101     | 44.18     | 8.03     | 78.71  | 94.12 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnext/resnext101_32x4d_b32x8_imagenet.py) | [model](https://download.openmmlab.com/mmclassification/v0/resnext/resnext101_32x4d_b32x8_imagenet_20210506-e0fa3dd5.pth) &#124; [log](https://download.openmmlab.com/mmclassification/v0/resnext/resnext101_32x4d_b32x8_imagenet_20210506-e0fa3dd5.log.json) |
| ResNeXt-32x8d-101     | 88.79     | 16.5     | 79.23 | 94.58 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnext/resnext101_32x8d_b32x8_imagenet.py) | [model](https://download.openmmlab.com/mmclassification/v0/resnext/resnext101_32x8d_b32x8_imagenet_20210506-23a247d5.pth) &#124; [log](https://download.openmmlab.com/mmclassification/v0/resnext/resnext101_32x8d_b32x8_imagenet_20210506-23a247d5.log.json) |
| ResNeXt-32x4d-152     | 59.95     | 11.8     | 78.93 | 94.41 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnext/resnext152_32x4d_b32x8_imagenet.py) | [model](https://download.openmmlab.com/mmclassification/v0/resnext/resnext152_32x4d_b32x8_imagenet_20210524-927787be.pth) &#124; [log](https://download.openmmlab.com/mmclassification/v0/resnext/resnext152_32x4d_b32x8_imagenet_20210524-927787be.log.json) |
