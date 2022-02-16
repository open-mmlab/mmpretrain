# ResNeXt

> [Aggregated Residual Transformations for Deep Neural Networks](https://openaccess.thecvf.com/content_cvpr_2017/html/Xie_Aggregated_Residual_Transformations_CVPR_2017_paper.html)
<!-- [ALGORITHM] -->

## Abstract

We present a simple, highly modularized network architecture for image classification. Our network is constructed by repeating a building block that aggregates a set of transformations with the same topology. Our simple design results in a homogeneous, multi-branch architecture that has only a few hyper-parameters to set. This strategy exposes a new dimension, which we call "cardinality" (the size of the set of transformations), as an essential factor in addition to the dimensions of depth and width. On the ImageNet-1K dataset, we empirically show that even under the restricted condition of maintaining complexity, increasing cardinality is able to improve classification accuracy. Moreover, increasing cardinality is more effective than going deeper or wider when we increase the capacity. Our models, named ResNeXt, are the foundations of our entry to the ILSVRC 2016 classification task in which we secured 2nd place. We further investigate ResNeXt on an ImageNet-5K set and the COCO detection set, also showing better results than its ResNet counterpart. The code and models are publicly available online.

<div align=center>
<img src="https://user-images.githubusercontent.com/26739999/142574479-21fb00a2-e63e-4bc6-a9f2-989cd6e15528.png" width="70%"/>
</div>

## Results and models

### ImageNet-1k

|         Model         | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) | Config | Download |
|:---------------------:|:---------:|:--------:|:---------:|:---------:|:---------:|:--------:|
| ResNeXt-32x4d-50      | 25.03     | 4.27     | 77.90 | 93.66 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnext/resnext50-32x4d_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/resnext/resnext50_32x4d_b32x8_imagenet_20210429-56066e27.pth) &#124; [log](https://download.openmmlab.com/mmclassification/v0/resnext/resnext50_32x4d_b32x8_imagenet_20210429-56066e27.log.json) |
| ResNeXt-32x4d-101     | 44.18     | 8.03     | 78.61  | 94.17 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnext/resnext101-32x4d_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/resnext/resnext101_32x4d_b32x8_imagenet_20210506-e0fa3dd5.pth) &#124; [log](https://download.openmmlab.com/mmclassification/v0/resnext/resnext101_32x4d_b32x8_imagenet_20210506-e0fa3dd5.log.json) |
| ResNeXt-32x8d-101     | 88.79     | 16.5     | 79.27 | 94.58 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnext/resnext101-32x8d_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/resnext/resnext101_32x8d_b32x8_imagenet_20210506-23a247d5.pth) &#124; [log](https://download.openmmlab.com/mmclassification/v0/resnext/resnext101_32x8d_b32x8_imagenet_20210506-23a247d5.log.json) |
| ResNeXt-32x4d-152     | 59.95     | 11.8     | 78.88 | 94.33 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnext/resnext152-32x4d_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/resnext/resnext152_32x4d_b32x8_imagenet_20210524-927787be.pth) &#124; [log](https://download.openmmlab.com/mmclassification/v0/resnext/resnext152_32x4d_b32x8_imagenet_20210524-927787be.log.json) |

## Citation

```
@inproceedings{xie2017aggregated,
  title={Aggregated residual transformations for deep neural networks},
  author={Xie, Saining and Girshick, Ross and Doll{\'a}r, Piotr and Tu, Zhuowen and He, Kaiming},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={1492--1500},
  year={2017}
}
```
