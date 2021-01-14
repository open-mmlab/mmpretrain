# MobileNetV2: Inverted Residuals and Linear Bottlenecks

## Introduction

[ALGORITHM]

```latex
@INPROCEEDINGS{8578572,
  author={M. {Sandler} and A. {Howard} and M. {Zhu} and A. {Zhmoginov} and L. {Chen}},
  booktitle={2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  title={MobileNetV2: Inverted Residuals and Linear Bottlenecks},
  year={2018},
  volume={},
  number={},
  pages={4510-4520},
  doi={10.1109/CVPR.2018.00474}}
}
```

## Results and models

### ImageNet

|         Model         | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) | Config | Download |
|:---------------------:|:---------:|:--------:|:---------:|:---------:|:---------:|:--------:|
| MobileNet V2          | 3.5       | 0.319    | 71.86 | 90.42 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/mobilenet_v2/mobilenet_v2_b32x8_imagenet.py) | [model](https://download.openmmlab.com/mmclassification/v0/mobilenet_v2/mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth) &#124; [log](https://download.openmmlab.com/mmclassification/v0/mobilenet_v2/mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.log.json) |
