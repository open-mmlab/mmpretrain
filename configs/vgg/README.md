# Very Deep Convolutional Networks for Large-Scale Image Recognition

## Introduction

<!-- [ALGORITHM] -->

```latex
@article{simonyan2014very,
  title={Very deep convolutional networks for large-scale image recognition},
  author={Simonyan, Karen and Zisserman, Andrew},
  journal={arXiv preprint arXiv:1409.1556},
  year={2014}
}

```

## Results and models

### ImageNet

|         Model         | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) | Config | Download |
|:---------------------:|:---------:|:--------:|:---------:|:---------:|:---------:|:--------:|
| VGG-11 | 132.86 | 7.63 | 68.75 | 88.87 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/vgg/vgg11_b32x8_imagenet.py) | [model](https://download.openmmlab.com/mmclassification/v0/vgg/vgg11_batch256_imagenet_20210208-4271cd6c.pth) &#124; [log](https://download.openmmlab.com/mmclassification/v0/vgg/vgg11_batch256_imagenet_20210208-4271cd6c.log.json) |
| VGG-13 | 133.05 | 11.34 | 70.02 | 89.46 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/vgg/vgg13_b32x8_imagenet.py) | [model](https://download.openmmlab.com/mmclassification/v0/vgg/vgg13_batch256_imagenet_20210208-4d1d6080.pth) &#124; [log](https://download.openmmlab.com/mmclassification/v0/vgg/vgg13_batch256_imagenet_20210208-4d1d6080.log.json) |
| VGG-16 | 138.36 | 15.5 | 71.62 | 90.49 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/vgg/vgg16_b32x8_imagenet.py) | [model](https://download.openmmlab.com/mmclassification/v0/vgg/vgg16_batch256_imagenet_20210208-db26f1a5.pth) &#124; [log](https://download.openmmlab.com/mmclassification/v0/vgg/vgg16_batch256_imagenet_20210208-db26f1a5.log.json) |
| VGG-19 | 143.67 | 19.67 | 72.41 | 90.80 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/vgg/vgg19_b32x8_imagenet.py) | [model](https://download.openmmlab.com/mmclassification/v0/vgg/vgg19_batch256_imagenet_20210208-e6920e4a.pth) &#124; [log](https://download.openmmlab.com/mmclassification/v0/vgg/vgg19_batch256_imagenet_20210208-e6920e4a.log.json)|
| VGG-11-BN | 132.87 | 7.64 | 70.75 | 90.12 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/vgg/vgg11bn_b32x8_imagenet.py) | [model](https://download.openmmlab.com/mmclassification/v0/vgg/vgg11_bn_batch256_imagenet_20210207-f244902c.pth) &#124; [log](https://download.openmmlab.com/mmclassification/v0/vgg/vgg11_bn_batch256_imagenet_20210207-f244902c.log.json) |
| VGG-13-BN | 133.05 | 11.36 | 72.15 | 90.71 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/vgg/vgg13bn_b32x8_imagenet.py) | [model](https://download.openmmlab.com/mmclassification/v0/vgg/vgg13_bn_batch256_imagenet_20210207-1a8b7864.pth) &#124; [log](https://download.openmmlab.com/mmclassification/v0/vgg/vgg13_bn_batch256_imagenet_20210207-1a8b7864.log.json) |
| VGG-16-BN | 138.37 | 15.53 | 73.72 | 91.68 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/vgg/vgg16_b32x8_imagenet.py) | [model](https://download.openmmlab.com/mmclassification/v0/vgg/vgg16_bn_batch256_imagenet_20210208-7e55cd29.pth) &#124; [log](https://download.openmmlab.com/mmclassification/v0/vgg/vgg16_bn_batch256_imagenet_20210208-7e55cd29.log.json) |
| VGG-19-BN | 143.68 | 19.7 | 74.70 | 92.24 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/vgg/vgg19bn_b32x8_imagenet.py) | [model](https://download.openmmlab.com/mmclassification/v0/vgg/vgg19_bn_batch256_imagenet_20210208-da620c4f.pth) &#124; [log](https://download.openmmlab.com/mmclassification/v0/vgg/vgg19_bn_batch256_imagenet_20210208-da620c4f.log.json)|
