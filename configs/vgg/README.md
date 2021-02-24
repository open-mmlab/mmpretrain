# Very Deep Convolutional Networks for Large-Scale Image Recognition

## Introduction

[ALGORITHM]

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
| VGG-11 | 132.86 | 7.63 | 69.03 | 88.63 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/vgg/vgg11_imagenet.py) | [model](https://download.openmmlab.com/mmclassification/v0/vgg/vgg11_imagenet-01ecd97e.pth)* |
| VGG-13 | 133.05 | 11.34 | 69.93 | 89.26 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/vgg/vgg13_imagenet.py) | [model](https://download.openmmlab.com/mmclassification/v0/vgg/vgg13_imagenet-9ad3945d.pth)*|
| VGG-16 | 138.36 | 15.5 | 71.59 | 90.39 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/vgg/vgg16_imagenet.py) | [model](https://download.openmmlab.com/mmclassification/v0/vgg/vgg16_imagenet-91b6d117.pth)*|
| VGG-19 | 143.67 | 19.67 | 72.38 | 90.88 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/vgg/vgg19_imagenet.py) | [model](https://download.openmmlab.com/mmclassification/v0/vgg/vgg19_imagenet-fee352a8.pth)*|
| VGG-11-BN | 132.87 | 7.64 | 70.37 | 89.81 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/vgg/vgg11bn_imagenet.py) | [model](https://download.openmmlab.com/mmclassification/v0/vgg/vgg11_bn_imagenet-6fbbbf3f.pth)*|
| VGG-13-BN | 133.05 | 11.36 | 71.55 | 90.37 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/vgg/vgg13bn_imagenet.py) | [model](https://download.openmmlab.com/mmclassification/v0/vgg/vgg13_bn_imagenet-4b5f9390.pth)*|
| VGG-16-BN | 138.37 | 15.53 | 73.36 | 91.5 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/vgg/vgg16_imagenet.py) | [model](https://download.openmmlab.com/mmclassification/v0/vgg/vgg16_bn_imagenet-3ac6d8fd.pth)*|
| VGG-19-BN | 143.68 | 19.7 | 74.24 | 91.84 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/vgg/vgg19bn_imagenet.py) | [model](https://download.openmmlab.com/mmclassification/v0/vgg/vgg19_bn_imagenet-7c058385.pth)*|
