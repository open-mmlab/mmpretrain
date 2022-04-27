# DenseNet

> [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)
<!-- [ALGORITHM] -->

## Abstract

Recent work has shown that convolutional networks can be substantially deeper, more accurate, and efficient to train if they contain shorter connections between layers close to the input and those close to the output. In this paper, we embrace this observation and introduce the Dense Convolutional Network (DenseNet), which connects each layer to every other layer in a feed-forward fashion. Whereas traditional convolutional networks with L layers have L connections - one between each layer and its subsequent layer - our network has L(L+1)/2 direct connections. For each layer, the feature-maps of all preceding layers are used as inputs, and its own feature-maps are used as inputs into all subsequent layers. DenseNets have several compelling advantages: they alleviate the vanishing-gradient problem, strengthen feature propagation, encourage feature reuse, and substantially reduce the number of parameters. We evaluate our proposed architecture on four highly competitive object recognition benchmark tasks (CIFAR-10, CIFAR-100, SVHN, and ImageNet). DenseNets obtain significant improvements over the state-of-the-art on most of them, whilst requiring less computation to achieve high performance.

<div align=center>
<img src="https://user-images.githubusercontent.com/42952108/162675098-9a670883-b13a-4a5a-a9c9-06c39c616a0a.png" width="100%"/>
</div>

## Results and models

### ImageNet-1k

|      Model      | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) | Config | Download |
|:---------------:|:---------:|:--------:|:---------:|:---------:|:------:|:--------:|
| DenseNet121\*   | 7.98      | 2.88     | 74.96     | 92.21     | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/densenet/densenet121_4xb256_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/densenet/densenet121_4xb256_in1k_20220426-07450f99.pth) |
| DenseNet169\*   | 14.15     | 3.42     | 76.08     | 93.11     | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/densenet/densenet169_4xb256_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/densenet/densenet169_4xb256_in1k_20220426-a2889902.pth) |
| DenseNet201\*   | 20.01     | 4.37     | 77.32     | 93.64     | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/densenet/densenet201_4xb256_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/densenet/densenet201_4xb256_in1k_20220426-05cae4ef.pth) |
| DenseNet161\*   | 28.68     | 7.82     | 77.61     | 93.83     | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/densenet/densenet161_4xb256_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/densenet/densenet161_4xb256_in1k_20220426-ee6a80a9.pth) |

*Models with \* are converted from [pytorch](https://pytorch.org/vision/stable/models.html), guided by [original repo](https://github.com/liuzhuang13/DenseNet). The config files of these models are only for inference. We don't ensure these config files' training accuracy and welcome you to contribute your reproduction results.*


## Citation

```bibtex
@misc{https://doi.org/10.48550/arxiv.1608.06993,
      doi = {10.48550/ARXIV.1608.06993},
      url = {https://arxiv.org/abs/1608.06993},
      author = {Huang, Gao and Liu, Zhuang and van der Maaten, Laurens and Weinberger, Kilian Q.},
      keywords = {Computer Vision and Pattern Recognition (cs.CV), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
      title = {Densely Connected Convolutional Networks},
      publisher = {arXiv},
      year = {2016},
      copyright = {arXiv.org perpetual, non-exclusive license}
}
```
