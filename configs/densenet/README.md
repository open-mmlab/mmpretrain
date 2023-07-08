# DenseNet

> [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)

<!-- [ALGORITHM] -->

## Abstract

Recent work has shown that convolutional networks can be substantially deeper, more accurate, and efficient to train if they contain shorter connections between layers close to the input and those close to the output. In this paper, we embrace this observation and introduce the Dense Convolutional Network (DenseNet), which connects each layer to every layer in a feed-forward fashion. Whereas traditional convolutional networks with L layers have L connections - one between each layer and its subsequent layer - our network has L(L+1)/2 direct connections. For each layer, the feature-maps of all preceding layers are used as inputs, and its own feature-maps are used as inputs into all subsequent layers. DenseNets have several compelling advantages: they alleviate the vanishing-gradient problem, strengthen feature propagation, encourage feature reuse, and substantially reduce the number of parameters. We evaluate our proposed architecture on four highly competitive object recognition benchmark tasks (CIFAR-10, CIFAR-100, SVHN, and ImageNet). DenseNets obtain significant improvements over the state-of-the-art on most of them, whilst requiring less computation to achieve high performance.

<div align=center>
<img src="https://user-images.githubusercontent.com/42952108/162675098-9a670883-b13a-4a5a-a9c9-06c39c616a0a.png" width="100%"/>
</div>

## How to use it?

<!-- [TABS-BEGIN] -->

**Predict image**

```python
from mmpretrain import inference_model

predict = inference_model('densenet121_3rdparty_in1k', 'demo/bird.JPEG')
print(predict['pred_class'])
print(predict['pred_score'])
```

**Use the model**

```python
import torch
from mmpretrain import get_model

model = get_model('densenet121_3rdparty_in1k', pretrained=True)
inputs = torch.rand(1, 3, 224, 224)
out = model(inputs)
print(type(out))
# To extract features.
feats = model.extract_feat(inputs)
print(type(feats))
```

**Test Command**

Prepare your dataset according to the [docs](https://mmpretrain.readthedocs.io/en/latest/user_guides/dataset_prepare.html#prepare-dataset).

Test:

```shell
python tools/test.py configs/densenet/densenet121_4xb256_in1k.py https://download.openmmlab.com/mmclassification/v0/densenet/densenet121_4xb256_in1k_20220426-07450f99.pth
```

<!-- [TABS-END] -->

## Models and results

### Image Classification on ImageNet-1k

| Model                         |   Pretrain   | Params (M) | Flops (G) | Top-1 (%) | Top-5 (%) |                Config                |                                        Download                                        |
| :---------------------------- | :----------: | :--------: | :-------: | :-------: | :-------: | :----------------------------------: | :------------------------------------------------------------------------------------: |
| `densenet121_3rdparty_in1k`\* | From scratch |    7.98    |   2.88    |   74.96   |   92.21   | [config](densenet121_4xb256_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/densenet/densenet121_4xb256_in1k_20220426-07450f99.pth) |
| `densenet169_3rdparty_in1k`\* | From scratch |   14.15    |   3.42    |   76.08   |   93.11   | [config](densenet169_4xb256_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/densenet/densenet169_4xb256_in1k_20220426-a2889902.pth) |
| `densenet201_3rdparty_in1k`\* | From scratch |   20.01    |   4.37    |   77.32   |   93.64   | [config](densenet201_4xb256_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/densenet/densenet201_4xb256_in1k_20220426-05cae4ef.pth) |
| `densenet161_3rdparty_in1k`\* | From scratch |   28.68    |   7.82    |   77.61   |   93.83   | [config](densenet161_4xb256_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/densenet/densenet161_4xb256_in1k_20220426-ee6a80a9.pth) |

*Models with * are converted from the [official repo](https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py). The config files of these models are only for inference. We haven't reproduce the training results.*

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
