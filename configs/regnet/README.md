# Designing Network Design Spaces
<!-- {RegNet} -->
<!-- [ALGORITHM] -->

## Abstract
<!-- [ABSTRACT] -->
In this work, we present a new network design paradigm. Our goal is to help advance the understanding of network design and discover design principles that generalize across settings. Instead of focusing on designing individual network instances, we design network design spaces that parametrize populations of networks. The overall process is analogous to classic manual design of networks, but elevated to the design space level. Using our methodology we explore the structure aspect of network design and arrive at a low-dimensional design space consisting of simple, regular networks that we call RegNet. The core insight of the RegNet parametrization is surprisingly simple: widths and depths of good networks can be explained by a quantized linear function. We analyze the RegNet design space and arrive at interesting findings that do not match the current practice of network design. The RegNet design space provides simple and fast networks that work well across a wide range of flop regimes. Under comparable training settings and flops, the RegNet models outperform the popular EfficientNet models while being up to 5x faster on GPUs.

<!-- [IMAGE] -->
<div align=center>
<img src="https://user-images.githubusercontent.com/26739999/142572813-5dad3317-9d58-4177-971f-d346e01fb3c4.png" width=60%/>
</div>

## Citation
```latex
@article{radosavovic2020designing,
    title={Designing Network Design Spaces},
    author={Ilija Radosavovic and Raj Prateek Kosaraju and Ross Girshick and Kaiming He and Piotr Dollár},
    year={2020},
    eprint={2003.13678},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## Pretrain model

The pre-trained modles are converted from [model zoo of pycls](https://github.com/facebookresearch/pycls/blob/master/MODEL_ZOO.md).

### ImageNet

|         Model         | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) | Download |
|:---------------------:|:---------:|:--------:|:---------:|:---------:|:--------:|
| RegNetX-400MF         |   5.16    |   0.41   |   72.55   |   90.91   | [model](https://download.openmmlab.com/mmclassification/v0/regnet/convert/RegNetX-400MF-0db9f35c.pth)|
| RegNetX-800MF         |   7.26    |   0.81   |   75.21   |   92.37   | [model](https://download.openmmlab.com/mmclassification/v0/regnet/convert/RegNetX-800MF-4f9d1e8a.pth)|
| RegNetX-1.6GF         |   9.19    |   1.63   |   77.04   |   93.51   | [model](https://download.openmmlab.com/mmclassification/v0/regnet/convert/RegNetX-1.6GF-cfb32375.pth)|
| RegNetX-3.2GF         |   15.3    |   3.21   |   78.26   |   94.20   | [model](https://download.openmmlab.com/mmclassification/v0/regnet/convert/RegNetX-3.2GF-82c43fd5.pth)|
| RegNetX-4.0GF         |   22.12   |   4.0    |   78.72   |   94.22   | [model](https://download.openmmlab.com/mmclassification/v0/regnet/convert/RegNetX-4.0GF-ef8bb32c.pth)|
| RegNetX-6.4GF         |   26.21   |   6.51   |   79.22   |   94.61   | [model](https://download.openmmlab.com/mmclassification/v0/regnet/convert/RegNetX-6.4GF-6888c0ea.pth)|
| RegNetX-8.0GF         |   39.57   |   8.03   |   79.31   |   94.57   | [model](https://download.openmmlab.com/mmclassification/v0/regnet/convert/RegNetX-8.0GF-cb4c77ec.pth)|
| RegNetX-12GF          |   46.11   |   12.15  |   79.91   |   94.78   | [model](https://download.openmmlab.com/mmclassification/v0/regnet/convert/RegNetX-12GF-0574538f.pth)|

## Results and models

Waiting for adding.
