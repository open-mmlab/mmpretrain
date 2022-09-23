# RegNet

> [Designing Network Design Spaces](https://arxiv.org/abs/2003.13678)

<!-- [ALGORITHM] -->

## Abstract

In this work, we present a new network design paradigm. Our goal is to help advance the understanding of network design and discover design principles that generalize across settings. Instead of focusing on designing individual network instances, we design network design spaces that parametrize populations of networks. The overall process is analogous to classic manual design of networks, but elevated to the design space level. Using our methodology we explore the structure aspect of network design and arrive at a low-dimensional design space consisting of simple, regular networks that we call RegNet. The core insight of the RegNet parametrization is surprisingly simple: widths and depths of good networks can be explained by a quantized linear function. We analyze the RegNet design space and arrive at interesting findings that do not match the current practice of network design. The RegNet design space provides simple and fast networks that work well across a wide range of flop regimes. Under comparable training settings and flops, the RegNet models outperform the popular EfficientNet models while being up to 5x faster on GPUs.

<div align=center>
<img src="https://user-images.githubusercontent.com/26739999/142572813-5dad3317-9d58-4177-971f-d346e01fb3c4.png" width=60%/>
</div>

## Results and models

### ImageNet-1k

|      Model      | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) |                                   Config                                   |                                   Download                                   |
| :-------------: | :-------: | :------: | :-------: | :-------: | :------------------------------------------------------------------------: | :--------------------------------------------------------------------------: |
|  RegNetX-400MF  |   5.16    |   0.41   |   72.56   |   90.78   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/regnet/regnetx-400mf_8xb128_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/regnet/regnetx-400mf_8xb128_in1k_20211213-89bfc226.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/regnet/regnetx-400mf_8xb128_in1k_20211208_143316.log.json) |
|  RegNetX-800MF  |   7.26    |   0.81   |   74.76   |   92.32   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/regnet/regnetx-800mf_8xb128_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/regnet/regnetx-800mf_8xb128_in1k_20211213-222b0f11.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/regnet/regnetx-800mf_8xb128_in1k_20211207_143037.log.json) |
|  RegNetX-1.6GF  |   9.19    |   1.63   |   76.84   |   93.31   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/regnet/regnetx-1.6gf_8xb128_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/regnet/regnetx-1.6gf_8xb128_in1k_20211213-d1b89758.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/regnet/regnetx-1.6gf_8xb128_in1k_20211208_143018.log.json) |
|  RegNetX-3.2GF  |   15.3    |   3.21   |   78.09   |   94.08   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/regnet/regnetx-3.2gf_8xb64_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/regnet/regnetx-3.2gf_8xb64_in1k_20211213-1fdd82ae.pth)  \| [log](https://download.openmmlab.com/mmclassification/v0/regnet/regnetx-3.2gf_8xb64_in1k_20211208_142720.log.json) |
|  RegNetX-4.0GF  |   22.12   |   4.0    |   78.60   |   94.17   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/regnet/regnetx-4.0gf_8xb64_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/regnet/regnetx-4.0gf_8xb64_in1k_20211213-efed675c.pth)  \| [log](https://download.openmmlab.com/mmclassification/v0/regnet/regnetx-4.0gf_8xb64_in1k_20211207_150431.log.json) |
|  RegNetX-6.4GF  |   26.21   |   6.51   |   79.38   |   94.65   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/regnet/regnetx-6.4gf_8xb64_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/regnet/regnetx-6.4gf_8xb64_in1k_20211215-5c6089da.pth)  \| [log](https://download.openmmlab.com/mmclassification/v0/regnet/regnetx-6.4gf_8xb64_in1k_20211213_172748.log.json) |
|  RegNetX-8.0GF  |   39.57   |   8.03   |   79.12   |   94.51   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/regnet/regnetx-8.0gf_8xb64_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/regnet/regnetx-8.0gf_8xb64_in1k_20211213-9a9fcc76.pth)  \| [log](https://download.openmmlab.com/mmclassification/v0/regnet/regnetx-8.0gf_8xb64_in1k_20211208_103250.log.json) |
|  RegNetX-12GF   |   46.11   |  12.15   |   79.67   |   95.03   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/regnet/regnetx-12gf_8xb64_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/regnet/regnetx-12gf_8xb64_in1k_20211213-5df8c2f8.pth)  \| [log](https://download.openmmlab.com/mmclassification/v0/regnet/regnetx-12gf_8xb64_in1k_20211208_143713.log.json) |
| RegNetX-400MF\* |   5.16    |   0.41   |   72.55   |   90.91   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/regnet/regnetx-400mf_8xb128_in1k) | [model](https://download.openmmlab.com/mmclassification/v0/regnet/convert/RegNetX-400MF-0db9f35c.pth) |
| RegNetX-800MF\* |   7.26    |   0.81   |   75.21   |   92.37   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/regnet/regnetx-800mf_8xb128_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/regnet/convert/RegNetX-800MF-4f9d1e8a.pth) |
| RegNetX-1.6GF\* |   9.19    |   1.63   |   77.04   |   93.51   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/regnet/regnetx-1.6gf_8xb128_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/regnet/convert/RegNetX-1.6GF-cfb32375.pth) |
| RegNetX-3.2GF\* |   15.3    |   3.21   |   78.26   |   94.20   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/regnet/regnetx-3.2gf_8xb64_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/regnet/convert/RegNetX-3.2GF-82c43fd5.pth) |
| RegNetX-4.0GF\* |   22.12   |   4.0    |   78.72   |   94.22   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/regnet/regnetx-4.0gf_8xb64_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/regnet/convert/RegNetX-4.0GF-ef8bb32c.pth) |
| RegNetX-6.4GF\* |   26.21   |   6.51   |   79.22   |   94.61   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/regnet/regnetx-6.4gf_8xb64_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/regnet/convert/RegNetX-6.4GF-6888c0ea.pth) |
| RegNetX-8.0GF\* |   39.57   |   8.03   |   79.31   |   94.57   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/regnet/regnetx-8.0gf_8xb64_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/regnet/convert/RegNetX-8.0GF-cb4c77ec.pth) |
| RegNetX-12GF\*  |   46.11   |  12.15   |   79.91   |   94.78   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/regnet/regnetx-12gf_8xb64_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/regnet/convert/RegNetX-12GF-0574538f.pth) |

*Models with * are converted from [pycls](https://github.com/facebookresearch/pycls/blob/master/MODEL_ZOO.md). The config files of these models are only for validation.*

## Citation

```
@article{radosavovic2020designing,
    title={Designing Network Design Spaces},
    author={Ilija Radosavovic and Raj Prateek Kosaraju and Ross Girshick and Kaiming He and Piotr Dollár},
    year={2020},
    eprint={2003.13678},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
