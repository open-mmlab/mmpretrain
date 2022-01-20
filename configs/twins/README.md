# Twins: Revisiting the Design of Spatial Attention in Vision Transformers

## Introduction

<!-- [ALGORITHM] -->

<a href = "https://github.com/Meituan-AutoML/Twins">Official Repo</a>

<a href="https://github.com/open-mmlab/mmclassification/blob/master/mmcls/models/backbones/twins.py">Code Snippet</a>

## Abstract

Very recently, a variety of vision transformer architectures for dense prediction tasks have been proposed and they show that the design of spatial attention is critical to their success in these tasks. In this work, we revisit the design of the spatial attention and demonstrate that a carefully-devised yet simple spatial attention mechanism performs favourably against the state-of-the-art schemes. As a result, we propose two vision transformer architectures, namely, Twins-PCPVT and Twins-SVT. Our proposed architectures are highly-efficient and easy to implement, only involving matrix multiplications that are highly optimized in modern deep learning frameworks. More importantly, the proposed architectures achieve excellent performance on a wide range of visual tasks, including image level classification as well as dense detection and segmentation. The simplicity and strong performance suggest that our proposed architectures may serve as stronger backbones for many vision tasks. Our code is released at [this https URL](https://github.com/Meituan-AutoML/Twins).

<!-- [IMAGE] -->
<div align=center>
<img src="https://user-images.githubusercontent.com/24582831/145021310-57826cf5-5e03-4c7c-9081-ffa744bdae27.png" width="80%"/>
</div>

<details>
<summary align = "right"> <a href = "https://arxiv.org/pdf/2104.13840.pdf" >Twins (NeurIPS'2021)</a></summary>

```latex
@article{chu2021twins,
  title={Twins: Revisiting spatial attention design in vision transformers},
  author={Chu, Xiangxiang and Tian, Zhi and Wang, Yuqing and Zhang, Bo and Ren, Haibing and Wei, Xiaolin and Xia, Huaxia and Shen, Chunhua},
  journal={arXiv preprint arXiv:2104.13840},
  year={2021}altgvt
}
```

</details>

## Usage

To use other repositories' pre-trained models, it is necessary to convert keys.

We provide a script [`twins2mmcls.py`](https://github.com/open-mmlab/mmclassification/tree/master/tools/convert_models/twins2mmcls.py) in the tools directory to convert the key of models from [the official repo](https://github.com/Meituan-AutoML/Twins) to MMClassification style.

```shell
python tools/model_converters/twins2mmcls.py ${PRETRAIN_PATH} ${STORE_PATH}
```

This script convert `pcpvt` or `svt` pretrained model from `PRETRAIN_PATH` and store the converted model in `STORE_PATH`.

For example,

```shell
python tools/model_converters/twins2mmcls.py ./alt_gvt_base.pth ./twins_alt_gvt_base.pth
```

## Results and models

### ImageNet-1k

|      Model     | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) | Config | Download |
|:--------------:|:---------:|:--------:|:---------:|:---------:|:------:|:--------:|
|   PCPVT-small  |   24.11   |   3.67   |   81.14   |   95.69   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/twins/twins-pcpvt-small_8xb128_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/twins/twins-pcpvt-small_3rdparty_8xb128_in1k_20220120-9f87e819.pth) |
|   PCPVT-base   |   43.83   |   6.45   |   82.66   |   96.26   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/twins/twins-pcpvt-base_8xb128_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/twins/twins-pcpvt-base_3rdparty_8xb128_in1k_20220120-27a7e6d7.pth) |
|   PCPVT-large  |   60.99   |   9.51   |   83.09   |   96.59   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/twins/twins-pcpvt-large_16xb64_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/twins/twins-pcpvt-large_3rdparty_16xb64_in1k_20220120-6f613882.pth) |
|     SVT-small  |   24.06   |   2.82   |   81.77   |   95.57   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/twins/twins-svt-small_8xb128_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/twins/twins-svt-small_3rdparty_8xb128_in1k_20220120-1667a72a.pth) |
|     SVT-base   |   56.07   |   8.35   |   83.13   |   96.29   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/twins/twins-svt-base_8xb128_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/twins/twins-svt-base_3rdparty_8xb128_in1k_20220120-2ecf4da4.pth)  |
|    SVT-large   |   99.27   |   14.82  |   83.60   |   96.50   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/twins/twins-svt-large_16xb64_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/twins/twins-svt-large_3rdparty_16xb64_in1k_20220120-2861aed1.pth) |

*Models with \* are converted from [the official repo](https://github.com/Meituan-AutoML/Twins). The config files of these models are only for validation. We don't ensure these config files' training accuracy and welcome you to contribute your reproduction results. The validation accuracy is a little different from the official paper because of the PyTorch version. This result is get in PyTorch=1.9 while the official result is get in PyTorch=1.7*
