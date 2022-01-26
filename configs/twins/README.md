# Twins

> [Twins: Revisiting the Design of Spatial Attention in Vision Transformers](http://arxiv-export-lb.library.cornell.edu/abs/2104.13840)
<!-- [ALGORITHM] -->

## Abstract

Very recently, a variety of vision transformer architectures for dense prediction tasks have been proposed and they show that the design of spatial attention is critical to their success in these tasks. In this work, we revisit the design of the spatial attention and demonstrate that a carefully-devised yet simple spatial attention mechanism performs favourably against the state-of-the-art schemes. As a result, we propose two vision transformer architectures, namely, Twins-PCPVT and Twins-SVT. Our proposed architectures are highly-efficient and easy to implement, only involving matrix multiplications that are highly optimized in modern deep learning frameworks. More importantly, the proposed architectures achieve excellent performance on a wide range of visual tasks, including image level classification as well as dense detection and segmentation. The simplicity and strong performance suggest that our proposed architectures may serve as stronger backbones for many vision tasks. Our code is released at [this https URL](https://github.com/Meituan-AutoML/Twins).

<div align=center>
<img src="https://user-images.githubusercontent.com/24582831/145021310-57826cf5-5e03-4c7c-9081-ffa744bdae27.png" width="80%"/>
</div>

## Results and models

### ImageNet-1k

|      Model     | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) | Config | Download |
|:--------------:|:---------:|:--------:|:---------:|:---------:|:------:|:--------:|
|   PCPVT-small\*  |   24.11   |   3.67   |   81.14   |   95.69   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/twins/twins-pcpvt-small_8xb128_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/twins/twins-pcpvt-small_3rdparty_8xb128_in1k_20220126-ef23c132.pth) |
|   PCPVT-base\*   |   43.83   |   6.45   |   82.66   |   96.26   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/twins/twins-pcpvt-base_8xb128_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/twins/twins-pcpvt-base_3rdparty_8xb128_in1k_20220126-f8c4b0d5.pth) |
|   PCPVT-large\*  |   60.99   |   9.51   |   83.09   |   96.59   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/twins/twins-pcpvt-large_16xb64_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/twins/twins-pcpvt-large_3rdparty_16xb64_in1k_20220126-c1ef8d80.pth) |
|     SVT-small\*  |   24.06   |   2.82   |   81.77   |   95.57   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/twins/twins-svt-small_8xb128_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/twins/twins-svt-small_3rdparty_8xb128_in1k_20220126-8fe5205b.pth) |
|     SVT-base\*   |   56.07   |   8.35   |   83.13   |   96.29   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/twins/twins-svt-base_8xb128_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/twins/twins-svt-base_3rdparty_8xb128_in1k_20220126-e31cc8e9.pth)  |
|    SVT-large\*   |   99.27   |   14.82  |   83.60   |   96.50   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/twins/twins-svt-large_16xb64_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/twins/twins-svt-large_3rdparty_16xb64_in1k_20220126-4817645f.pth) |

*Models with \* are converted from [the official repo](https://github.com/Meituan-AutoML/Twins). The config files of these models are only for validation. We don't ensure these config files' training accuracy and welcome you to contribute your reproduction results. The validation accuracy is a little different from the official paper because of the PyTorch version. This result is get in PyTorch=1.9 while the official result is get in PyTorch=1.7*

## Citation

```
@article{chu2021twins,
  title={Twins: Revisiting spatial attention design in vision transformers},
  author={Chu, Xiangxiang and Tian, Zhi and Wang, Yuqing and Zhang, Bo and Ren, Haibing and Wei, Xiaolin and Xia, Huaxia and Shen, Chunhua},
  journal={arXiv preprint arXiv:2104.13840},
  year={2021}altgvt
}
```
