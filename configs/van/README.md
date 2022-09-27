# Visual Attention Network

> [Visual Attention Network](https://arxiv.org/pdf/2202.09741v2.pdf)

<!-- [ALGORITHM] -->

## Abstract

While originally designed for natural language processing (NLP) tasks, the self-attention mechanism has recently taken various computer vision areas by storm. However, the 2D nature of images brings three challenges for applying self-attention in computer vision. (1) Treating images as 1D sequences neglects their 2D structures. (2) The quadratic complexity is too expensive for high-resolution images. (3) It only captures spatial adaptability but ignores channel adaptability. In this paper, we propose a novel large kernel attention (LKA) module to enable self-adaptive and long-range correlations in self-attention while avoiding the above issues. We further introduce a novel neural network based on LKA, namely Visual Attention Network (VAN). While extremely simple and efficient, VAN outperforms the state-of-the-art vision transformers and convolutional neural networks with a large margin in extensive experiments, including image classification, object detection, semantic segmentation, instance segmentation, etc.

<div align=center>
<img src="https://user-images.githubusercontent.com/24734142/157409411-2f622ba7-553c-4702-91be-eba03f9ea04f.png" width="80%"/>
</div>

## Results and models

### ImageNet-1k

|  Model   |   Pretrain   | resolution | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) |                               Config                               |                               Download                                |
| :------: | :----------: | :--------: | :-------: | :------: | :-------: | :-------: | :----------------------------------------------------------------: | :-------------------------------------------------------------------: |
| VAN-B0\* | From scratch |  224x224   |   4.11    |   0.88   |   75.41   |   93.02   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/van/van-b0_8xb128_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/van/van-tiny_8xb128_in1k_20220501-385941af.pth) |
| VAN-B1\* | From scratch |  224x224   |   13.86   |   2.52   |   81.01   |   95.63   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/van/van-b1_8xb128_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/van/van-small_8xb128_in1k_20220501-17bc91aa.pth) |
| VAN-B2\* | From scratch |  224x224   |   26.58   |   5.03   |   82.80   |   96.21   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/van/van-b2_8xb128_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/van/van-base_8xb128_in1k_20220501-6a4cc31b.pth) |
| VAN-B3\* | From scratch |  224x224   |   44.77   |   8.99   |   83.86   |   96.73   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/van/van-b3_8xb128_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/van/van-large_8xb128_in1k_20220501-f212ba21.pth) |
| VAN-B4\* | From scratch |  224x224   |   60.28   |  12.22   |   84.13   |   96.86   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/van/van-b4_8xb128_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/van/van-b4_3rdparty_in1k_20220909-f4665b92.pth) |

\*Models with * are converted from [the official repo](https://github.com/Visual-Attention-Network/VAN-Classification). The config files of these models are only for validation. We don't ensure these config files' training accuracy and welcome you to contribute your reproduction results.

### Pre-trained Models

The pre-trained models on ImageNet-21k are used to fine-tune on the downstream tasks.

|  Model   |   Pretrain   | resolution | Params(M) | Flops(G) |                                                  Download                                                   |
| :------: | :----------: | :--------: | :-------: | :------: | :---------------------------------------------------------------------------------------------------------: |
| VAN-B4\* | ImageNet-21k |  224x224   |   60.28   |  12.22   | [model](https://download.openmmlab.com/mmclassification/v0/van/van-b4_3rdparty_in21k_20220909-db926b18.pth) |
| VAN-B5\* | ImageNet-21k |  224x224   |   89.97   |  17.21   | [model](https://download.openmmlab.com/mmclassification/v0/van/van-b5_3rdparty_in21k_20220909-18e904e3.pth) |
| VAN-B6\* | ImageNet-21k |  224x224   |   283.9   |  55.28   | [model](https://download.openmmlab.com/mmclassification/v0/van/van-b6_3rdparty_in21k_20220909-96c2cb3a.pth) |

\*Models with * are converted from [the official repo](https://github.com/Visual-Attention-Network/VAN-Classification).

## Citation

```
@article{guo2022visual,
  title={Visual Attention Network},
  author={Guo, Meng-Hao and Lu, Cheng-Ze and Liu, Zheng-Ning and Cheng, Ming-Ming and Hu, Shi-Min},
  journal={arXiv preprint arXiv:2202.09741},
  year={2022}
}
```
