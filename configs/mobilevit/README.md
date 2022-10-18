# MobileVit

> [MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer](https://arxiv.org/abs/2110.02178)

<!-- [ALGORITHM] -->

## Abstract

Light-weight convolutional neural networks (CNNs) are the de-facto for mobile vision tasks. Their spatial inductive biases allow them to learn representations with fewer parameters across different vision tasks. However, these networks are spatially local. To learn global representations, self-attention-based vision trans-formers (ViTs) have been adopted. Unlike CNNs, ViTs are heavy-weight. In this paper, we ask the following question: is it possible to combine the strengths of CNNs and ViTs to build a light-weight and low latency network for mobile vision tasks? Towards this end, we introduce MobileViT, a light-weight and general-purpose vision transformer for mobile devices. MobileViT presents a different perspective for the global processing of information with transformers, i.e., transformers as convolutions. Our results show that MobileViT significantly outperforms CNN- and ViT-based networks across different tasks and datasets. On the ImageNet-1k dataset, MobileViT achieves top-1 accuracy of 78.4% with about 6 million parameters, which is 3.2% and 6.2% more accurate than MobileNetv3 (CNN-based) and DeIT (ViT-based) for a similar number of parameters. On the MS-COCO object detection task, MobileViT is 5.7% more accurate than MobileNetv3 for a similar number of parameters.

<div align=center>
<img src="https://user-images.githubusercontent.com/42952108/193229983-822bf025-89a6-4d95-b6be-76b7f1a62f2c.png" width="70%"/>
</div>

## Results and models

### ImageNet-1k

|        Model        | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) |                    Config                    |                                                Download                                                |
| :-----------------: | :-------: | :------: | :-------: | :-------: | :------------------------------------------: | :----------------------------------------------------------------------------------------------------: |
| MobileViT-XXSmall\* |   1.27    |   0.42   |   69.02   |   88.91   | [config](./mobilevit-xxsmall_8xb128_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/mobilevit/mobilevit-xxsmall_3rdparty_in1k_20221018-77835605.pth) |
| MobileViT-XSmall\*  |   2.32    |   1.05   |   74.75   |   92.32   | [config](./mobilevit-xsmall_8xb128_in1k.py)  | [model](https://download.openmmlab.com/mmclassification/v0/mobilevit/mobilevit-xsmall_3rdparty_in1k_20221018-be39a6e7.pth) |
|  MobileViT-Small\*  |   5.58    |   2.03   |   78.25   |   94.09   |  [config](./mobilevit-small_8xb128_in1k.py)  | [model](https://download.openmmlab.com/mmclassification/v0/mobilevit/mobilevit-small_3rdparty_in1k_20221018-cb4f741c.pth) |

*Models with * are converted from [ml-cvnets](https://github.com/apple/ml-cvnets). The config files of these models are only for validation. We don't ensure these config files' training accuracy and welcome you to contribute your reproduction results.*

## Citation

```
@article{mehta2021mobilevit,
  title={MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer},
  author={Mehta, Sachin and Rastegari, Mohammad},
  journal={arXiv preprint arXiv:2110.02178},
  year={2021}
}
```
