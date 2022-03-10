# Visual Attention Network

> [Visual Attention Network](https://arxiv.org/pdf/2202.09741v2.pdf)
<!-- [ALGORITHM] -->

## Abstract

While originally designed for natural language processing (NLP) tasks, the self-attention mechanism has recently taken various computer vision areas by storm. However, the 2D nature of images brings three challenges for applying self-attention in computer vision. (1) Treating images as 1D sequences neglects their 2D structures. (2) The quadratic complexity is too expensive for high-resolution images. (3) It only captures spatial adaptability but ignores channel adaptability. In this paper, we propose a novel large kernel attention (LKA) module to enable self-adaptive and long-range correlations in self-attention while avoiding the above issues. We further introduce a novel neural network based on LKA, namely Visual Attention Network (VAN). While extremely simple and efficient, VAN outperforms the state-of-the-art vision transformers and convolutional neural networks with a large margin in extensive experiments, including image classification, object detection, semantic segmentation, instance segmentation, etc.

<div align=center>
<img src="https://user-images.githubusercontent.com/24734142/157409484-f26fcc1f-a856-48c2-a7a7-d157c38877ac.png" width="90%"/>
</div>

<div align=center>
<img src="https://user-images.githubusercontent.com/24734142/157409411-2f622ba7-553c-4702-91be-eba03f9ea04f.png" width="90%"/>
</div>


## Results and models

### ImageNet-1k

|   Model   |   Pretrain   | resolution  | Params(M) |  Flops(G) | Top-1 (%) | Top-5 (%) | Config | Download |
|:---------:|:------------:|:-----------:|:---------:|:---------:|:---------:|:---------:|:------:|:--------:|
|  VAN-T   | From scratch |   224x224   |   4.1   |    0.9   |   75.4   |      | [config](van-tiny-8xb128_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_tiny_224_b16x64_300e_imagenet_20210616_090925-66df6be6.pth)  &#124; [log](https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_tiny_224_b16x64_300e_imagenet_20210616_090925.log.json)|
|  VAN-S   | From scratch |   224x224   |   13.9   |    2.5   |   81.1   |      | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/swin_transformer/swin-small_16xb64_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_small_224_b16x64_300e_imagenet_20210615_110219-7f9d988b.pth)  &#124; [log](https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_small_224_b16x64_300e_imagenet_20210615_110219.log.json)|
|  VAN-B   | From scratch |   224x224   |   26.6   |   5.0   |   82.8   |      | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/swin_transformer/swin_base_224_b16x64_300e_imagenet.py) | [model](https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_base_224_b16x64_300e_imagenet_20210616_190742-93230b0d.pth)  &#124; [log](https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_base_224_b16x64_300e_imagenet_20210616_190742.log.json)|
|  VAN-L | From scratch |   224x224   |   44.8   |    9.0   |   83.9   |      | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/swin_transformer/swin-small_16xb64_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin_small_patch4_window7_224-cc7a01c9.pth) |

## Citation

```
@article{guo2022visual,
  title={Visual Attention Network},
  author={Guo, Meng-Hao and Lu, Cheng-Ze and Liu, Zheng-Ning and Cheng, Ming-Ming and Hu, Shi-Min},
  journal={arXiv preprint arXiv:2202.09741},
  year={2022}
}
```
