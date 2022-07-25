# Swin Transformer V2

> [Swin Transformer V2: Scaling Up Capacity and Resolution](https://arxiv.org/abs/2111.09883.pdf)

<!-- [ALGORITHM] -->

## Abstract

Large-scale NLP models have been shown to significantly improve the performance on language tasks with no signs of saturation. They also demonstrate amazing few-shot capabilities like that of human beings. This paper aims to explore large-scale models in computer vision. We tackle three major issues in training and application of large vision models, including training instability, resolution gaps between pre-training and fine-tuning, and hunger on labelled data. Three main techniques are proposed: 1) a residual-post-norm method combined with cosine attention to improve training stability; 2) A log-spaced continuous position bias method to effectively transfer models pre-trained using low-resolution images to downstream tasks with high-resolution inputs; 3) A self-supervised pre-training method, SimMIM, to reduce the needs of vast labeled images. Through these techniques, this paper successfully trained a 3 billion-parameter Swin Transformer V2 model, which is the largest dense vision model to date, and makes it capable of training with images of up to 1,536×1,536 resolution. It set new performance records on 4 representative vision tasks, including ImageNet-V2 image classification, COCO object detection, ADE20K semantic segmentation, and Kinetics-400 video action classification. Also note our training is much more efficient than that in Google's billion-level visual models, which consumes 40 times less labelled data and 40 times less training time.

<div align=center>
<img src="https://user-images.githubusercontent.com/42952108/180748696-ee7ed23d-7fee-4ccf-9eb5-f117db228a42.png" width="100%"/>
</div>

## Results and models

### ImageNet-21k

The pre-trained models on ImageNet-21k are used to fine-tune, and therefore don't have evaluation results.

| Model  | resolution | Params(M) | Flops(G) |                                                        Download                                                         |
| :----: | :--------: | :-------: | :------: | :---------------------------------------------------------------------------------------------------------------------: |
| Swin-B |  224x224   |   86.74   |  15.14   |    [model](https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin-base_3rdparty_in21k.pth)    |
| Swin-B |  384x384   |   86.88   |  44.49   | [model](https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin-base_3rdparty_in21k-384px.pth) |
| Swin-L |  224x224   |  195.00   |  34.04   |   [model](https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin-large_3rdparty_in21k.pth)    |
| Swin-L |  384x384   |  195.20   |  100.04  | [model](https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin-base_3rdparty_in21k-384px.pth) |

### ImageNet-1k

|  Model   |   Pretrain   | resolution | window | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) |                             Config                              |                              Download                              |
| :------: | :----------: | :--------: | :----: | :-------: | :------: | :-------: | :-------: | :-------------------------------------------------------------: | :----------------------------------------------------------------: |
| Swin-T\* | From scratch |  256x256   |  8x8   |   28.29   |   4.36   |   81.84   |   95.94   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/swin_transformer_v2/swin-tiny_16xb64_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_tiny_224_b16x64_300e_imagenet_20210616_090925-66df6be6.pth) |
| Swin-T\* | From scratch |  256x256   | 16x16  |   28.29   |   4.36   |   82.83   |   96.29   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/swin_transformer_v2/swin-tiny_16xb64_window16_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_tiny_224_b16x64_300e_imagenet_20210616_090925-66df6be6.pth) |
| Swin-S\* | From scratch |  256x256   |  8x8   |   28.29   |   4.36   |   83.76   |   96.66   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/swin_transformer_v2/swin-small_16xb64_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_tiny_224_b16x64_300e_imagenet_20210616_090925-66df6be6.pth) |
| Swin-S\* | From scratch |  256x256   | 16x16  |   28.29   |   4.36   |   84.32   |   96.89   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/swin_transformer_v2/swin-small_window16_16xb64_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_tiny_224_b16x64_300e_imagenet_20210616_090925-66df6be6.pth) |
| Swin-B\* | From scratch |  256x256   |  8x8   |   28.29   |   4.36   |   84.28   |   96.94   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/swin_transformer_v2/swin-base_16xb64_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_tiny_224_b16x64_300e_imagenet_20210616_090925-66df6be6.pth) |
| Swin-B\* | From scratch |  256x256   | 16x16  |   28.29   |   4.36   |   84.75   |   97.07   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/swin_transformer_v2/swin-base_16xb64_window16_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_tiny_224_b16x64_300e_imagenet_20210616_090925-66df6be6.pth) |
| Swin-B\* | ImageNet-21k |  256x256   | 16x16  |   28.29   |   4.36   |   86.32   |   98.04   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/swin_transformer_v2/swin-base_16xb64_window16_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_tiny_224_b16x64_300e_imagenet_20210616_090925-66df6be6.pth) |
| Swin-B\* | ImageNet-21k |  384x384   | 24x24  |   28.29   |   4.36   |   87.14   |   98.23   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/swin_transformer_v2/swin-base_16xb64_window16_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_tiny_224_b16x64_300e_imagenet_20210616_090925-66df6be6.pth) |
| Swin-L\* | ImageNet-21k |  256X256   | 16x16  |   28.29   |   4.36   |   87.13   |   98.15   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/swin_transformer_v2/swin-base_16xb64_window16_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_tiny_224_b16x64_300e_imagenet_20210616_090925-66df6be6.pth) |
| Swin-L\* | ImageNet-21k |  384x384   | 24x24  |   28.29   |   4.36   |   87.59   |   98.27   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/swin_transformer_v2/swin-base_16xb64_window16_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_tiny_224_b16x64_300e_imagenet_20210616_090925-66df6be6.pth) |

*Models with * are converted from the [official repo](https://github.com/microsoft/Swin-Transformer#main-results-on-imagenet-with-pretrained-models). The config files of these models are only for validation. We don't ensure these config files' training accuracy and welcome you to contribute your reproduction results.*

*ImageNet-21k pretrained models with input resolution of 256x256 and 384x384 both fine-tuned from the same pre-training model using a smaller input resolution of 192x192.*

## Citation

```
@article{https://doi.org/10.48550/arxiv.2111.09883,
  doi = {10.48550/ARXIV.2111.09883},
  url = {https://arxiv.org/abs/2111.09883},
  author = {Liu, Ze and Hu, Han and Lin, Yutong and Yao, Zhuliang and Xie, Zhenda and Wei, Yixuan and Ning, Jia and Cao, Yue and Zhang, Zheng and Dong, Li and Wei, Furu and Guo, Baining},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Swin Transformer V2: Scaling Up Capacity and Resolution},
  publisher = {arXiv},
  year = {2021},
  copyright = {Creative Commons Attribution 4.0 International}
}
```
