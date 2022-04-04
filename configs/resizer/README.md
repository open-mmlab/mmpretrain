# Learning to Resize Images for Computer Vision Tasks

> [Learning to Resize Images for Computer Vision Tasks](https://arxiv.org/pdf/2103.09950.pdf)
<!-- [ALGORITHM] -->

## Abstract

For all the ways convolutional neural nets have revolutionized computer vision in recent years, one important aspect has received surprisingly little attention: the effect of image size on the accuracy of tasks being trained for. Typically, to be efficient, the input images are resized to a relatively small spatial resolution (e.g. 224x224), and both training and inference are carried out at this resolution. The actual mechanism for this re-scaling has been an afterthought: Namely, off-the-shelf image resizers such as bilinear and bicubic are commonly used in most machine learning software frameworks. But do these resizers limit the on task performance of the trained networks? The answer is yes. Indeed, we show that the typical linear resizer can be replaced with learned resizers that can substantially improve performance. Importantly, while the classical resizers typically result in better perceptual quality of the downscaled images, our proposed learned resizers do not necessarily give better visual quality, but instead improve task performance. Our learned image resizer is jointly trained with a baseline vision model. This learned CNN-based resizer creates machine friendly visual manipulations that lead to a consistent improvement of the end task metric over the baseline model. Specifically, here we focus on the classification task with the ImageNet dataset, and experiment with four different models to learn resizers adapted to each model. Moreover, we show that the proposed resizer can also be useful for fine-tuning the classification baselines for other vision tasks. To this end, we experiment with three different baselines to develop image quality assessment (IQA) models on the AVA dataset.

<div align=center>
<img src="https://user-images.githubusercontent.com/26739999/142576715-14668c6b-5cb8-4de8-ac51-419fae773c90.png" width="90%"/>
</div>

## Results and models
### CUB-200-2011

|       Model      |  Pretrain   | resolution  | Params(M) | Flops(G) | Top-1 (%) |  Config | Download |
|:----------------:|:------------:|:---------:|:---------:|:--------:|:---------:|:---------:|:---------:|
|      Swin-L      |  [ImageNet-21k](https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin-base_3rdparty_in21k-384px.pth) |   768x768   |     |      |   91.06   |  [config](resizerx2_swin-large_1xb8_cub_384px.py) |  |
