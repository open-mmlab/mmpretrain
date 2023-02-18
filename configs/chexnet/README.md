# CheXNet

> [CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning](https://arxiv.org/abs/1711.05225v3)

<!-- [ALGORITHM] -->

## Abstract

We develop an algorithm that can detect pneumonia from chest X-rays at a level exceeding practicing radiologists. Our algorithm, CheXNet, is a 121-layer convolutional neural network trained on ChestX-ray14, currently the largest publicly available chest X-ray dataset, containing over 100,000 frontal-view X-ray images with 14 diseases. Four practicing academic radiologists annotate a test set, on which we compare the performance of CheXNet to that of radiologists. We find that CheXNet exceeds average radiologist performance on the F1 metric. We extend CheXNet to detect all 14 diseases in ChestX-ray14 and achieve state of the art results on all 14 diseases.

<div align="center">
<img src="https://user-images.githubusercontent.com/24734142/213636097-45de1275-ffbe-4d67-90ba-4b8aae274510.png" width="70%"/>
</div>

## Results and models

### ImageNet-1k

| Model                                      |  Pretrain   | Params(M) | Flops(G) | ROCAUC |                  Config                   |  Download   |
| :----------------------------------------- | :---------: | :-------: | :------: | :----: | :---------------------------------------: | :---------: |
| CheXNet (`chexnet_1xb16_nih_chestxrays`)\* | ImageNet-1k |           |          |        | [config](./eva-g-p14_8xb16_in1k-336px.py) | [model](<>) |
