# SimCLR

> [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)

<!-- [ALGORITHM] -->

## Abstract

This paper presents SimCLR: a simple framework for contrastive learning of visual representations. We simplify recently proposed contrastive self-supervised learning algorithms without requiring specialized architectures or a memory bank. In order to understand what enables the contrastive prediction tasks to learn useful representations, we systematically study the major components of our framework. We show that (1) composition of data augmentations plays a critical role in defining effective predictive tasks, (2) introducing a learnable nonlinear transformation between the representation and the contrastive loss substantially improves the quality of the learned representations, and (3) contrastive learning benefits from larger batch sizes and more training steps compared to supervised learning. By combining these findings, we are able to considerably outperform previous methods for self-supervised and semi-supervised learning on ImageNet. A linear classifier trained on self-supervised representations learned by SimCLR achieves 76.5% top-1 accuracy, which is a 7% relative improvement over previous state-of-the-art, matching the performance of a supervised ResNet-50.

<div align="center">
<img  src="https://user-images.githubusercontent.com/36138628/149723851-cf5f309e-d891-454d-90c0-e5337e5a11ed.png" width="400" />
</div>

## Results and Models

In this page, we provide benchmarks as much as possible to evaluate our pre-trained models. If not mentioned, all models are pre-trained on ImageNet-1k dataset.

### Classification

The classification benchmarks includes 4 downstream task datasets, **VOC**, **ImageNet**,  **iNaturalist2018** and **Places205**. If not specified, the results are Top-1 (%).

#### VOC SVM / Low-shot SVM

The **Best Layer** indicates that the best results are obtained from which layers feature map. For example, if the **Best Layer** is **feature3**, its best result is obtained from the second stage of ResNet (1 for stem layer, 2-5 for 4 stage layers).

Besides, k=1 to 96 indicates the hyper-parameter of Low-shot SVM.

| Self-Supervised Config                                                                                                                        | Best Layer | SVM   | k=1   | k=2   | k=4   | k=8   | k=16  | k=32  | k=64  | k=96 |
| --------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ---- |
| [resnet50_8xb32-coslr-200e](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/selfsup/simclr/simclr_resnet50_8xb32-coslr-200e_in1k.py) | feature5   | 79.98 | 35.02 | 42.79 | 54.87 | 61.91 | 67.38 | 71.88 | 75.56 | 77.4 |

#### ImageNet Linear Evaluation

The **Feature1 - Feature5** don't have the GlobalAveragePooling, the feature map is pooled to the specific dimensions and then follows a Linear layer to do the classification. Please refer to [resnet50_mhead_linear-8xb32-steplr-90e_in1k](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/benchmarks/classification/imagenet/resnet50_mhead_linear-8xb32-steplr-90e_in1k.py) for details of config.

| Self-Supervised Config                                                                                                                            | Feature1 | Feature2 | Feature3 | Feature4 | Feature5 |
| ------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | -------- | -------- | -------- | -------- |
| [resnet50_8xb32-coslr-200e](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/selfsup/simclr/simclr_resnet50_8xb32-coslr-200e_in1k.py)     | 16.29    | 31.11    | 39.99    | 55.06    | 62.91    |
| [resnet50_16xb256-coslr-200e](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/selfsup/simclr/simclr_resnet50_16xb256-coslr-200e_in1k.py) | 15.44    | 31.47    | 41.83    | 59.44    | 66.41    |

<table class="docutils">
<thead>
  <tr>
	    <th rowspan="2">Algorithm</th>
	    <th rowspan="2">Backbone</th>
	    <th rowspan="2">Epoch</th>
      <th rowspan="2">Batch Size</th>
      <th colspan="2" align="center">Results (Top-1 %)</th>
      <th colspan="3" align="center">Links</th>
	</tr>
	<tr>
      <th>Linear Eval</th>
      <th>Fine-tuning</th>
      <th>Pretrain</th>
      <th>Linear Eval</th>
      <th>Fine-tuning</th>
	</tr>
  </thead>
  <tbody>
  <tr>
	    <td rowspan="3">SimCLR</td>
	    <td>ResNet50</td>
	    <td>200</td>
      <td>256</td>
      <td>62.7</td>
      <td>/</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/simclr/simclr_resnet50_8xb32-coslr-200e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/simclr/simclr_resnet50_8xb32-coslr-200e_in1k/simclr_resnet50_8xb32-coslr-200e_in1k_20220825-15f807a4.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/simclr/simclr_resnet50_8xb32-coslr-200e_in1k/simclr_resnet50_8xb32-coslr-200e_in1k_20220721_103223.json'>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/resnet50_linear-8xb512-coslr-90e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/simclr/simclr_resnet50_8xb32-coslr-200e_in1k/resnet50_linear-8xb512-coslr-90e_in1k/resnet50_linear-8xb512-coslr-90e_in1k_20220825-9596a505.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/simclr/simclr_resnet50_8xb32-coslr-200e_in1k/resnet50_linear-8xb512-coslr-90e_in1k/resnet50_linear-8xb512-coslr-90e_in1k_20220724_210050.json'>log</a></td>
      <td>/</td>
	</tr>
  <tr>
	    <td>ResNet50</td>
	    <td>200</td>
      <td>4096</td>
      <td>66.9</td>
      <td>/</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/simclr/simclr_resnet50_16xb256-coslr-200e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/simclr/simclr_resnet50_16xb256-coslr-200e_in1k/simclr_resnet50_16xb256-coslr-200e_in1k_20220825-4d9cce50.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/simclr/simclr_resnet50_16xb256-coslr-200e_in1k/simclr_resnet50_16xb256-coslr-200e_in1k_20220721_150508.json'>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/resnet50_linear-8xb512-coslr-90e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/simclr/simclr_resnet50_16xb256-coslr-200e_in1k/resnet50_linear-8xb512-coslr-90e_in1k/resnet50_linear-8xb512-coslr-90e_in1k_20220825-f12c0457.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/simclr/simclr_resnet50_16xb256-coslr-200e_in1k/resnet50_linear-8xb512-coslr-90e_in1k/resnet50_linear-8xb512-coslr-90e_in1k_20220724_172050.json'>log</a></td>
      <td>/</td>
	</tr>
  <tr>
	    <td>ResNet50</td>
	    <td>800</td>
      <td>4096</td>
      <td>69.2</td>
      <td>/</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/simclr/simclr_resnet50_16xb256-coslr-800e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/simclr/simclr_resnet50_16xb256-coslr-800e_in1k/simclr_resnet50_16xb256-coslr-800e_in1k_20220825-85fcc4de.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/simclr/simclr_resnet50_16xb256-coslr-800e_in1k/simclr_resnet50_16xb256-coslr-800e_in1k_20220725_112248.json'>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/resnet50_linear-8xb512-coslr-90e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/simclr/simclr_resnet50_16xb256-coslr-800e_in1k/resnet50_linear-8xb512-coslr-90e_in1k/resnet50_linear-8xb512-coslr-90e_in1k_20220825-b80ae1e5.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/simclr/simclr_resnet50_16xb256-coslr-800e_in1k/resnet50_linear-8xb512-coslr-90e_in1k/resnet50_linear-8xb512-coslr-90e_in1k_20220730_165101.json'>log</a></td>
      <td>/</td>
	</tr>
  </tbody>
</table>

#### Places205 Linear Evaluation

The **Feature1 - Feature5** don't have the GlobalAveragePooling, the feature map is pooled to the specific dimensions and then follows a Linear layer to do the classification. Please refer to [resnet50_mhead_8xb32-steplr-28e_places205.py](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/benchmarks/classification/places205/resnet50_mhead_8xb32-steplr-28e_places205.py) for details of config.

| Self-Supervised Config                                                                                                                        | Feature1 | Feature2 | Feature3 | Feature4 | Feature5 |
| --------------------------------------------------------------------------------------------------------------------------------------------- | -------- | -------- | -------- | -------- | -------- |
| [resnet50_8xb32-coslr-200e](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/selfsup/simclr/simclr_resnet50_8xb32-coslr-200e_in1k.py) | 20.60    | 33.62    | 38.86    | 45.25    | 50.91    |

#### ImageNet Nearest-Neighbor Classification

The results are obtained from the features after GlobalAveragePooling. Here, k=10 to 200 indicates different number of nearest neighbors.

| Self-Supervised Config                                                                                                                        | k=10 | k=20 | k=100 | k=200 |
| --------------------------------------------------------------------------------------------------------------------------------------------- | ---- | ---- | ----- | ----- |
| [resnet50_8xb32-coslr-200e](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/selfsup/simclr/simclr_resnet50_8xb32-coslr-200e_in1k.py) | 47.8 | 48.4 | 46.7  | 45.2  |

### Detection

The detection benchmarks includes 2 downstream task datasets, **Pascal VOC 2007 + 2012** and **COCO2017**. This benchmark follows the evluation protocols set up by MoCo.

#### Pascal VOC 2007 + 2012

Please refer to [config](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/benchmarks/mmdetection/voc0712/faster-rcnn_r50-c4_ms-24k_voc0712.py) for details.

| Self-Supervised Config                                                                                                                        | AP50  |
| --------------------------------------------------------------------------------------------------------------------------------------------- | ----- |
| [resnet50_8xb32-coslr-200e](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/selfsup/simclr/simclr_resnet50_8xb32-coslr-200e_in1k.py) | 79.38 |

#### COCO2017

Please refer to [config](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/benchmarks/mmdetection/coco/mask-rcnn_r50_fpn_ms-1x_coco.py) for details.

| Self-Supervised Config                                                                                                                        | mAP(Box) | AP50(Box) | AP75(Box) | mAP(Mask) | AP50(Mask) | AP75(Mask) |
| --------------------------------------------------------------------------------------------------------------------------------------------- | -------- | --------- | --------- | --------- | ---------- | ---------- |
| [resnet50_8xb32-coslr-200e](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/selfsup/simclr/simclr_resnet50_8xb32-coslr-200e_in1k.py) | 38.7     | 58.1      | 42.4      | 34.9      | 55.3       | 37.5       |

### Segmentation

The segmentation benchmarks includes 2 downstream task datasets, **Cityscapes** and **Pascal VOC 2012 + Aug**. It follows the evluation protocols set up by MMSegmentation.

#### Pascal VOC 2012 + Aug

Please refer to [config](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/benchmarks/mmsegmentation/voc12aug/fcn_r50-d8_4xb4-20k_voc12aug-512x512.py) for details.

| Self-Supervised Config                                                                                                                        | mIOU  |
| --------------------------------------------------------------------------------------------------------------------------------------------- | ----- |
| [resnet50_8xb32-coslr-200e](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/selfsup/simclr/simclr_resnet50_8xb32-coslr-200e_in1k.py) | 64.03 |

## Citation

```bibtex
@inproceedings{chen2020simple,
  title={A simple framework for contrastive learning of visual representations},
  author={Chen, Ting and Kornblith, Simon and Norouzi, Mohammad and Hinton, Geoffrey},
  booktitle={ICML},
  year={2020},
}
```
