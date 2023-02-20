# DenseCL

> [Dense Contrastive Learning for Self-Supervised Visual Pre-Training](https://arxiv.org/abs/2011.09157)

<!-- [ALGORITHM] -->

## Abstract

To date, most existing self-supervised learning methods are designed and optimized for image classification. These pre-trained models can be sub-optimal for dense prediction tasks due to the discrepancy between image-level prediction and pixel-level prediction. To fill this gap, we aim to design an effective, dense self-supervised learning method that directly works at the level of pixels (or local features) by taking into account the correspondence between local features. We present dense contrastive learning (DenseCL), which implements self-supervised learning by optimizing a pairwise contrastive (dis)similarity loss at the pixel level between two views of input images.

<div align="center">
<img src="https://user-images.githubusercontent.com/36138628/149721111-bab03a6d-a30d-418e-b338-43c3689cfc65.png" width="900" />
</div>

## Models and Benchmarks

In this page, we provide benchmarks as much as possible to evaluate our pre-trained models. If not mentioned, all models are pre-trained on ImageNet-1k dataset.

### Classification

The classification benchmarks includes 4 downstream task datasets, **VOC**, **ImageNet**,  **iNaturalist2018** and **Places205**. If not specified, the results are Top-1 (%).

#### VOC SVM / Low-shot SVM

The **Best Layer** indicates that the best results are obtained from which layers feature map. For example, if the **Best Layer** is **feature3**, its best result is obtained from the second stage of ResNet (1 for stem layer, 2-5 for 4 stage layers).

Besides, k=1 to 96 indicates the hyper-parameter of Low-shot SVM.

| Self-Supervised Config                                                                                                                          | Best Layer | SVM  | k=1   | k=2   | k=4   | k=8   | k=16  | k=32  | k=64  | k=96  |
| ----------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | ---- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| [resnet50_8xb32-coslr-200e](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/selfsup/densecl/densecl_resnet50_8xb32-coslr-200e_in1k.py) | feature5   | 82.5 | 42.68 | 50.64 | 61.74 | 68.17 | 72.99 | 76.07 | 79.19 | 80.55 |

#### ImageNet Linear Evaluation

The **Feature1 - Feature5** don't have the GlobalAveragePooling, the feature map is pooled to the specific dimensions and then follows a Linear layer to do the classification. Please refer to [resnet50_mhead_linear-8xb32-steplr-90e_in1k](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/benchmarks/classification/imagenet/resnet50_mhead_linear-8xb32-steplr-90e_in1k.py) for details of config.

| Self-Supervised Config                                                                                                                          | Feature1 | Feature2 | Feature3 | Feature4 | Feature5 |
| ----------------------------------------------------------------------------------------------------------------------------------------------- | -------- | -------- | -------- | -------- | -------- |
| [resnet50_8xb32-coslr-200e](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/selfsup/densecl/densecl_resnet50_8xb32-coslr-200e_in1k.py) | 15.86    | 35.47    | 49.46    | 64.06    | 62.95    |

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
	    <td>DenseCL</td>
	    <td>ResNet50</td>
	    <td>200</td>
      <td>256</td>
      <td>63.5</td>
      <td>/</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/densecl/densecl_resnet50_8xb32-coslr-200e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/densecl/densecl_resnet50_8xb32-coslr-200e_in1k/densecl_resnet50_8xb32-coslr-200e_in1k_20220825-3078723b.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/densecl/densecl_resnet50_8xb32-coslr-200e_in1k/densecl_resnet50_8xb32-coslr-200e_in1k_20220727_221415.json'>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/resnet50_linear-8xb32-steplr-100e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/densecl/densecl_resnet50_8xb32-coslr-200e_in1k/resnet50_linear-8xb32-steplr-100e_in1k/resnet50_linear-8xb32-steplr-100e_in1k_20220825-f0f0a579.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/densecl/densecl_resnet50_8xb32-coslr-200e_in1k/resnet50_linear-8xb32-steplr-100e_in1k/resnet50_linear-8xb32-steplr-100e_in1k_20220730_091650.json'>log</a></td>
      <td>/</td>
	</tr>
  </tbody>
</table>

#### Places205 Linear Evaluation

The **Feature1 - Feature5** don't have the GlobalAveragePooling, the feature map is pooled to the specific dimensions and then follows a Linear layer to do the classification. Please refer to [resnet50_mhead_8xb32-steplr-28e_places205.py](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/benchmarks/classification/places205/resnet50_mhead_8xb32-steplr-28e_places205.py) for details of config.

| Self-Supervised Config                                                                                                                          | Feature1 | Feature2 | Feature3 | Feature4 | Feature5 |
| ----------------------------------------------------------------------------------------------------------------------------------------------- | -------- | -------- | -------- | -------- | -------- |
| [resnet50_8xb32-coslr-200e](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/selfsup/densecl/densecl_resnet50_8xb32-coslr-200e_in1k.py) | 21.32    | 36.20    | 43.97    | 51.04    | 50.45    |

#### ImageNet Nearest-Neighbor Classification

The results are obtained from the features after GlobalAveragePooling. Here, k=10 to 200 indicates different number of nearest neighbors.

| Self-Supervised Config                                                                                                                          | k=10 | k=20 | k=100 | k=200 |
| ----------------------------------------------------------------------------------------------------------------------------------------------- | ---- | ---- | ----- | ----- |
| [resnet50_8xb32-coslr-200e](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/selfsup/densecl/densecl_resnet50_8xb32-coslr-200e_in1k.py) | 48.2 | 48.5 | 46.8  | 45.6  |

### Detection

The detection benchmarks includes 2 downstream task datasets, **Pascal VOC 2007 + 2012** and **COCO2017**. This benchmark follows the evluation protocols set up by MoCo.

#### Pascal VOC 2007 + 2012

Please refer to [config](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/benchmarks/mmdetection/voc0712/faster-rcnn_r50-c4_ms-24k_voc0712.py) for details.

| Self-Supervised Config                                                                                                                          | AP50  |
| ----------------------------------------------------------------------------------------------------------------------------------------------- | ----- |
| [resnet50_8xb32-coslr-200e](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/selfsup/densecl/densecl_resnet50_8xb32-coslr-200e_in1k.py) | 82.14 |

#### COCO2017

Please refer to [config](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/benchmarks/mmdetection/coco/mask-rcnn_r50_fpn_ms-1x_coco.py) for details.

| Self-Supervised Config                                                                                                                          | mAP(Box) | AP50(Box) | AP75(Box) | mAP(Mask) | AP50(Mask) | AP75(Mask) |
| ----------------------------------------------------------------------------------------------------------------------------------------------- | -------- | --------- | --------- | --------- | ---------- | ---------- |
| [resnet50_8xb32-coslr-200e](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/selfsup/densecl/densecl_resnet50_8xb32-coslr-200e_in1k.py) |          |           |           |           |            |            |

### Segmentation

The segmentation benchmarks includes 2 downstream task datasets, **Cityscapes** and **Pascal VOC 2012 + Aug**. It follows the evluation protocols set up by MMSegmentation.

#### Pascal VOC 2012 + Aug

Please refer to [config](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/benchmarks/mmsegmentation/voc12aug/fcn_r50-d8_4xb4-20k_voc12aug-512x512.py) for details.

| Self-Supervised Config                                                                                                                          | mIOU  |
| ----------------------------------------------------------------------------------------------------------------------------------------------- | ----- |
| [resnet50_8xb32-coslr-200e](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/selfsup/densecl/densecl_resnet50_8xb32-coslr-200e_in1k.py) | 69.47 |

## Citation

```bibtex
@inproceedings{wang2021dense,
  title={Dense contrastive learning for self-supervised visual pre-training},
  author={Wang, Xinlong and Zhang, Rufeng and Shen, Chunhua and Kong, Tao and Li, Lei},
  booktitle={CVPR},
  year={2021}
}
```
