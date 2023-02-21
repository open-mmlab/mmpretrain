# MoCo v2

> [Improved Baselines with Momentum Contrastive Learning](https://arxiv.org/abs/2003.04297)

<!-- [ALGORITHM] -->

## Abstract

Contrastive unsupervised learning has recently shown encouraging progress, e.g., in Momentum Contrast (MoCo) and SimCLR. In this note, we verify the effectiveness of two of SimCLR’s design improvements by implementing them in the MoCo framework. With simple modifications to MoCo—namely, using an MLP projection head and more data augmentation—we establish stronger baselines that outperform SimCLR and do not require large training batches. We hope this will make state-of-the-art unsupervised learning research more accessible.

<div align="center">
<img  src="https://user-images.githubusercontent.com/36138628/149720067-b65e5736-d425-45b3-93ed-6f2427fc6217.png" width="500" />
</div>

## Models and Benchmarks

In this page, we provide benchmarks as much as possible to evaluate our pre-trained models. If not mentioned, all models are pre-trained on ImageNet-1k dataset.

### Classification

The classification benchmarks includes 4 downstream task datasets, **VOC**, **ImageNet**,  **iNaturalist2018** and **Places205**. If not specified, the results are  Top-1 (%).

#### VOC SVM / Low-shot SVM

The **Best Layer** indicates that the best results are obtained from which layers feature map. For example, if the **Best Layer** is **feature3**, its best result is obtained from the second stage of ResNet (1 for stem layer, 2-5 for 4 stage layers).

Besides, k=1 to 96 indicates the hyper-parameter of Low-shot SVM.

| Self-Supervised Config                                                                                                                        | Best Layer | SVM   | k=1   | k=2   | k=4   | k=8   | k=16  | k=32  | k=64  | k=96  |
| --------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| [resnet50_8xb32-coslr-200e](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/selfsup/mocov2/mocov2_resnet50_8xb32-coslr-200e_in1k.py) | feature5   | 84.04 | 43.14 | 53.29 | 65.34 | 71.03 | 75.42 | 78.48 | 80.88 | 82.23 |

#### ImageNet Linear Evaluation

The **Feature1 - Feature5** don't have the GlobalAveragePooling, the feature map is pooled to the specific dimensions and then follows a Linear layer to do the classification. Please refer to [resnet50_mhead_linear-8xb32-steplr-90e_in1k](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/benchmarks/classification/imagenet/resnet50_mhead_linear-8xb32-steplr-90e_in1k.py) for details of config.

| Self-Supervised Config                                                                                                                        | Feature1 | Feature2 | Feature3 | Feature4 | Feature5 |
| --------------------------------------------------------------------------------------------------------------------------------------------- | -------- | -------- | -------- | -------- | -------- |
| [resnet50_8xb32-coslr-200e](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/selfsup/mocov2/mocov2_resnet50_8xb32-coslr-200e_in1k.py) | 15.96    | 34.22    | 45.78    | 61.11    | 66.24    |

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
	    <td>MoCo v2</td>
	    <td>ResNet50</td>
	    <td>200</td>
      <td>256</td>
      <td>67.5</td>
      <td>/</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/mocov2/mocov2_resnet50_8xb32-coslr-200e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mocov2/mocov2_resnet50_8xb32-coslr-200e_in1k/mocov2_resnet50_8xb32-coslr-200e_in1k_20220825-b6d23c86.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mocov2/mocov2_resnet50_8xb32-coslr-200e_in1k/mocov2_resnet50_8xb32-coslr-200e_in1k_20220721_215805.json'>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/resnet50_linear-8xb32-steplr-100e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mocov2/mocov2_resnet50_8xb32-coslr-200e_in1k/resnet50_linear-8xb32-steplr-100e_in1k/resnet50_linear-8xb32-steplr-100e_in1k_20220825-994c4128.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mocov2/mocov2_resnet50_8xb32-coslr-200e_in1k/resnet50_linear-8xb32-steplr-100e_in1k/resnet50_linear-8xb32-steplr-100e_in1k_20220724_172046.json'>log</a></td>
      <td>/</td>
	</tr>
  </tbody>
</table>

#### Places205 Linear Evaluation

The **Feature1 - Feature5** don't have the GlobalAveragePooling, the feature map is pooled to the specific dimensions and then follows a Linear layer to do the classification. Please refer to [resnet50_mhead_8xb32-steplr-28e_places205.py](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/benchmarks/classification/places205/resnet50_mhead_8xb32-steplr-28e_places205.py) for details of config.

| Self-Supervised Config                                                                                                                        | Feature1 | Feature2 | Feature3 | Feature4 | Feature5 |
| --------------------------------------------------------------------------------------------------------------------------------------------- | -------- | -------- | -------- | -------- | -------- |
| [resnet50_8xb32-coslr-200e](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/selfsup/mocov2/mocov2_resnet50_8xb32-coslr-200e_in1k.py) | 20.92    | 35.72    | 42.62    | 49.79    | 52.25    |

#### ImageNet Nearest-Neighbor Classification

The results are obtained from the features after GlobalAveragePooling. Here, k=10 to 200 indicates different number of nearest neighbors.

| Self-Supervised Config                                                                                                                        | k=10 | k=20 | k=100 | k=200 |
| --------------------------------------------------------------------------------------------------------------------------------------------- | ---- | ---- | ----- | ----- |
| [resnet50_8xb32-coslr-200e](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/selfsup/mocov2/mocov2_resnet50_8xb32-coslr-200e_in1k.py) | 55.6 | 55.7 | 53.7  | 52.5  |

### Detection

The detection benchmarks includes 2 downstream task datasets, **Pascal VOC 2007 + 2012** and **COCO2017**. This benchmark follows the evluation protocols set up by MoCo.

#### Pascal VOC 2007 + 2012

Please refer to [config](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/benchmarks/mmdetection/voc0712/faster-rcnn_r50-c4_ms-24k_voc0712.py) for details.

| Self-Supervised Config                                                                                                                        | AP50  |
| --------------------------------------------------------------------------------------------------------------------------------------------- | ----- |
| [resnet50_8xb32-coslr-200e](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/selfsup/mocov2/mocov2_resnet50_8xb32-coslr-200e_in1k.py) | 81.06 |

#### COCO2017

Please refer to [config](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/benchmarks/mmdetection/coco/mask-rcnn_r50_fpn_ms-1x_coco.py) for details.

| Self-Supervised Config                                                                                                                        | mAP(Box) | AP50(Box) | AP75(Box) | mAP(Mask) | AP50(Mask) | AP75(Mask) |
| --------------------------------------------------------------------------------------------------------------------------------------------- | -------- | --------- | --------- | --------- | ---------- | ---------- |
| [resnet50_8xb32-coslr-200e](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/selfsup/mocov2/mocov2_resnet50_8xb32-coslr-200e_in1k.py) | 40.2     | 59.7      | 44.2      | 36.1      | 56.7       | 38.8       |

### Segmentation

The segmentation benchmarks includes 2 downstream task datasets, **Cityscapes** and **Pascal VOC 2012 + Aug**. It follows the evluation protocols set up by MMSegmentation.

#### Pascal VOC 2012 + Aug

Please refer to [config](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/benchmarks/mmsegmentation/voc12aug/fcn_r50-d8_4xb4-20k_voc12aug-512x512.py) for details.

| Self-Supervised Config                                                                                                                        | mIOU  |
| --------------------------------------------------------------------------------------------------------------------------------------------- | ----- |
| [resnet50_8xb32-coslr-200e](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/selfsup/mocov2/mocov2_resnet50_8xb32-coslr-200e_in1k.py) | 67.55 |

## Citation

```bibtex
@article{chen2020improved,
  title={Improved baselines with momentum contrastive learning},
  author={Chen, Xinlei and Fan, Haoqi and Girshick, Ross and He, Kaiming},
  journal={arXiv preprint arXiv:2003.04297},
  year={2020}
}
```
