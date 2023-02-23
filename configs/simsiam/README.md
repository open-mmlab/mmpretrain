# SimSiam

> [Exploring Simple Siamese Representation Learning](https://arxiv.org/abs/2011.10566)

<!-- [ALGORITHM] -->

## Abstract

Siamese networks have become a common structure in various recent models for unsupervised visual representation learning. These models maximize the similarity between two augmentations of one image, subject to certain conditions for avoiding collapsing solutions. In this paper, we report surprising empirical results that simple Siamese networks can learn meaningful representations even using none of the following: (i) negative sample pairs, (ii) large batches, (iii) momentum encoders. Our experiments show that collapsing solutions do exist for the loss and structure, but a stop-gradient operation plays an essential role in preventing collapsing. We provide a hypothesis on the implication of stop-gradient, and further show proof-of-concept experiments verifying it. Our “SimSiam” method achieves competitive results on ImageNet and downstream tasks. We hope this simple baseline will motivate people to rethink the roles of Siamese architectures for unsupervised representation learning.

<div align="center">
<img  src="https://user-images.githubusercontent.com/36138628/149724180-bc7bac6a-fcb8-421e-b8f1-9550c624d154.png" width="500" />
</div>

## Models and Benchmarks

In this page, we provide benchmarks as much as possible to evaluate our pre-trained models. If not mentioned, all models are pre-trained on ImageNet-1k dataset.

### Classification

The classification benchmarks includes 4 downstream task datasets, **VOC**, **ImageNet**,  **iNaturalist2018** and **Places205**. If not specified, the results are Top-1 (%).

#### VOC SVM / Low-shot SVM

The **Best Layer** indicates that the best results are obtained from which layers feature map. For example, if the **Best Layer** is **feature3**, its best result is obtained from the second stage of ResNet (1 for stem layer, 2-5 for 4 stage layers).

Besides, k=1 to 96 indicates the hyper-parameter of Low-shot SVM.

| Self-Supervised Config                                                                                                                          | Best Layer | SVM   | k=1   | k=2   | k=4   | k=8   | k=16  | k=32  | k=64  | k=96  |
| ----------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| [resnet50_8xb32-coslr-100e](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/selfsup/simsiam/simsiam_resnet50_8xb32-coslr-100e_in1k.py) | feature5   | 84.64 | 39.65 | 49.86 | 62.48 | 69.50 | 74.48 | 78.31 | 81.06 | 82.56 |
| [resnet50_8xb32-coslr-200e](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/selfsup/simsiam/simsiam_resnet50_8xb32-coslr-200e_in1k.py) | feature5   | 85.20 | 39.85 | 50.44 | 63.73 | 70.93 | 75.74 | 79.42 | 82.02 | 83.44 |

#### ImageNet Linear Evaluation

The **Feature1 - Feature5** don't have the GlobalAveragePooling, the feature map is pooled to the specific dimensions and then follows a Linear layer to do the classification. Please refer to [resnet50_mhead_linear-8xb32-steplr-90e_in1k](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/benchmarks/classification/imagenet/resnet50_mhead_linear-8xb32-steplr-90e_in1k.py) for details of config.

| Self-Supervised Config                                                                                                                          | Feature1 | Feature2 | Feature3 | Feature4 | Feature5 |
| ----------------------------------------------------------------------------------------------------------------------------------------------- | -------- | -------- | -------- | -------- | -------- |
| [resnet50_8xb32-coslr-100e](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/selfsup/simsiam/simsiam_resnet50_8xb32-coslr-100e_in1k.py) | 16.27    | 33.77    | 45.80    | 60.83    | 68.21    |
| [resnet50_8xb32-coslr-200e](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/selfsup/simsiam/simsiam_resnet50_8xb32-coslr-200e_in1k.py) | 15.57    | 37.21    | 47.28    | 62.21    | 69.85    |

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
	    <td rowspan="2">SimSiam</td>
	    <td>ResNet50</td>
	    <td>100</td>
      <td>256</td>
      <td>68.3</td>
      <td>/</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/simsiam/simsiam_resnet50_8xb32-coslr-100e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/simsiam/simsiam_resnet50_8xb32-coslr-100e_in1k/simsiam_resnet50_8xb32-coslr-100e_in1k_20220825-d07cb2e6.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/simsiam/simsiam_resnet50_8xb32-coslr-100e_in1k/simsiam_resnet50_8xb32-coslr-100e_in1k_20220725_224724.json'>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/resnet50_linear-8xb512-coslr-90e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/simsiam/simsiam_resnet50_8xb32-coslr-100e_in1k/resnet50_linear-8xb512-coslr-90e_in1k/resnet50_linear-8xb512-coslr-90e_in1k_20220825-f53ba400.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/simsiam/simsiam_resnet50_8xb32-coslr-100e_in1k/resnet50_linear-8xb512-coslr-90e_in1k/resnet50_linear-8xb512-coslr-90e_in1k_20220804_175115.json'>log</a></td>
      <td>/</td>
	</tr>
  <tr>
	    <td>ResNet50</td>
	    <td>200</td>
      <td>256</td>
      <td>69.8</td>
      <td>/</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/simsiam/simsiam_resnet50_8xb32-coslr-200e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/simsiam/simsiam_resnet50_8xb32-coslr-200e_in1k/simsiam_resnet50_8xb32-coslr-200e_in1k_20220825-efe91299.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/simsiam/simsiam_resnet50_8xb32-coslr-200e_in1k/simsiam_resnet50_8xb32-coslr-200e_in1k_20220726_033722.json'>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/resnet50_linear-8xb512-coslr-90e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/simsiam/simsiam_resnet50_8xb32-coslr-200e_in1k/resnet50_linear-8xb512-coslr-90e_in1k/resnet50_linear-8xb512-coslr-90e_in1k_20220825-519b5135.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/simsiam/simsiam_resnet50_8xb32-coslr-200e_in1k/resnet50_linear-8xb512-coslr-90e_in1k/resnet50_linear-8xb512-coslr-90e_in1k_20220802_120717.json'>log</a></td>
      <td>/</td>
	</tr>
  </tbody>
</table>

#### Places205 Linear Evaluation

The **Feature1 - Feature5** don't have the GlobalAveragePooling, the feature map is pooled to the specific dimensions and then follows a Linear layer to do the classification. Please refer to [resnet50_mhead_8xb32-steplr-28e_places205.py](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/benchmarks/classification/places205/resnet50_mhead_8xb32-steplr-28e_places205.py) for details of config.

| Self-Supervised Config                                                                                                                          | Feature1 | Feature2 | Feature3 | Feature4 | Feature5 |
| ----------------------------------------------------------------------------------------------------------------------------------------------- | -------- | -------- | -------- | -------- | -------- |
| [resnet50_8xb32-coslr-100e](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/selfsup/simsiam/simsiam_resnet50_8xb32-coslr-100e_in1k.py) | 21.32    | 35.66    | 43.05    | 50.79    | 53.27    |
| [resnet50_8xb32-coslr-200e](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/selfsup/simsiam/simsiam_resnet50_8xb32-coslr-200e_in1k.py) | 21.17    | 35.85    | 43.49    | 50.99    | 54.10    |

#### ImageNet Nearest-Neighbor Classification

The results are obtained from the features after GlobalAveragePooling. Here, k=10 to 200 indicates different number of nearest neighbors.

| Self-Supervised Config                                                                                                                          | k=10 | k=20 | k=100 | k=200 |
| ----------------------------------------------------------------------------------------------------------------------------------------------- | ---- | ---- | ----- | ----- |
| [resnet50_8xb32-coslr-100e](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/selfsup/simsiam/simsiam_resnet50_8xb32-coslr-100e_in1k.py) | 57.4 | 57.6 | 55.8  | 54.2  |
| [resnet50_8xb32-coslr-200e](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/selfsup/simsiam/simsiam_resnet50_8xb32-coslr-200e_in1k.py) | 60.2 | 60.4 | 58.8  | 57.4  |

### Detection

The detection benchmarks includes 2 downstream task datasets, **Pascal VOC 2007 + 2012** and **COCO2017**. This benchmark follows the evluation protocols set up by MoCo.

#### Pascal VOC 2007 + 2012

Please refer to [config](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/benchmarks/mmdetection/voc0712/faster-rcnn_r50-c4_ms-24k_voc0712.py) for details.

| Self-Supervised Config                                                                                                                          | AP50  |
| ----------------------------------------------------------------------------------------------------------------------------------------------- | ----- |
| [resnet50_8xb32-coslr-100e](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/selfsup/simsiam/simsiam_resnet50_8xb32-coslr-100e_in1k.py) | 79.80 |
| [resnet50_8xb32-coslr-200e](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/selfsup/simsiam/simsiam_resnet50_8xb32-coslr-200e_in1k.py) | 79.85 |

#### COCO2017

Please refer to [config](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/benchmarks/mmdetection/coco/mask-rcnn_r50_fpn_ms-1x_coco.py) for details.

| Self-Supervised Config                                                                                                                          | mAP(Box) | AP50(Box) | AP75(Box) | mAP(Mask) | AP50(Mask) | AP75(Mask) |
| ----------------------------------------------------------------------------------------------------------------------------------------------- | -------- | --------- | --------- | --------- | ---------- | ---------- |
| [resnet50_8xb32-coslr-100e](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/selfsup/simsiam/simsiam_resnet50_8xb32-coslr-100e_in1k.py) | 38.6     | 57.6      | 42.3      | 34.6      | 54.8       | 36.9       |
| [resnet50_8xb32-coslr-200e](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/selfsup/simsiam/simsiam_resnet50_8xb32-coslr-200e_in1k.py) | 38.8     | 58.0      | 42.3      | 34.9      | 55.3       | 37.6       |

### Segmentation

The segmentation benchmarks includes 2 downstream task datasets, **Cityscapes** and **Pascal VOC 2012 + Aug**. It follows the evluation protocols set up by MMSegmentation.

#### Pascal VOC 2012 + Aug

Please refer to [config](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/benchmarks/mmsegmentation/voc12aug/fcn_r50-d8_4xb4-20k_voc12aug-512x512.py) for details.

| Self-Supervised Config                                                                                                                          | mIOU  |
| ----------------------------------------------------------------------------------------------------------------------------------------------- | ----- |
| [resnet50_8xb32-coslr-100e](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/selfsup/simsiam/simsiam_resnet50_8xb32-coslr-100e_in1k.py) | 48.35 |
| [resnet50_8xb32-coslr-200e](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/selfsup/simsiam/simsiam_resnet50_8xb32-coslr-200e_in1k.py) | 46.27 |

## Citation

```bibtex
@inproceedings{chen2021exploring,
  title={Exploring simple siamese representation learning},
  author={Chen, Xinlei and He, Kaiming},
  booktitle={CVPR},
  year={2021}
}
```
