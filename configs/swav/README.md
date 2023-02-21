# SwAV

> [Unsupervised Learning of Visual Features by Contrasting Cluster Assignments](https://arxiv.org/abs/2006.09882)

<!-- [ALGORITHM] -->

## Abstract

Unsupervised image representations have significantly reduced the gap with supervised pretraining, notably with the recent achievements of contrastive learning methods. These contrastive methods typically work online and rely on a large number of explicit pairwise feature comparisons, which is computationally challenging. In this paper, we propose an online algorithm, SwAV, that takes advantage of contrastive methods without requiring to compute pairwise comparisons. Specifically, our method simultaneously clusters the data while enforcing consistency between cluster assignments produced for different augmentations (or “views”) of the same image, instead of comparing features directly as in contrastive learning. Simply put, we use a “swapped” prediction mechanism where we predict the code of a view from the representation of another view. Our method can be trained with large and small batches and can scale to unlimited amounts of data. Compared to previous contrastive methods, our method is more memory efficient since it does not require a large memory bank or a special momentum network. In addition, we also propose a new data augmentation strategy, multi-crop, that uses a mix of views with different resolutions in place of two full-resolution views, without increasing the memory or compute requirements.

<div align="center">
<img  src="https://user-images.githubusercontent.com/36138628/149724517-9f1e7bdf-04c7-43e3-92f4-2b8fc1399123.png" width="500" />
</div>

## Models and Benchmarks

In this page, we provide benchmarks as much as possible to evaluate our pre-trained models. If not mentioned, all models are pre-trained on ImageNet-1k dataset.

### Classification

The classification benchmarks includes 4 downstream task datasets, **VOC**, **ImageNet**,  **iNaturalist2018** and **Places205**. If not specified, the results are Top-1 (%).

#### VOC SVM / Low-shot SVM

The **Best Layer** indicates that the best results are obtained from which layers feature map. For example, if the **Best Layer** is **feature3**, its best result is obtained from the second stage of ResNet (1 for stem layer, 2-5 for 4 stage layers).

Besides, k=1 to 96 indicates the hyper-parameter of Low-shot SVM.

| Self-Supervised Config                                                                                                                                                              | Best Layer | SVM   | k=1   | k=2   | k=4   | k=8   | k=16  | k=32  | k=64  | k=96  |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| [resnet50_8xb32-mcrop-2-6-coslr-200e_in1k-224-96](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/swav/swav_resnet50_8xb32-mcrop-2-6-coslr-200e_in1k-224-96.py) | feature5   | 87.00 | 44.68 | 55.41 | 67.64 | 73.67 | 78.14 | 81.58 | 83.98 | 85.15 |

#### ImageNet Linear Evaluation

The **Feature1 - Feature5** don't have the GlobalAveragePooling, the feature map is pooled to the specific dimensions and then follows a Linear layer to do the classification. Please refer to [resnet50_mhead_linear-8xb32-steplr-90e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/benchmarks/classification/imagenet/resnet50_mhead_linear-8xb32-steplr-90e_in1k.py) for details of config.

| Self-Supervised Config                                                                                                                                                              | Feature1 | Feature2 | Feature3 | Feature4 | Feature5 |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | -------- | -------- | -------- | -------- |
| [resnet50_8xb32-mcrop-2-6-coslr-200e_in1k-224-96](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/swav/swav_resnet50_8xb32-mcrop-2-6-coslr-200e_in1k-224-96.py) | 16.98    | 34.96    | 49.26    | 65.98    | 70.74    |

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
	    <td>SwAV</td>
	    <td>ResNet50</td>
	    <td>200</td>
      <td>256</td>
      <td>70.5</td>
      <td>/</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/swav/swav_resnet50_8xb32-mcrop-2-6-coslr-200e_in1k-224-96.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/swav/swav_resnet50_8xb32-mcrop-2-6-coslr-200e_in1k-224-96/swav_resnet50_8xb32-mcrop-2-6-coslr-200e_in1k-224-96_20220825-5b3fc7fc.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/swav/swav_resnet50_8xb32-mcrop-2-6-coslr-200e_in1k-224-96/swav_resnet50_8xb32-mcrop-2-6-coslr-200e_in1k-224-96_20220728_141003.json'>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/resnet50_linear-8xb32-coslr-100e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/swav/swav_resnet50_8xb32-mcrop-2-6-coslr-200e_in1k-224-96/resnet50_linear-8xb32-coslr-100e_in1k/resnet50_linear-8xb32-coslr-100e_in1k_20220825-80341e08.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/swav/swav_resnet50_8xb32-mcrop-2-6-coslr-200e_in1k-224-96/resnet50_linear-8xb32-coslr-100e_in1k/resnet50_linear-8xb32-coslr-100e_in1k_20220802_145230.json'>log</a></td>
      <td>/</td>
	</tr>
  </tbody>
</table>

#### Places205 Linear Evaluation

The **Feature1 - Feature5** don't have the GlobalAveragePooling, the feature map is pooled to the specific dimensions and then follows a Linear layer to do the classification. Please refer to [resnet50_mhead_8xb32-steplr-28e_places205.py](https://github.com/open-mmlab/mmselfsup/blob/master/configs/benchmarks/classification/places205/resnet50_mhead_8xb32-steplr-28e_places205.py) for details of config.

| Self-Supervised Config                                                                                                                                                              | Feature1 | Feature2 | Feature3 | Feature4 | Feature5 |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | -------- | -------- | -------- | -------- |
| [resnet50_8xb32-mcrop-2-6-coslr-200e_in1k-224-96](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/swav/swav_resnet50_8xb32-mcrop-2-6-coslr-200e_in1k-224-96.py) | 23.33    | 35.45    | 43.13    | 51.98    | 55.09    |

#### ImageNet Nearest-Neighbor Classification

The results are obtained from the features after GlobalAveragePooling. Here, k=10 to 200 indicates different number of nearest neighbors.

| Self-Supervised Config                                                                                                                                                              | k=10 | k=20 | k=100 | k=200 |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---- | ---- | ----- | ----- |
| [resnet50_8xb32-mcrop-2-6-coslr-200e_in1k-224-96](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/swav/swav_resnet50_8xb32-mcrop-2-6-coslr-200e_in1k-224-96.py) | 60.5 | 60.6 | 59.0  | 57.6  |

### Detection

The detection benchmarks includes 2 downstream task datasets, **Pascal VOC 2007 + 2012** and **COCO2017**. This benchmark follows the evluation protocols set up by MoCo.

#### Pascal VOC 2007 + 2012

Please refer to [config](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/benchmarks/mmdetection/voc0712/faster-rcnn_r50-c4_ms-24k_voc0712.py) for details.

| Self-Supervised Config                                                                                                                                                              | AP50  |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----- |
| [resnet50_8xb32-mcrop-2-6-coslr-200e_in1k-224-96](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/swav/swav_resnet50_8xb32-mcrop-2-6-coslr-200e_in1k-224-96.py) | 77.64 |

#### COCO2017

Please refer to [config](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/benchmarks/mmdetection/coco/mask-rcnn_r50_fpn_ms-1x_coco.py) for details.

| Self-Supervised Config                                                                                                                                                              | mAP(Box) | AP50(Box) | AP75(Box) | mAP(Mask) | AP50(Mask) | AP75(Mask) |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | --------- | --------- | --------- | ---------- | ---------- |
| [resnet50_8xb32-mcrop-2-6-coslr-200e_in1k-224-96](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/swav/swav_resnet50_8xb32-mcrop-2-6-coslr-200e_in1k-224-96.py) | 40.2     | 60.5      | 43.9      | 36.3      | 57.5       | 38.8       |

### Segmentation

The segmentation benchmarks includes 2 downstream task datasets, **Cityscapes** and **Pascal VOC 2012 + Aug**. It follows the evluation protocols set up by MMSegmentation.

#### Pascal VOC 2012 + Aug

Please refer to [config](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/benchmarks/mmsegmentation/voc12aug/fcn_r50-d8_4xb4-20k_voc12aug-512x512.py) for details.

| Self-Supervised Config                                                                                                                                                              | mIOU  |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----- |
| [resnet50_8xb32-mcrop-2-6-coslr-200e_in1k-224-96](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/swav/swav_resnet50_8xb32-mcrop-2-6-coslr-200e_in1k-224-96.py) | 63.73 |

## Citation

```bibtex
@article{caron2020unsupervised,
  title={Unsupervised Learning of Visual Features by Contrasting Cluster Assignments},
  author={Caron, Mathilde and Misra, Ishan and Mairal, Julien and Goyal, Priya and Bojanowski, Piotr and Joulin, Armand},
  booktitle={NeurIPS},
  year={2020}
}
```
