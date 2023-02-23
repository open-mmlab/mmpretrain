# MoCo v3

> [An Empirical Study of Training Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.02057)

<!-- [ALGORITHM] -->

## Abstract

This paper does not describe a novel method. Instead, it studies a straightforward, incremental, yet must-know baseline given the recent progress in computer vision: self-supervised learning for Vision Transformers (ViT). While the training recipes for standard convolutional networks have been highly mature and robust, the recipes for ViT are yet to be built, especially in the self-supervised scenarios where training becomes more challenging. In this work, we go back to basics and investigate the effects of several fundamental components for training self-supervised ViT. We observe that instability is a major issue that degrades accuracy, and it can be hidden by apparently good results. We reveal that these results are indeed partial failure, and they can be improved when training is made more stable. We benchmark ViT results in MoCo v3 and several other self-supervised frameworks, with ablations in various aspects. We discuss the currently positive evidence as well as challenges and open questions. We hope that this work will provide useful data points and experience for future research.

<div align="center">
<img  src="https://user-images.githubusercontent.com/36138628/151305362-e6e8ea35-b3b8-45f6-8819-634e67083218.png" width="500" />
</div>

## Models and Benchmarks

In this page, we provide benchmarks as much as possible to evaluate our pre-trained models. If not mentioned, all models are pre-trained on ImageNet-1k dataset.

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
      <td rowspan="6">MoCo v3</td>
	    <td>ResNet50</td>
	    <td>100</td>
      <td>4096</td>
      <td>69.6</td>
      <td>/</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/mocov3/mocov3_resnet50_8xb512-amp-coslr-100e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_resnet50_8xb512-amp-coslr-100e_in1k/mocov3_resnet50_8xb512-amp-coslr-100e_in1k_20220927-f1144efa.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_resnet50_8xb512-amp-coslr-100e_in1k/mocov3_resnet50_8xb512-amp-coslr-100e_in1k_20220915_154635.json'>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/resnet50_linear-8xb128-coslr-90e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_resnet50_8xb512-amp-coslr-100e_in1k/resnet50_linear-8xb128-coslr-90e_in1k/resnet50_linear-8xb128-coslr-90e_in1k_20220927-8f7d937e.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_resnet50_8xb512-amp-coslr-100e_in1k/resnet50_linear-8xb128-coslr-90e_in1k/resnet50_linear-8xb128-coslr-90e_in1k_20220920_113350.json'>log</a></td>
      <td>/</td>
	</tr>
  <tr>
	    <td>ResNet50</td>
	    <td>300</td>
      <td>4096</td>
      <td>72.8</td>
      <td>/</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/mocov3/mocov3_resnet50_8xb512-amp-coslr-300e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_resnet50_8xb512-amp-coslr-300e_in1k/mocov3_resnet50_8xb512-amp-coslr-300e_in1k_20220927-1e4f3304.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_resnet50_8xb512-amp-coslr-300e_in1k/mocov3_resnet50_8xb512-amp-coslr-300e_in1k_20220915_180538.json'>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/resnet50_linear-8xb128-coslr-90e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_resnet50_8xb512-amp-coslr-300e_in1k/resnet50_linear-8xb128-coslr-90e_in1k/resnet50_linear-8xb128-coslr-90e_in1k_20220927-d21ddac2.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_resnet50_8xb512-amp-coslr-300e_in1k/resnet50_linear-8xb128-coslr-90e_in1k/resnet50_linear-8xb128-coslr-90e_in1k_20220920_113403.json'>log</a></td>
      <td>/</td>
	</tr>
  <tr>
	    <td>ResNet50</td>
	    <td>800</td>
      <td>4096</td>
      <td>74.4</td>
      <td>/</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/mocov3/mocov3_resnet50_8xb512-amp-coslr-800e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_resnet50_8xb512-amp-coslr-800e_in1k/mocov3_resnet50_8xb512-amp-coslr-800e_in1k_20220927-e043f51a.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_resnet50_8xb512-amp-coslr-800e_in1k/mocov3_resnet50_8xb512-amp-coslr-800e_in1k_20220919_111209.json'>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/resnet50_linear-8xb128-coslr-90e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_resnet50_8xb512-amp-coslr-800e_in1k/resnet50_linear-8xb128-coslr-90e_in1k/resnet50_linear-8xb128-coslr-90e_in1k_20220927-0e97a483.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_resnet50_8xb512-amp-coslr-800e_in1k/resnet50_linear-8xb128-coslr-90e_in1k/resnet50_linear-8xb128-coslr-90e_in1k_20220926_102021.json'>log</a></td>
      <td>/</td>
	</tr>
  <tr>
	    <td>ViT-small</td>
	    <td>300</td>
      <td>4096</td>
      <td>73.6</td>
      <td>/</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/mocov3/mocov3_vit-small-p16_16xb256-amp-coslr-300e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_vit-small-p16_16xb256-amp-coslr-300e_in1k/mocov3_vit-small-p16_16xb256-amp-coslr-300e_in1k-224_20220826-08bc52f7.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_vit-small-p16_16xb256-amp-coslr-300e_in1k/mocov3_vit-small-p16_16xb256-amp-coslr-300e_in1k-224_20220721_153833.json'>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/vit-small-p16_linear-8xb128-coslr-90e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_vit-small-p16_16xb256-amp-coslr-300e_in1k/vit-small-p16_linear-8xb128-coslr-90e_in1k/vit-small-p16_linear-8xb128-coslr-90e_in1k_20220826-376674ef.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_vit-small-p16_16xb256-amp-coslr-300e_in1k/vit-small-p16_linear-8xb128-coslr-90e_in1k/vit-small-p16_linear-8xb128-coslr-90e_in1k_20220724_140850.json'>log</a></td>
      <td>/</td>
	</tr>
  <tr>
	    <td>ViT-base</td>
	    <td>300</td>
      <td>4096</td>
      <td>76.9</td>
      <td>83.0</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/mocov3/mocov3_vit-base-p16_16xb256-amp-coslr-300e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_vit-base-p16_16xb256-amp-coslr-300e_in1k/mocov3_vit-base-p16_16xb256-amp-coslr-300e_in1k-224_20220826-25213343.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_vit-base-p16_16xb256-amp-coslr-300e_in1k/mocov3_vit-base-p16_16xb256-amp-coslr-300e_in1k-224_20220725_104223.json'>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/vit-base-p16_linear-8xb128-coslr-90e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_vit-base-p16_16xb256-amp-coslr-300e_in1k/vit-base-p16_linear-8xb128-coslr-90e_in1k/vit-base-p16_linear-8xb128-coslr-90e_in1k_20220826-83be7758.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_vit-base-p16_16xb256-amp-coslr-300e_in1k/vit-base-p16_linear-8xb128-coslr-90e_in1k/vit-base-p16_linear-8xb128-coslr-90e_in1k_20220729_004628.json'>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/vit-base-p16_ft-8xb64-coslr-150e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_vit-base-p16_16xb256-amp-coslr-300e_in1k/vit-base-p16_ft-8xb64-coslr-150e_in1k/vit-base-p16_ft-8xb64-coslr-150e_in1k_20220826-f1e6c442.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_vit-base-p16_16xb256-amp-coslr-300e_in1k/vit-base-p16_ft-8xb64-coslr-150e_in1k/vit-base-p16_ft-8xb64-coslr-150e_in1k_20220809_103500.json'>log</a></td>
	</tr>
  <tr>
	    <td>ViT-large</td>
	    <td>300</td>
      <td>4096</td>
      <td>/</td>
      <td>83.7</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/mocov3/mocov3_vit-large-p16_64xb64-amp-coslr-300e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_vit-large-p16_64xb64-amp-coslr-300e_in1k/mocov3_vit-large-p16_64xb64-amp-coslr-300e_in1k-224_20220829-9b88a442.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_vit-large-p16_64xb64-amp-coslr-300e_in1k/mocov3_vit-large-p16_64xb64-amp-coslr-300e_in1k-224_20220818_143032.json'>log</a></td>
      <td>/</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/vit-large-p16_ft-8xb64-coslr-100e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_vit-large-p16_64xb64-amp-coslr-300e_in1k/vit-large-p16_ft-8xb64-coslr-100e_in1k/vit-large-p16_ft-8xb64-coslr-100e_in1k_20220829-878a2f7f.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_vit-large-p16_64xb64-amp-coslr-300e_in1k/vit-large-p16_ft-8xb64-coslr-100e_in1k/vit-large-p16_ft-8xb64-coslr-100e_in1k_20220825_201433.json'>log</a></td>
	</tr>
    </tbody>
</table>

## Citation

```bibtex
@InProceedings{Chen_2021_ICCV,
    title     = {An Empirical Study of Training Self-Supervised Vision Transformers},
    author    = {Chen, Xinlei and Xie, Saining and He, Kaiming},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    year      = {2021}
}
```
