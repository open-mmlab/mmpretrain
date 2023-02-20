# MaskFeat

> [Masked Feature Prediction for Self-Supervised Visual Pre-Training](https://arxiv.org/abs/2112.09133v1)

<!-- [ALGORITHM] -->

## Abstract

We present Masked Feature Prediction (MaskFeat) for self-supervised pre-training of video models. Our approach first randomly masks out a portion of the input sequence and then predicts the feature of the masked regions. We study five different types of features and find Histograms of Oriented Gradients (HOG), a hand-crafted feature descriptor, works particularly well in terms of both performance and efficiency. We observe that the local contrast normalization in HOG is essential for good results, which is in line with earlier work using HOG for visual recognition. Our approach can learn abundant visual knowledge and drive large-scale Transformer-based models. Without using extra model weights or supervision, MaskFeat pre-trained on unlabeled videos achieves unprecedented results of 86.7% with MViT-L on Kinetics-400, 88.3% on Kinetics-600, 80.4% on Kinetics-700, 38.8 mAP on AVA, and 75.0% on SSv2. MaskFeat further generalizes to image input, which can be interpreted as a video with a single frame and obtains competitive results on ImageNet.

<div align="center">
<img src="https://user-images.githubusercontent.com/48178838/190090285-428f07c0-0887-4ce8-b94f-f719cfd25622.png" width="60%"/>
</div>

## Models and Benchmarks

Here, we report the results of the model on ImageNet, the details are below:

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
  <tr>
      <td>MaskFeat</td>
	    <td>ViT-base</td>
	    <td>300</td>
      <td>2048</td>
      <td>/</td>
      <td>83.4</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/maskfeat/maskfeat_vit-base-p16_8xb256-amp-coslr-300e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/maskfeat/maskfeat_vit-base-p16_8xb256-amp-coslr-300e_in1k/maskfeat_vit-base-p16_8xb256-amp-coslr-300e_in1k_20221101-6dfc8bf3.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/maskfeat/maskfeat_vit-base-p16_8xb256-amp-coslr-300e_in1k/maskfeat_vit-base-p16_8xb256-amp-coslr-300e_in1k_20221019_194256.json'>log</a></td>
      <td>/</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/vit-base-p16_ft-8xb256-coslr-100e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/maskfeat/maskfeat_vit-base-p16_8xb256-amp-coslr-300e_in1k/vit-base-p16_ft-8xb256-coslr-100e_in1k/vit-base-p16_ft-8xb256-coslr-100e_in1k_20221028-5134431c.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/maskfeat/maskfeat_vit-base-p16_8xb256-amp-coslr-300e_in1k/vit-base-p16_ft-8xb256-coslr-100e_in1k/vit-base-p16_ft-8xb256-coslr-100e_in1k_20221026_105344.json'>log</a></td>
	</tr>
  </tbody>
</table>

## Citation

```bibtex
@InProceedings{wei2022masked,
    author    = {Wei, Chen and Fan, Haoqi and Xie, Saining and Wu, Chao-Yuan and Yuille, Alan and Feichtenhofer, Christoph},
    title     = {Masked Feature Prediction for Self-Supervised Visual Pre-Training},
    booktitle = {CVPR},
    year      = {2022},
}
```
