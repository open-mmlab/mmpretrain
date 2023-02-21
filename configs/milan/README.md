# MILAN

> [MILAN: Masked Image Pretraining on
> Language Assisted Representation
> ](https://arxiv.org/pdf/2208.06049)

<!-- [ALGORITHM] -->

## Abstract

Self-attention based transformer models have been dominating many computer
vision tasks in the past few years. Their superb model qualities heavily depend
on the excessively large labeled image datasets. In order to reduce the reliance
on large labeled datasets, reconstruction based masked autoencoders are gaining
popularity, which learn high quality transferable representations from unlabeled
images. For the same purpose, recent weakly supervised image pretraining methods
explore language supervision from text captions accompanying the images. In this
work, we propose masked image pretraining on language assisted representation,
dubbed as MILAN. Instead of predicting raw pixels or low level features, our
pretraining objective is to reconstruct the image features with substantial semantic
signals that are obtained using caption supervision. Moreover, to accommodate our
reconstruction target, we propose a more efficient prompting decoder architecture
and a semantic aware mask sampling mechanism, which further advance the
transfer performance of the pretrained model. Experimental results demonstrate
that MILAN delivers higher accuracy than the previous works. When the masked
autoencoder is pretrained and finetuned on ImageNet-1K dataset with an input
resolution of 224×224, MILAN achieves a top-1 accuracy of 85.4% on ViTB/16, surpassing previous state-of-the-arts by 1%. In the downstream semantic
segmentation task, MILAN achieves 52.7 mIoU using ViT-B/16 backbone on
ADE20K dataset, outperforming previous masked pretraining results by 4 points.

<div align="center">
<img src="https://user-images.githubusercontent.com/30762564/205210369-41a65c4c-bcd4-4147-91ea-c6c9061ab455.png" width="80%"/>
</div>

## Models and Benchmarks

Here, we report the results of the model, which is pre-trained on ImageNet-1k
for 400 epochs, the details are below:

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
      <td rowspan="1">MILAN</td>
	    <td>ViT-B/16</td>
	    <td>400</td>
      <td>4096</td>
      <td>78.9</td>
      <td>85.3</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/milan/milan_vit-base-p16_16xb256-amp-coslr-400e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/milan/milan_vit-base-p16_16xb256-amp-coslr-400e_in1k/milan_vit-base-p16_16xb256-amp-coslr-400e_in1k_20221129-180922e8.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/milan/milan_vit-base-p16_16xb256-amp-coslr-400e_in1k/milan_vit-base-p16_16xb256-amp-coslr-400e_in1k_20221123_112721.json'>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/milan/classification/vit-base-p16_linear-8xb2048-coslr-100e_in1k.py'>config</a> |<a href='https://download.openmmlab.com/mmselfsup/1.x/milan/milan_vit-base-p16_16xb256-amp-coslr-400e_in1k/vit-base-p16_linear-8xb2048-coslr-100e_in1k/vit-base-p16_linear-8xb2048-coslr-100e_in1k_20221129-03f26f85.pth'> model </a>| <a href='https://download.openmmlab.com/mmselfsup/1.x/milan/milan_vit-base-p16_16xb256-amp-coslr-400e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k-milan_20221125_031826.json'>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/milan/classification/vit-base-p16_ft-8xb128-coslr-100e_in1k-milan.py'>config</a> |<a href='https://download.openmmlab.com/mmselfsup/1.x/milan/milan_vit-base-p16_16xb256-amp-coslr-400e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k-milan_20221129-74ac94fa.pth'> model </a>| <a href='https://download.openmmlab.com/mmselfsup/1.x/milan/milan_vit-base-p16_16xb256-amp-coslr-400e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k-milan_20221125_031826.json'>log</a></td>
	</tr>
</tbody>
</table>

## Citation

```bibtex
@article{Hou2022MILANMI,
  title={MILAN: Masked Image Pretraining on Language Assisted Representation},
  author={Zejiang Hou and Fei Sun and Yen-Kuang Chen and Yuan Xie and S. Y. Kung},
  journal={ArXiv},
  year={2022}
}
```
