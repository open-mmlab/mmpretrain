# MAE

> [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377)

<!-- [ALGORITHM] -->

## Abstract

This paper shows that masked autoencoders (MAE) are
scalable self-supervised learners for computer vision. Our
MAE approach is simple: we mask random patches of the
input image and reconstruct the missing pixels. It is based
on two core designs. First, we develop an asymmetric
encoder-decoder architecture, with an encoder that operates only on the
visible subset of patches (without mask tokens), along with a lightweight
decoder that reconstructs the original image from the latent representation
and mask tokens. Second, we find that masking a high proportion
of the input image, e.g., 75%, yields a nontrivial and
meaningful self-supervisory task. Coupling these two designs enables us to
train large models efficiently and effectively: we accelerate
training (by 3× or more) and improve accuracy. Our scalable approach allows
for learning high-capacity models that generalize well: e.g., a vanilla
ViT-Huge model achieves the best accuracy (87.8%) among
methods that use only ImageNet-1K data. Transfer performance in downstream tasks outperforms supervised pretraining and shows promising scaling behavior.

<div align="center">
<img src="https://user-images.githubusercontent.com/30762564/150733959-2959852a-c7bd-4d3f-911f-3e8d8839fe67.png" width="40%"/>
</div>

## Models and Benchmarks

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
      <td rowspan="9">MAE</td>
	    <td>ViT-base</td>
	    <td>300</td>
      <td>4096</td>
      <td>60.8</td>
      <td>83.1</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/mae/mae_vit-base-p16_8xb512-amp-coslr-300e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-base-p16_8xb512-fp16-coslr-300e_in1k/mae_vit-base-p16_8xb512-coslr-300e-fp16_in1k_20220829-c2cf66ba.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-base-p16_8xb512-fp16-coslr-300e_in1k/mae_vit-base-p16_8xb512-coslr-300e-fp16_in1k_20220718_152424.json'>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/vit-base-p16_linear-8xb2048-coslr-90e_in1k.py'>config</a> | model | <a href='https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-base-p16_8xb512-fp16-coslr-300e_in1k/vit-base-p16_linear-8xb2048-coslr-90e_in1k/vit-base-p16_linear-8xb2048-coslr-90e_in1k_20220720_104514.json'>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/vit-base-p16_ft-8xb128-coslr-100e_in1k.py'>config</a> | model | <a href='https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-base-p16_8xb512-fp16-coslr-300e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k_20220713_140138.json'>log</a></td>
	</tr>
  <tr>
	    <td>ViT-base</td>
	    <td>400</td>
      <td>4096</td>
      <td>62.5</td>
      <td>83.3</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/mae/mae_vit-base-p16_8xb512-amp-coslr-400e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-base-p16_8xb512-fp16-coslr-400e_in1k/mae_vit-base-p16_8xb512-coslr-400e-fp16_in1k_20220825-bc79e40b.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-base-p16_8xb512-fp16-coslr-400e_in1k/mae_vit-base-p16_8xb512-coslr-400e-fp16_in1k_20220628_200815.json'>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/vit-base-p16_linear-8xb2048-coslr-90e_in1k.py'>config</a> | model | <a href='https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-base-p16_8xb512-fp16-coslr-400e_in1k/vit-base-p16_linear-8xb2048-coslr-90e_in1k/vit-base-p16_linear-8xb2048-coslr-90e_in1k_20220713_142534.json'>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/vit-base-p16_ft-8xb128-coslr-100e_in1k.py'>config</a> | model | <a href='https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-base-p16_8xb512-fp16-coslr-400e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k_20220708_183134.json'>log</a></td>
	</tr>
  <tr>
	    <td>ViT-base</td>
	    <td>800</td>
      <td>4096</td>
      <td>65.1</td>
      <td>83.3</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/mae/mae_vit-base-p16_8xb512-amp-coslr-800e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-base-p16_8xb512-fp16-coslr-800e_in1k/mae_vit-base-p16_8xb512-coslr-800e-fp16_in1k_20220825-5d81fbc4.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-base-p16_8xb512-fp16-coslr-800e_in1k/mae_vit-base-p16_8xb512-coslr-800e-fp16_in1k_20220718_134405.json'>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/vit-base-p16_linear-8xb2048-coslr-90e_in1k.py'>config</a> | model | <a href='https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-base-p16_8xb512-fp16-coslr-800e_in1k/vit-base-p16_linear-8xb2048-coslr-90e_in1k/vit-base-p16_linear-8xb2048-coslr-90e_in1k20220721_203941.json'>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/vit-base-p16_ft-8xb128-coslr-100e_in1k.py'>config</a> | model | <a href='https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-base-p16_8xb512-fp16-coslr-800e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k_20220724_232940.json'>log</a></td>
	</tr>
  <tr>
	    <td>ViT-base</td>
	    <td>1600</td>
      <td>4096</td>
      <td>67.1</td>
      <td>83.5</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/mae/mae_vit-base-p16_8xb512-amp-coslr-1600e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-base-p16_8xb512-fp16-coslr-1600e_in1k/mae_vit-base-p16_8xb512-fp16-coslr-1600e_in1k_20220825-f7569ca2.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-base-p16_8xb512-fp16-coslr-1600e_in1k/mae_vit-base-p16_8xb512-fp16-coslr-1600e_in1k_20220815_103458.json'>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/vit-base-p16_linear-8xb2048-coslr-90e_in1k.py'>config</a> | model | <a href='https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-base-p16_8xb512-fp16-coslr-1600e_in1k/vit-base-p16_linear-8xb2048-coslr-90e_in1k/vit-base-p16_linear-8xb2048-coslr-90e_in1k_20220724_232557.json'>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/vit-base-p16_ft-8xb128-coslr-100e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-base-p16_8xb512-fp16-coslr-1600e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k_20220825-cf70aa21.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-base-p16_8xb512-fp16-coslr-1600e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k_20220721_202304.json'>log</a></td>
	</tr>
  <tr>
	    <td>ViT-large</td>
	    <td>400</td>
      <td>4096</td>
      <td>70.7</td>
      <td>85.2</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/mae/mae_vit-large-p16_8xb512-amp-coslr-400e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-large-p16_8xb512-fp16-coslr-400e_in1k/mae_vit-large-p16_8xb512-fp16-coslr-400e_in1k_20220825-b11d0425.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-large-p16_8xb512-fp16-coslr-400e_in1k/mae_vit-large-p16_8xb512-fp16-coslr-400e_in1k_20220726_202204.json'>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/vit-large-p16_linear-8xb2048-coslr-90e_in1k.py'>config</a> | model | <a href='https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-large-p16_8xb512-fp16-coslr-400e_in1k/vit-large-p16_linear-8xb2048-coslr-90e_in1k/vit-large-p16_linear-8xb2048-coslr-90e_in1k_20220803_101331.json'>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/vit-large-p16_ft-8xb128-coslr-50e_in1k.py'>config</a> | model | <a href='https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-large-p16_8xb512-fp16-coslr-400e_in1k/vit-large-p16_ft-8xb128-coslr-50e_in1k/vit-large-p16_ft-8xb128-coslr-50e_in1k_20220729_122511.json'>log</a></td>
	</tr>
  <tr>
	    <td>ViT-large</td>
	    <td>800</td>
      <td>4096</td>
      <td>73.7</td>
      <td>85.4</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/mae/mae_vit-large-p16_8xb512-amp-coslr-800e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-large-p16_8xb512-fp16-coslr-800e_in1k/mae_vit-large-p16_8xb512-fp16-coslr-800e_in1k_20220825-df72726a.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-large-p16_8xb512-fp16-coslr-800e_in1k/mae_vit-large-p16_8xb512-fp16-coslr-800e_in1k_20220804_104018.json'>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/vit-large-p16_linear-8xb2048-coslr-90e_in1k.py'>config</a> | model | <a href='https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-large-p16_8xb512-fp16-coslr-800e_in1k/vit-large-p16_linear-8xb2048-coslr-90e_in1k/vit-large-p16_linear-8xb2048-coslr-90e_in1k_20220808_092730.json'>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/vit-large-p16_ft-8xb128-coslr-50e_in1k.py'>config</a> | model | <a href='https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-large-p16_8xb512-fp16-coslr-800e_in1k/vit-large-p16_ft-8xb128-coslr-50e_in1k/vit-large-p16_ft-8xb128-coslr-50e_in1k_20220730_235819.json'>log</a></td>
	</tr>
  <tr>
	    <td>ViT-large</td>
	    <td>1600</td>
      <td>4096</td>
      <td>75.5</td>
      <td>85.7</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/mae/mae_vit-large-p16_8xb512-amp-coslr-1600e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-large-p16_8xb512-fp16-coslr-1600e_in1k/mae_vit-large-p16_8xb512-fp16-coslr-1600e_in1k_20220825-cc7e98c9.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-large-p16_8xb512-fp16-coslr-1600e_in1k/mae_vit-large-p16_8xb512-fp16-coslr-1600e_in1k_20220806_210725.json'>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/vit-large-p16_linear-8xb2048-coslr-90e_in1k.py'>config</a> | model | <a href='https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-large-p16_8xb512-fp16-coslr-1600e_in1k/vit-large-p16_linear-8xb2048-coslr-90e_in1k/vit-large-p16_linear-8xb2048-coslr-90e_in1k_20220813_155615.json'>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/vit-large-p16_ft-8xb128-coslr-50e_in1k.py'>config</a> | model | <a href='https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-large-p16_8xb512-fp16-coslr-1600e_in1k/vit-large-p16_ft-8xb128-coslr-50e_in1k/vit-large-p16_ft-8xb128-coslr-50e_in1k_20220813_125305.json'>log</a></td>
	</tr>
  <tr>
	    <td>ViT-huge-FT-224</td>
	    <td>1600</td>
      <td>4096</td>
      <td>/</td>
      <td>86.9</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/mae/mae_vit-huge-p16_8xb512-amp-coslr-1600e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-huge-p16_8xb512-fp16-coslr-1600e_in1k/mae_vit-huge-p16_8xb512-fp16-coslr-1600e_in1k_20220916-ff848775.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-huge-p16_8xb512-fp16-coslr-1600e_in1k/mae_vit-huge-p16_8xb512-fp16-coslr-1600e_in1k_20220814_135241.json'>log</a></td>
      <td>/</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/vit-huge-p16_ft-8xb128-coslr-50e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-huge-p16_8xb512-fp16-coslr-1600e_in1k/vit-huge-p16_ft-8xb128-coslr-50e_in1k/vit-huge-p16_ft-8xb128-coslr-50e_in1k_20220916-0bfc9bfd.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-huge-p16_8xb512-fp16-coslr-1600e_in1k/vit-huge-p16_ft-8xb128-coslr-50e_in1k/vit-huge-p16_ft-8xb128-coslr-50e_in1k_20220829_114027.json'>log</a></td>
	</tr>
  <tr>
	    <td>ViT-huge-FT-448</td>
	    <td>1600</td>
      <td>4096</td>
      <td>/</td>
      <td>87.3</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/mae/mae_vit-huge-p16_8xb512-amp-coslr-1600e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-huge-p16_8xb512-fp16-coslr-1600e_in1k/mae_vit-huge-p16_8xb512-fp16-coslr-1600e_in1k_20220916-ff848775.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-huge-p16_8xb512-fp16-coslr-1600e_in1k/mae_vit-huge-p16_8xb512-fp16-coslr-1600e_in1k_20220814_135241.json'>log</a></td>
      <td>/</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/vit-huge-p16_ft-32xb8-coslr-50e_in1k-448.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-huge-p16_8xb512-fp16-coslr-1600e_in1k/vit-huge-p16_ft-32xb8-coslr-50e_in1k-448/vit-huge-p16_ft-32xb8-coslr-50e_in1k-448_20220916-95b6a0ce.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-huge-p16_8xb512-fp16-coslr-1600e_in1k/vit-huge-p16_ft-32xb8-coslr-50e_in1k-448/vit-huge-p16_ft-32xb8-coslr-50e_in1k-448_20220913_113737.json'>log</a></td>
	</tr>
</tbody>
</table>

## Evaluating MAE on Detection and Segmentation

If you want to evaluate your model on detection or segmentation task, we provide a [script](https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/tools/model_converters/mmcls2timm.py) to convert the model keys from MMClassification style to timm style.

```sh
cd $MMSELFSUP
python tools/model_converters/mmcls2timm.py $src_ckpt $dst_ckpt
```

Then, using this converted ckpt, you can evaluate your model on detection task, following [Detectron2](https://github.com/facebookresearch/detectron2/tree/main/projects/ViTDet)，
and on semantic segmentation task, following this [project](https://github.com/implus/mae_segmentation). Besides, using the unconverted ckpt, you can use
[MMSegmentation](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/mae) to evaluate your model.

## Citation

```bibtex
@article{He2021MaskedAA,
  title={Masked Autoencoders Are Scalable Vision Learners},
  author={Kaiming He and Xinlei Chen and Saining Xie and Yanghao Li and
  Piotr Doll'ar and Ross B. Girshick},
  journal={arXiv},
  year={2021}
}
```
