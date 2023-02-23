# SimMIM

> [SimMIM: A Simple Framework for Masked Image Modeling](https://arxiv.org/abs/2111.09886)

<!-- [ALGORITHM] -->

## Abstract

This paper presents SimMIM, a simple framework for masked image modeling. We simplify recently proposed related approaches without special designs such as blockwise masking and tokenization via discrete VAE or clustering. To study what let the masked image modeling task learn good representations, we systematically study the major components in our framework, and find that simple designs of each component have revealed very strong representation learning performance: 1) random masking of the input image with a moderately large masked patch size (e.g., 32) makes a strong pre-text task; 2) predicting raw pixels of RGB values by direct regression performs no worse than the patch classification approaches with complex designs; 3) the prediction head can be as light as a linear layer, with no worse performance than heavier ones. Using ViT-B, our approach achieves 83.8% top-1 fine-tuning accuracy on ImageNet-1K by pre-training also on this dataset, surpassing previous best approach by +0.6%. When applied on a larger model of about 650 million parameters, SwinV2H, it achieves 87.1% top-1 accuracy on ImageNet-1K using only ImageNet-1K data. We also leverage this approach to facilitate the training of a 3B model (SwinV2-G), that by 40× less data than that in previous practice, we achieve the state-of-the-art on four representative vision benchmarks. The code and models will be publicly available at https: //github.com/microsoft/SimMIM .

<div align="center">
<img src="https://user-images.githubusercontent.com/30762564/159404597-ac6d3a44-ee59-4cdc-8f6f-506a7d1b18b6.png" width="40%"/>
</div>

## Models and Benchmarks

Here, we report the results of the model, and more results will be coming soon.

<table class="docutils">
<thead>
  <tr>
	    <th rowspan="2">Algorithm</th>
	    <th rowspan="2">Backbone</th>
	    <th rowspan="2">Epoch</th>
      <th rowspan="2">Fine-tuning Size</th>
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
      <td>SimMIM</td>
	    <td>Swin-base</td>
	    <td>100</td>
      <td>192</td>
      <td>2048</td>
      <td>/</td>
      <td>82.7</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/simmim/simmim_swin-base_16xb128-amp-coslr-100e_in1k-192.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/simmim/simmim_swin-base_8xb256-amp-coslr-100e_in1k-192/simmim_swin-base_8xb256-amp-coslr-100e_in1k-192_20220829-0e15782d.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/simmim/simmim_swin-base_8xb256-amp-coslr-100e_in1k-192/simmim_swin-base_8xb256-amp-coslr-100e_in1k-192_20220827_034052.json'>log</a></td>
      <td>/</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/swin-base_ft-8xb256-coslr-100e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/simmim/simmim_swin-base_8xb256-amp-coslr-100e_in1k-192/swin-base_ft-8xb256-coslr-100e_in1k/swin-base_ft-8xb256-coslr-100e_in1k_20220829-9cf23aa1.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/simmim/simmim_swin-base_8xb256-amp-coslr-100e_in1k-192/swin-base_ft-8xb256-coslr-100e_in1k/swin-base_ft-8xb256-coslr-100e_in1k_20220829_001452.json'>log</a></td>
	</tr>
  <tr>
      <td>SimMIM</td>
	    <td>Swin-base</td>
	    <td>100</td>
      <td>224</td>
      <td>2048</td>
      <td>/</td>
      <td>83.5</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/simmim/simmim_swin-base_16xb128-amp-coslr-100e_in1k-192.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/simmim/simmim_swin-base_8xb256-amp-coslr-100e_in1k-192/simmim_swin-base_8xb256-amp-coslr-100e_in1k-192_20220829-0e15782d.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/simmim/simmim_swin-base_8xb256-amp-coslr-100e_in1k-192/simmim_swin-base_8xb256-amp-coslr-100e_in1k-192_20220827_034052.json'>log</a></td>
      <td>/</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/swin-base_ft-8xb256-coslr-100e_in1k-224.py'>config</a> | model | log</td>
	</tr>
  <tr>
      <td>SimMIM</td>
	    <td>Swin-base</td>
	    <td>800</td>
      <td>224</td>
      <td>2048</td>
      <td>/</td>
      <td>83.7</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/simmim/simmim_swin-base_16xb128-amp-coslr-800e_in1k-192.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/simmim/simmim_swin-base_16xb128-amp-coslr-800e_in1k-192/simmim_swin-base_16xb128-amp-coslr-800e_in1k-192_20220916-a0e931ac.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/simmim/simmim_swin-base_16xb128-amp-coslr-800e_in1k-192/simmim_swin-base_16xb128-amp-coslr-800e_in1k-192_20220906_141645.json'>log</a></td>
      <td>/</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/swin-base_ft-8xb256-coslr-100e_in1k-224.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/simmim/simmim_swin-base_16xb128-amp-coslr-800e_in1k-192/swin-base_ft-8xb256-coslr-100e_in1k-224/swin-base_ft-8xb256-coslr-100e_in1k-224_20221208-155cc6e6.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/simmim/simmim_swin-base_16xb128-amp-coslr-800e_in1k-192/swin-base_ft-8xb256-coslr-100e_in1k-224/swin-base_ft-8xb256-coslr-100e_in1k-224_20221207_135847.json'>log</a></td>
	</tr>
  <tr>
      <td>SimMIM</td>
	    <td>Swin-large</td>
	    <td>800</td>
      <td>224</td>
      <td>2048</td>
      <td>/</td>
      <td>84.8</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/simmim/simmim_swin-large_16xb128-amp-coslr-800e_in1k-192.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/simmim/simmim_swin-large_16xb128-amp-coslr-800e_in1k-192/simmim_swin-large_16xb128-amp-coslr-800e_in1k-192_20220916-4ad216d3.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/simmim/simmim_swin-large_16xb128-amp-coslr-800e_in1k-192/simmim_swin-large_16xb128-amp-coslr-800e_in1k-192_20220907_203738.json'>log</a></td>
      <td>/</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/swin-large_ft-8xb256-coslr-ws14-100e_in1k-224.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/simmim/simmim_swin-large_16xb128-amp-coslr-800e_in1k-192/swin-large_ft-8xb256-coslr-ws14-100e_in1k-224/swin-large_ft-8xb256-coslr-ws14-100e_in1k-224_20220916-d4865790.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/simmim/simmim_swin-large_16xb128-amp-coslr-800e_in1k-192/swin-large_ft-8xb256-coslr-ws14-100e_in1k-224/swin-large_ft-8xb256-coslr-ws14-100e_in1k-224_20220914_133331.json'>log</a></td>
	</tr>
</tbody>
</table>

## Citation

```bibtex
@inproceedings{xie2021simmim,
  title={SimMIM: A Simple Framework for Masked Image Modeling},
  author={Xie, Zhenda and Zhang, Zheng and Cao, Yue and Lin, Yutong and Bao, Jianmin and Yao, Zhuliang and Dai, Qi and Hu, Han},
  booktitle={International Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022}
}
```
