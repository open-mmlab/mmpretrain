# ConvViT

## Introduction

This is **ConvViT**, the hierarchical, mixed convolution-transformer encoder first proposed in [MCMAE](https://arxiv.org/abs/2205.03892). The architecture replaces the first transformer block in a ViT model with a few depth-wise convolution blocks at higher resolution to better capture local image features. The hierarchical features and local inductive bias benefits both coarse-grained (e.g., image classification, video recognition) and fine-grained (e.g., detection, segmentation) tasks.

Below is an overview of the **MCMAE** pretraining architecture. The gray shaded part is the **ConvViT** encoder architecture.

<div align="center">
<img src="https://github.com/Alpha-VL/ConvMAE/blob/main/figures/ConvMAE.png" width="100%" />
</div>

## Results and models

### ImageNet-1k

The following model weights are fine-tuned on ImageNet-1k.

With [MCMAE](https://arxiv.org/abs/2205.03892) pretraining:

| Model | Config | Download | 
| ----- | ------ | -------- |


## Citation

```
@inproceedings{gaomcmae,
  title={MCMAE: Masked Convolution Meets Masked Autoencoders},
  author={Gao, Peng and Ma, Teli and Li, Hongsheng and Lin, Ziyi and Dai, Jifeng and Qiao, Yu},
  booktitle={Advances in Neural Information Processing Systems}
}
```
