# BEiT

> [BEiT v2: Masked Image Modeling with Vector-Quantized Visual Tokenizers](https://arxiv.org/abs/2208.06366)

<!-- [ALGORITHM] -->

## Abstract

Masked image modeling (MIM) has demonstrated impressive results in self-supervised representation learning by recovering corrupted image patches. However, most existing studies operate on low-level image pixels, which hinders the exploitation of high-level semantics for representation models. In this work, we propose to use a semantic-rich visual tokenizer as the reconstruction target for masked prediction, providing a systematic way to promote MIM from pixel-level to semantic-level. Specifically, we propose vector-quantized knowledge distillation to train the tokenizer, which discretizes a continuous semantic space to compact codes. We then pretrain vision Transformers by predicting the original visual tokens for the masked image patches. Furthermore, we introduce a patch aggregation strategy which associates discrete image patches to enhance global semantic representation. Experiments on image classification and semantic segmentation show that BEiT v2 outperforms all compared MIM methods. On ImageNet-1K (224 size), the base-size BEiT v2 achieves 85.5% top-1 accuracy for fine-tuning and 80.1% top-1 accuracy for linear probing. The large-size BEiT v2 obtains 87.3% top-1 accuracy for ImageNet-1K (224 size) fine-tuning, and 56.7% mIoU on ADE20K for semantic segmentation.

<div align="center">
<img src="https://user-images.githubusercontent.com/36138628/203912182-5967a520-d455-49ea-bc67-dcbd500d76bf.png" width="70%"/>
</div>

## Results and models

### ImageNet-1k

|     Model     |   Pretrain   | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) |                  Config                   | Download  |
| :-----------: | :----------: | :-------: | :------: | :-------: | :-------: | :---------------------------------------: | :-------: |
| BEiTv2-base\* | ImageNet-21k |   86.86   |  33.03   |   86.47   |   97.99   | [config](./beitv2-base-p16_8xb64_in1k.py) | [model]() |


*Models with * are converted from the [official repo](https://github.com/microsoft/unilm/tree/master/beit2). The config files of these models are only for inference. We don't ensure these config files' training accuracy and welcome you to contribute your reproduction results.*

For BEiTv2 self-supervised learning algorithm, welcome to [MMSelfSup page](https://github.com/open-mmlab/mmselfsup/tree/dev-1.x/configs/selfsup/beitv2) to get more information.

## Citation

```bibtex
@article{beitv2,
    title={{BEiT v2}: Masked Image Modeling with Vector-Quantized Visual Tokenizers},
    author={Zhiliang Peng and Li Dong and Hangbo Bao and Qixiang Ye and Furu Wei},
    year={2022},
    eprint={2208.06366},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
