# BEiT

> [BEiT: BERT Pre-Training of Image Transformers](https://arxiv.org/abs/2106.08254)

<!-- [ALGORITHM] -->

## Abstract

We introduce a self-supervised vision representation model BEiT, which stands for Bidirectional Encoder representation from Image Transformers. Following BERT developed in the natural language processing area, we propose a masked image modeling task to pretrain vision Transformers. Specifically, each image has two views in our pre-training, i.e, image patches (such as 16x16 pixels), and visual tokens (i.e., discrete tokens). We first "tokenize" the original image into visual tokens. Then we randomly mask some image patches and fed them into the backbone Transformer. The pre-training objective is to recover the original visual tokens based on the corrupted image patches. After pre-training BEiT, we directly fine-tune the model parameters on downstream tasks by appending task layers upon the pretrained encoder. Experimental results on image classification and semantic segmentation show that our model achieves competitive results with previous pre-training methods. For example, base-size BEiT achieves 83.2% top-1 accuracy on ImageNet-1K, significantly outperforming from-scratch DeiT training (81.8%) with the same setup. Moreover, large-size BEiT obtains 86.3% only using ImageNet-1K, even outperforming ViT-L with supervised pre-training on ImageNet-22K (85.2%).

<div align="center">
<img src="https://user-images.githubusercontent.com/36138628/203688351-adac7146-4e71-4ab6-8958-5cfe643a2dc5.png" width="70%"/>
</div>

## Results and models

### ImageNet-1k

|    Model    |   Pretrain   | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) |                 Config                  |                                                Download                                                 |
| :---------: | :----------: | :-------: | :------: | :-------: | :-------: | :-------------------------------------: | :-----------------------------------------------------------------------------------------------------: |
| BEiT-base\* | ImageNet-21k |   86.53   |  17.58   |   85.28   |   97.59   | [config](./beit-base-p16_8xb64_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/beit/beit-base_3rdparty_in1k_20221114-c0a4df23.pth) |

*Models with * are converted from the [official repo](https://github.com/microsoft/unilm/tree/master/beit). The config files of these models are only for inference.*

For BEiT self-supervised learning algorithm, welcome to [MMSelfSup page](https://github.com/open-mmlab/mmselfsup/tree/dev-1.x/configs/selfsup/beit) to get more information.

## Citation

```bibtex
@article{beit,
    title={{BEiT}: {BERT} Pre-Training of Image Transformers},
    author={Hangbo Bao and Li Dong and Furu Wei},
    year={2021},
    eprint={2106.08254},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
