# CAE

> [Context Autoencoder for Self-Supervised Representation Learning](https://arxiv.org/abs/2202.03026)

<!-- [ALGORITHM] -->

## Abstract

We present a novel masked image modeling (MIM) approach, context autoencoder (CAE), for self-supervised learning. We randomly partition the image into two sets: visible patches and masked patches. The CAE architecture consists of: (i) an encoder that takes visible patches as input and outputs their latent representations, (ii) a latent context regressor that predicts the masked patch representations from the visible patch representations that are not updated in this regressor, (iii) a decoder that takes the estimated masked patch representations as input and makes predictions for the masked patches, and (iv) an alignment module that aligns the masked patch representation estimation with the masked patch representations computed from the encoder. In comparison to previous MIM methods that couple the encoding and decoding roles, e.g., using a single module in BEiT, our approach attempts to separate the encoding role (content understanding) from the decoding role (making predictions for masked patches) using different modules, improving the content understanding capability. In addition, our approach makes predictions from the visible patches to the masked patches in the latent representation space that is expected to take on semantics. In addition, we present the explanations about why contrastive pretraining and supervised pretraining perform similarly and why MIM potentially performs better. We demonstrate the effectiveness of our CAE through superior transfer performance in downstream tasks: semantic segmentation, and object detection and instance segmentation.

<div align="center">
<img src="https://user-images.githubusercontent.com/30762564/165459947-6c6ef13c-0593-4765-b44e-6da0a079802a.png" width="40%"/>
</div>

## Prerequisite

Create a new folder `cae_ckpt` under the root directory and download the
[weights](https://download.openmmlab.com/mmselfsup/cae/dalle_encoder.pth) for `dalle` encoder to that folder

## Models and Benchmarks

Here, we report the results of the model, which is pre-trained on ImageNet-1k
for 300 epochs, the details are below:

| Backbone | Pre-train epoch | Fine-tuning Top-1 |                                                         Pre-train Config                                                          |                                                                   Fine-tuning Config                                                                   |                                                                                                                        Download                                                                                                                         |
| :------: | :-------------: | :---------------: | :-------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ViT-B/16 |       300       |       83.2        | [config](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/cae/cae_vit-base-p16_8xb256-fp16-coslr-300e_in1k.py) | [config](https://github.com/open-mmlab/mmselfsup/blob/master/configs/benchmarks/classification/imagenet/vit-base-p16_ft-8xb128-coslr-100e-rpe_in1k.py) | [model](https://download.openmmlab.com/mmselfsup/cae/cae_vit-base-p16_16xb256-coslr-300e_in1k-224_20220427-4c786349.pth) \| [log](https://download.openmmlab.com/mmselfsup/cae/cae_vit-base-p16_16xb256-coslr-300e_in1k-224_20220427-4c786349.log.json) |

## Citation

```bibtex
@article{CAE,
  title={Context Autoencoder for Self-Supervised Representation Learning},
  author={Xiaokang Chen, Mingyu Ding, Xiaodi Wang, Ying Xin, Shentong Mo,
  Yunhao Wang, Shumin Han, Ping Luo, Gang Zeng, Jingdong Wang},
  journal={ArXiv},
  year={2022}
}
```
