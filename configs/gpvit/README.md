# GPViT

> [GPViT: A High Resolution Non-Hierarchical Vision Transformer with Group Propagation](https://arxiv.org/abs/2212.06795)

<!-- [ALGORITHM] -->

## Abstract

We present the Group Propagation Vision Transformer (GPViT): a novel nonhierarchical (i.e. non-pyramidal) transformer model designed for general visual recognition with high-resolution features. High-resolution features (or tokens) are a natural fit for tasks that involve perceiving fine-grained details such as detection and segmentation, but exchanging global information between these features is expensive in memory and computation because of the way self-attention scales. We provide a highly efficient alternative Group Propagation Block (GP Block) to exchange global information. In each GP Block, features are first grouped together by a fixed number of learnable group tokens; we then perform Group Propagation where global information is exchanged between the grouped features; finally, global information in the updated grouped features is returned back to the image features through a transformer decoder. We evaluate GPViT on a variety of visual recognition tasks including image classification, semantic segmentation, object detection, and instance segmentation. Our method achieves significant performance gains over previous works across all tasks, especially on tasks that require high-resolution outputs, for example, our GPViT-L3 outperforms Swin Transformer-B by 2.0 mIoU on ADE20K semantic segmentation with only half as many parameters.

<div align="center">
<img src="https://user-images.githubusercontent.com/24734142/210958699-8bd04b60-70d4-4cde-9bee-8631dcef3931.png" width="70%"/>
</div>

## Results and models

### ImageNet-1k

| Model                                 |   Pretrain   | resolution | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) |                Config                |  Download   |
| :------------------------------------ | :----------: | :--------: | :-------: | :------: | :-------: | :-------: | :----------------------------------: | :---------: |
| GPViT-L1 (`gpvit-l1_3rdparty_in1k`)\* | From scratch |  224x224   |   87.90   |   9.60   |   6.13    |   95.32   | [config](./gpvit-l1_16xb128_in1k.py) | [model](<>) |
| GPViT-L2 (`gpvit-l2_3rdparty_in1k`)\* | From scratch |  224x224   |   24.16   |  15.70   |   83.19   |   96.53   | [config](./gpvit-l2_16xb128_in1k.py) | [model](<>) |
| GPViT-L3 (`gpvit-l3_3rdparty_in1k`)\* | From scratch |  224x224   |   36.65   |  23.50   |   83.71   |   96.78   | [config](./gpvit-l3_16xb128_in1k.py) | [model](<>) |
| GPViT-L4 (`gpvit-l4_3rdparty_in1k`)\* | From scratch |  224x224   |   75.48   |  48.24   |   84.13   |   96.82   | [config](./gpvit-l4_16xb128_in1k.py) | [model](<>) |

*Models with * are converted from the [official repo](https://github.com/ChenhongyiYang/GPViT). The config files of these models are only for inference.*

## Citation

```bibtex
@article{yang2022gpvit,
      title={GPViT: A High Resolution Non-Hierarchical Vision Transformer with Group Propagation},
      author={Chenhongyi Yang and Jiarui Xu and Shalini De Mello and Elliot J. Crowley and Xiaolong Wang},
      journal={arXiv preprint 2212.06795}
      year={2022},
}
```
