# GCViT

> [Global Context Vision Transformers](https://arxiv.org/abs/2206.09959)

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

We propose global context vision transformer (GC ViT), a novel architecture that enhances parameter and compute utilization. Our method leverages global context self-attention modules, joint with local self-attention, to effectively yet efficiently model both long and short-range spatial interactions, without the need for expensive operations such as computing attention masks or shifting local windows. In addition, we address the issue of lack of the inductive bias in ViTs via proposing to use a modified fused inverted residual blocks in our architecture. Our proposed GC ViT achieves state-of-the-art results across image classification, object detection and semantic segmentation tasks. On ImageNet-1K dataset for classification, the tiny, small and base variants of GC ViT with 28M, 51M and 90M parameters achieve 83.3%, 83.9% and 84.5% Top-1 accuracy, respectively, surpassing comparably-sized prior art such as CNN-based ConvNeXt and ViT-based Swin Transformer by a large margin. Pre-trained GC ViT backbones in downstream tasks of object detection, instance segmentation, and semantic segmentation using MS COCO and ADE20K datasets outperform prior work consistently, sometimes by large margins.

<!-- [IMAGE] -->

<!-- <div align=center>
<img src="https://github.com/mmaaz60/EdgeNeXt/raw/main/images/EdgeNext.png" width="100%"/>
</div> -->

## Results and models

### ImageNet-1k

|     Model      |   Pretrain   | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) |           Config            |  Download   |
| :------------: | :----------: | :-------: | :------: | :-------: | :-------: | :-------------------------: | :---------: |
|  GCViT-base\*  | From scratch |    90     |   14.8   |   84.47   |   96.84   |  [config](./gcvit_base.py)  | [model](<>) |
| GCViT-small\*  | From scratch |    51     |   8.5    |   83.95   |   96.65   | [config](./gcvit_small.py)  | [model](<>) |
|  GCViT-tiny\*  | From scratch |    28     |   4.7    |   83.40   |   96.40   |  [config](./gcvit_tiny.py)  | [model](<>) |
| GCViT-xtiny\*  | From scratch |    20     |   2.6    |   82.04   |   95.99   | [config](./gcvit_xtiny.py)  | [model](<>) |
| GCViT-xxtiny\* | From scratch |    12     |   2.1    |   79.80   |   95.09   | [config](./gcvit_xxtiny.py) | [model](<>) |

*Models with * are converted from the [official repo](https://github.com/NVlabs/GCVit). The config files of these models are only for inference. We don't ensure these config files' training accuracy and welcome you to contribute your reproduction results.*

## Citation

```bibtex
@article{Hatamizadeh2022GlobalCV,
    title={Global Context Vision Transformers},
    author={Ali Hatamizadeh and Hongxu Yin and Jan Kautz and Pavlo Molchanov},
    journal={ArXiv},
    year={2022},
    volume={abs/2206.09959}
}
```
