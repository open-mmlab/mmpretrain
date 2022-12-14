# CSRA

> [Residual Attention: A Simple but Effective Method for Multi-Label Recognition](https://arxiv.org/abs/2108.02456)

<!-- [ALGORITHM] -->

## Abstract

Multi-label image recognition is a challenging computer vision task of practical use. Progresses in this area, however, are often characterized by complicated methods, heavy computations, and lack of intuitive explanations. To effectively capture different spatial regions occupied by objects from different categories, we propose an embarrassingly simple module, named class-specific residual attention (CSRA). CSRA generates class-specific features for every category by proposing a simple spatial attention score, and then combines it with the class-agnostic average pooling feature. CSRA achieves state-of-the-art results on multilabel recognition, and at the same time is much simpler than them. Furthermore, with only 4 lines of code, CSRA also leads to consistent improvement across many diverse pretrained models and datasets without any extra training. CSRA is both easy to implement and light in computations, which also enjoys intuitive explanations and visualizations.

<div align=center>
<img src="https://user-images.githubusercontent.com/84259897/176982245-3ffcff56-a4ea-4474-9967-bc2b612bbaa3.png" width="80%"/>
</div>

## Results and models

### VOC2007

|     Model      |                      Pretrain                      | Params(M) | Flops(G) |  mAP  | OF1 (%) | CF1 (%) |                      Config                       |                      Download                       |
| :------------: | :------------------------------------------------: | :-------: | :------: | :---: | :-----: | :-----: | :-----------------------------------------------: | :-------------------------------------------------: |
| Resnet101-CSRA | [ImageNet-1k](https://download.openmmlab.com/mmclassification/v0/resnet/resnet101_8xb32_in1k_20210831-539c63f8.pth) |   23.55   |   4.12   | 94.98 |  90.80  |  89.16  | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/csra/resnet101-csra_1xb16_voc07-448px.py) | [model](https://download.openmmlab.com/mmclassification/v0/csra/resnet101-csra_1xb16_voc07-448px_20220722-29efb40a.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/csra/resnet101-csra_1xb16_voc07-448px_20220722-29efb40a.log.json) |

## Citation

```bibtex
@misc{https://doi.org/10.48550/arxiv.2108.02456,
  doi = {10.48550/ARXIV.2108.02456},
  url = {https://arxiv.org/abs/2108.02456},
  author = {Zhu, Ke and Wu, Jianxin},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Residual Attention: A Simple but Effective Method for Multi-Label Recognition},
  publisher = {arXiv},
  year = {2021},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
