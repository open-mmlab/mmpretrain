# Transformer in Transformer

<!-- [ALGORITHM] -->
## Abstract
<!-- [ABSTRACT] -->
Transformer is a new kind of neural architecture which encodes the input data as powerful features via the attention mechanism. Basically, the visual transformers first divide the input images into several local patches and then calculate both representations and their relationship. Since natural images are of high complexity with abundant detail and color information, the granularity of the patch dividing is not fine enough for excavating features of objects in different scales and locations. In this paper, we point out that the attention inside these local patches are also essential for building visual transformers with high performance and we explore a new architecture, namely, Transformer iN Transformer (TNT). Specifically, we regard the local patches (e.g., 16×16) as "visual sentences" and present to further divide them into smaller patches (e.g., 4×4) as "visual words". The attention of each word will be calculated with other words in the given visual sentence with negligible computational costs. Features of both words and sentences will be aggregated to enhance the representation ability. Experiments on several benchmarks demonstrate the effectiveness of the proposed TNT architecture, e.g., we achieve an 81.5% top-1 accuracy on the ImageNet, which is about 1.7% higher than that of the state-of-the-art visual transformer with similar computational cost.

<!-- [IMAGE] -->
<div align=center>
<img src="https://user-images.githubusercontent.com/26739999/142578661-298d92a1-2e25-4910-a312-085587be6b65.png" width="80%"/>
</div>

## Citation
```latex
@misc{han2021transformer,
      title={Transformer in Transformer},
      author={Kai Han and An Xiao and Enhua Wu and Jianyuan Guo and Chunjing Xu and Yunhe Wang},
      year={2021},
      eprint={2103.00112},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Pretrain model

The pre-trained modles are converted from [timm](https://github.com/rwightman/pytorch-image-models/).

### ImageNet

|         Model         | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) | Download |
|:---------------------:|:---------:|:--------:|:---------:|:---------:|:--------:|
| Transformer in Transformer small\* |   23.76  |  3.36 | 81.52 | 95.73 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/tnt/tnt-s-p16_16xb64_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/tnt/tnt-small-p16_3rdparty_in1k_20210903-c56ee7df.pth)  &#124; [log]()|

*Models with \* are converted from other repos.*

## Results and models

Waiting for adding.
