# Transformer in Transformer

## Introduction

<!-- [ALGORITHM] -->

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
| Transformer in Transformer small\* |   23.76  |  3.36 | 81.52 | 95.73 | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/tnt/tnt_s_patch16_224_evalonly_imagenet.py) | [model](https://download.openmmlab.com/mmclassification/v0/tnt/tnt-small-p16_3rdparty_in1k_20210903-c56ee7df.pth)  &#124; [log]()|

*Models with \* are converted from other repos.*

## Results and models

Waiting for adding.
