# Res2Net: A New Multi-scale Backbone Architecture
<!-- {Res2Net} -->

## Introduction

<!-- [ALGORITHM] -->

```latex
@article{gao2019res2net,
  title={Res2Net: A New Multi-scale Backbone Architecture},
  author={Gao, Shang-Hua and Cheng, Ming-Ming and Zhao, Kai and Zhang, Xin-Yu and Yang, Ming-Hsuan and Torr, Philip},
  journal={IEEE TPAMI},
  year={2021},
  doi={10.1109/TPAMI.2019.2938758},
}
```

## Pretrain model

The pre-trained models are converted from [official repo](https://github.com/Res2Net/Res2Net-PretrainedModels).

### ImageNet 1k

|        Model          | resolution  | Params(M) |  Flops(G) | Top-1 (%) | Top-5 (%) | Download |
|:---------------------:|:-----------:|:---------:|:---------:|:---------:|:---------:|:--------:|
|  Res2Net-50-14w-8s\*  |   224x224   |   25.06   |    4.22   |   78.14   |   93.85   | [model](https://download.openmmlab.com/mmclassification/v0/res2net/res2net50-w14-s8_3rdparty_8xb32_in1k_20210927-bc967bf1.pth)|
|  Res2Net-50-26w-8s\*  |   224x224   |   48.40   |    8.39   |   79.20   |   94.36   | [model](https://download.openmmlab.com/mmclassification/v0/res2net/res2net50-w26-s8_3rdparty_8xb32_in1k_20210927-f547a94b.pth)|
|  Res2Net-101-26w-4s\* |   224x224   |   45.21   |    8.12   |   79.19   |   94.44   | [model](https://download.openmmlab.com/mmclassification/v0/res2net/res2net101-w26-s4_3rdparty_8xb32_in1k_20210927-870b6c36.pth)|

*Models with \* are converted from other repos.*
