# MLP-Mixer: An all-MLP Architecture for Vision
<!-- {Mlp-Mixer} -->

## Introduction

<!-- [ALGORITHM] -->

```latex
@misc{tolstikhin2021mlpmixer,
      title={MLP-Mixer: An all-MLP Architecture for Vision},
      author={Ilya Tolstikhin and Neil Houlsby and Alexander Kolesnikov and Lucas Beyer and Xiaohua Zhai and Thomas Unterthiner and Jessica Yung and Andreas Steiner and Daniel Keysers and Jakob Uszkoreit and Mario Lucic and Alexey Dosovitskiy},
      year={2021},
      eprint={2105.01601},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Pretrain model

The pre-trained modles are converted from [timm](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/mlp_mixer.py).

### ImageNet-1k

|      Model     | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) | Config | Download |
|:--------------:|:---------:|:--------:|:---------:|:---------:|:------:|:--------:|
| Mlp-Mixer_b16\* |  59.88   |  12.61    | 76.68     | 92.25     | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/mlp_mixer/mlp_mixer-base_p16_64xb64_in1k.py) | [model]()  &#124; [log]()|
| Mlp-Mixer_l16\* |   208.2   |  44.57    | 72.34     | 88.02     | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/mlp_mixer/mlp_mixer-large_p16_64xb64_in1k.py) | [model]()  &#124; [log]()|

*Models with \* are converted from other repos.*
