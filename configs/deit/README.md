# DeiT

> [Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877)
<!-- [ALGORITHM] -->

## Abstract

Recently, neural networks purely based on attention were shown to address image understanding tasks such as image classification. However, these visual transformers are pre-trained with hundreds of millions of images using an expensive infrastructure, thereby limiting their adoption.   In this work, we produce a competitive convolution-free transformer by training on Imagenet only. We train them on a single computer in less than 3 days. Our reference vision transformer (86M parameters) achieves top-1 accuracy of 83.1% (single-crop evaluation) on ImageNet with no external data.   More importantly, we introduce a teacher-student strategy specific to transformers. It relies on a distillation token ensuring that the student learns from the teacher through attention. We show the interest of this token-based distillation, especially when using a convnet as a teacher. This leads us to report results competitive with convnets for both Imagenet (where we obtain up to 85.2% accuracy) and when transferring to other tasks. We share our code and models.

<div align=center>
<img src="https://user-images.githubusercontent.com/26739999/143225703-c287c29e-82c9-4c85-a366-dfae30d198cd.png" width="40%"/>
</div>

## Results and models

### ImageNet-1k

The teacher of the distilled version DeiT is RegNetY-16GF.

|         Model         |    Pretrain  | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) | Config | Download |
|:---------------------:|:------------:|:---------:|:--------:|:---------:|:---------:|:------:|:--------:|
| DeiT-tiny             | From scratch | 5.72      | 1.08     | 74.50     | 92.24     | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/deit/deit-tiny_pt-4xb256_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/deit/deit-tiny_pt-4xb256_in1k_20220218-13b382a0.pth)  &#124; [log](https://download.openmmlab.com/mmclassification/v0/deit/deit-tiny_pt-4xb256_in1k_20220218-13b382a0.log.json) |
| DeiT-tiny distilled\* | From scratch | 5.72      | 1.08     | 74.51     | 91.90     | [config](https://github.com/open-mmlab/mmclassification/tree/master/configs/deit/deit-tiny-distilled_pt-4xb256_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/deit/deit-tiny-distilled_3rdparty_pt-4xb256_in1k_20211216-c429839a.pth) |
| DeiT-small            | From scratch | 22.05     | 4.24     | 80.69     | 95.06     | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/deit/deit-small_pt-4xb256_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/deit/deit-small_pt-4xb256_in1k_20220218-9425b9bb.pth)  &#124; [log](https://download.openmmlab.com/mmclassification/v0/deit/deit-small_pt-4xb256_in1k_20220218-9425b9bb.log.json) |
| DeiT-small distilled\*| From scratch | 22.05     | 4.24     | 81.17     | 95.40     | [config](https://github.com/open-mmlab/mmclassification/tree/master/configs/deit/deit-small-distilled_pt-4xb256_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/deit/deit-small-distilled_3rdparty_pt-4xb256_in1k_20211216-4de1d725.pth) |
| DeiT-base             | From scratch | 86.57     | 16.86    | 81.76     | 95.81     | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/deit/deit-base_pt-16xb64_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/deit/deit-base_pt-16xb64_in1k_20220216-db63c16c.pth)  &#124; [log](https://download.openmmlab.com/mmclassification/v0/deit/deit-base_pt-16xb64_in1k_20220216-db63c16c.log.json) |
| DeiT-base\*           | From scratch | 86.57     | 16.86    | 81.79     | 95.59     | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/deit/deit-base_pt-16xb64_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/deit/deit-base_3rdparty_pt-16xb64_in1k_20211124-6f40c188.pth) |
| DeiT-base distilled\* | From scratch | 86.57     | 16.86    | 83.33     | 96.49     | [config](https://github.com/open-mmlab/mmclassification/tree/master/configs/deit/deit-base-distilled_pt-16xb64_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/deit/deit-base-distilled_3rdparty_pt-16xb64_in1k_20211216-42891296.pth) |
| DeiT-base 384px\*     | ImageNet-1k  | 86.86     | 49.37    | 83.04     | 96.31     | [config](https://github.com/open-mmlab/mmclassification/tree/master/configs/deit/deit-base_ft-16xb32_in1k-384px.py) | [model](https://download.openmmlab.com/mmclassification/v0/deit/deit-base_3rdparty_ft-16xb32_in1k-384px_20211124-822d02f2.pth) |
| DeiT-base distilled 384px\* | ImageNet-1k | 86.86 | 49.37   | 85.55     | 97.35     | [config](https://github.com/open-mmlab/mmclassification/tree/master/configs/deit/deit-base-distilled_ft-16xb32_in1k-384px.py) | [model](https://download.openmmlab.com/mmclassification/v0/deit/deit-base-distilled_3rdparty_ft-16xb32_in1k-384px_20211216-e48d6000.pth) |

*Models with \* are converted from the [official repo](https://github.com/facebookresearch/deit). The config files of these models are only for validation. We don't ensure these config files' training accuracy and welcome you to contribute your reproduction results.*

```{warning}
MMClassification doesn't support training the distilled version DeiT.
And we provide distilled version checkpoints for inference only.
```

## Citation

```
@InProceedings{pmlr-v139-touvron21a,
  title =     {Training data-efficient image transformers &amp; distillation through attention},
  author =    {Touvron, Hugo and Cord, Matthieu and Douze, Matthijs and Massa, Francisco and Sablayrolles, Alexandre and Jegou, Herve},
  booktitle = {International Conference on Machine Learning},
  pages =     {10347--10357},
  year =      {2021},
  volume =    {139},
  month =     {July}
}
```
