# MFF

> [Improving Pixel-based MIM by Reducing Wasted Modeling Capability](https://arxiv.org/abs/2308.00261)

<!-- [ALGORITHM] -->

## Abstract

There has been significant progress in Masked Image Modeling (MIM). Existing MIM methods can be broadly categorized into two groups based on the reconstruction target: pixel-based and tokenizer-based approaches. The former offers a simpler pipeline and lower computational cost, but it is known to be biased toward high-frequency details. In this paper, we provide a set of empirical studies to confirm this limitation of pixel-based MIM and propose a new method that explicitly utilizes low-level features from shallow layers to aid pixel reconstruction. By incorporating this design into our base method, MAE, we reduce the wasted modeling capability of pixel-based MIM, improving its convergence and achieving non-trivial improvements across various downstream tasks. To the best of our knowledge, we are the first to systematically investigate multi-level feature fusion for isotropic architectures like the standard Vision Transformer (ViT). Notably, when applied to a smaller model (e.g., ViT-S), our method yields significant performance gains, such as 1.2% on fine-tuning, 2.8% on linear probing, and 2.6% on semantic segmentation.

<div align=center>
<img src="https://user-images.githubusercontent.com/30762564/257412932-5f36b11b-ee64-4ce7-b7d1-a31000302bd8.png" width="80%"/>
</div>

**Train/Test Command**

Prepare your dataset according to the [docs](https://mmpretrain.readthedocs.io/en/latest/user_guides/dataset_prepare.html#prepare-dataset).

Train:

```shell
python tools/train.py configs/mff/mff_vit-base-p16_8xb512-amp-coslr-300e_in1k.py
```

Test:

```shell
python tools/test.py configs/mff/benchmarks/vit-base-p16_8xb128-coslr-100e_in1k.py None
```

<!-- [TABS-END] -->

## Models and results

### Pretrained models

| Model                                         | Params (M) | Flops (G) |                          Config                          |                                     Download                                     |
| :-------------------------------------------- | :--------: | :-------: | :------------------------------------------------------: | :------------------------------------------------------------------------------: |
| `mff_vit-base-p16_8xb512-amp-coslr-300e_in1k` |     -      |     -     | [config](mff_vit-base-p16_8xb512-amp-coslr-300e_in1k.py) | [model](https://download.openmmlab.com/mmpretrain/v1.0/mff/mff_vit-base-p16_8xb512-amp-coslr-300e_in1k/mff_vit-base-p16_8xb512-amp-coslr-300e_in1k_20230801-3c1bcce4.pth) \| [log](https://download.openmmlab.com/mmpretrain/v1.0/mff/mff_vit-base-p16_8xb512-amp-coslr-300e_in1k/mff_vit-base-p16_8xb512-amp-coslr-300e_in1k_20230801-3c1bcce4.json) |
| `mff_vit-base-p16_8xb512-amp-coslr-800e_in1k` |     -      |     -     | [config](mff_vit-base-p16_8xb512-amp-coslr-300e_in1k.py) | [model](https://download.openmmlab.com/mmpretrain/v1.0/mff/mff_vit-base-p16_8xb512-amp-coslr-800e_in1k/mff_vit-base-p16_8xb512-amp-coslr-800e_in1k_20230801-3af7cd9d.pth) \| [log](https://download.openmmlab.com/mmpretrain/v1.0/mff/mff_vit-base-p16_8xb512-amp-coslr-800e_in1k/mff_vit-base-p16_8xb512-amp-coslr-800e_in1k_20230801-3af7cd9d.json) |

### Image Classification on ImageNet-1k

| Model                                     |                   Pretrain                   | Params (M) | Flops (G) | Top-1 (%) |                   Config                   |                   Download                    |
| :---------------------------------------- | :------------------------------------------: | :--------: | :-------: | :-------: | :----------------------------------------: | :-------------------------------------------: |
| `vit-base-p16_mff-300e-pre_8xb128-coslr-100e_in1k` | [MFF 300-Epochs](https://download.openmmlab.com/mmpretrain/v1.0/mff/mff_vit-base-p16_8xb512-amp-coslr-300e_in1k/mff_vit-base-p16_8xb512-amp-coslr-300e_in1k_20230801-3c1bcce4.pth) |   86.57    |   17.58   |   83.00   | [config](benchmarks/vit-base-p16_8xb128-coslr-100e_in1k.py) | [model](https://download.openmmlab.com/mmpretrain/v1.0/mff/mff_vit-base-p16_8xb512-amp-coslr-300e_in1k/vit-base-p16_8xb128-coslr-100e_in1k/vit-base-p16_8xb128-coslr-100e_in1k_20230802-d746fdb7.pth)  /   [log](https://download.openmmlab.com/mmpretrain/v1.0/mff/mff_vit-base-p16_8xb512-amp-coslr-300e_in1k/vit-base-p16_8xb128-coslr-100e_in1k/vit-base-p16_8xb128-coslr-100e_in1k_20230802-d746fdb7.json) |
| `vit-base-p16_mff-800e-pre_8xb128-coslr-100e_in1k` | [MFF 800-Epochs](https://download.openmmlab.com/mmpretrain/v1.0/mff/mff_vit-base-p16_8xb512-amp-coslr-800e_in1k/mff_vit-base-p16_8xb512-amp-coslr-800e_in1k_20230801-3af7cd9d.pth) |   86.57    |   17.58   |   83.70   | [config](benchmarks/vit-base-p16_8xb128-coslr-100e_in1k.py) | [model](https://download.openmmlab.com/mmpretrain/v1.0/mff/mff_vit-base-p16_8xb512-amp-coslr-800e_in1k/vit-base-p16_8xb128-coslr-100e/vit-base-p16_8xb128-coslr-100e_20230802-6780e47d.pth) / [log](https://download.openmmlab.com/mmpretrain/v1.0/mff/mff_vit-base-p16_8xb512-amp-coslr-800e_in1k/vit-base-p16_8xb128-coslr-100e/vit-base-p16_8xb128-coslr-100e_20230802-6780e47d.json) |
| `vit-base-p16_mff-300e-pre_8xb2048-linear-coslr-90e_in1k` | [MFF 300-Epochs](https://download.openmmlab.com/mmpretrain/v1.0/mff/mff_vit-base-p16_8xb512-amp-coslr-300e_in1k/mff_vit-base-p16_8xb512-amp-coslr-300e_in1k_20230801-3c1bcce4.pth) |   304.33   |   61.60   |   64.20   | [config](benchmarks/vit-base-p16_8xb128-coslr-100e_in1k.py) | [log](https://download.openmmlab.com/mmpretrain/v1.0/mff/mff_vit-base-p16_8xb512-amp-coslr-300e_in1k/vit-base-p16_8xb2048-linear-coslr-90e_in1k/vit-base-p16_8xb2048-linear-coslr-90e_in1k.json) |
| `vit-base-p16_mff-800e-pre_8xb2048-linear-coslr-90e_in1k` | [MFF 800-Epochs](https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-base-p16_8xb512-fp16-coslr-1600e_in1k/mae_vit-base-p16_8xb512-fp16-coslr-1600e_in1k_20220825-f7569ca2.pth) |   304.33   |   61.60   |   68.30   | [config](benchmarks/vit-base-p16_8xb128-coslr-100e_in1k.py) | [model](https://download.openmmlab.com/mmpretrain/v1.0/mff/mff_vit-base-p16_8xb512-amp-coslr-800e_in1k/vit-base-p16_8xb2048-linear-coslr-90e/vit-base-p16_8xb2048-linear-coslr-90e_20230802-6b1f7bc8.pth)  / [log](https://download.openmmlab.com/mmpretrain/v1.0/mff/mff_vit-base-p16_8xb512-amp-coslr-800e_in1k/vit-base-p16_8xb2048-linear-coslr-90e/vit-base-p16_8xb2048-linear-coslr-90e_20230802-6b1f7bc8.json) |

## Citation

```bibtex
@article{MFF,
  title={Improving Pixel-based MIM by Reducing Wasted Modeling Capability},
  author={Yuan Liu, Songyang Zhang, Jiacheng Chen, Zhaohui Yu, Kai Chen, Dahua Lin},
  journal={arXiv},
  year={2023}
}
```
