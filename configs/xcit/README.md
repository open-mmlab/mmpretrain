# XCiT

> [XCiT: Cross-Covariance Image Transformers](https://arxiv.org/abs/2106.09681)

<!-- [ALGORITHM] -->

## Abstract

Following their success in natural language processing, transformers have recently shown much promise for computer vision. The self-attention operation underlying transformers yields global interactions between all tokens ,i.e. words or image patches, and enables flexible modelling of image data beyond the local interactions of convolutions. This flexibility, however, comes with a quadratic complexity in time and memory, hindering application to long sequences and high-resolution images. We propose a "transposed" version of self-attention that operates across feature channels rather than tokens, where the interactions are based on the cross-covariance matrix between keys and queries. The resulting cross-covariance attention (XCA) has linear complexity in the number of tokens, and allows efficient processing of high-resolution images. Our cross-covariance image transformer (XCiT) is built upon XCA. It combines the accuracy of conventional transformers with the scalability of convolutional architectures. We validate the effectiveness and generality of XCiT by reporting excellent results on multiple vision benchmarks, including image classification and self-supervised feature learning on ImageNet-1k, object detection and instance segmentation on COCO, and semantic segmentation on ADE20k.

<div align=center>
<img src="https://user-images.githubusercontent.com/26739999/218900814-64a44606-150b-4757-aec8-7015c77a9fd1.png" width="60%"/>
</div>

## How to use it?

<!-- [TABS-BEGIN] -->

**Use the model**

```python
import torch
from mmpretrain import get_model

model = get_model('xcit-nano-12-p16_3rdparty_in1k', pretrained=True)
inputs = torch.rand(1, 3, 224, 224)
out = model(inputs)
print(type(out))
# To extract features.
feats = model.extract_feat(inputs)
print(type(feats))
```

**Test Command**

Prepare your dataset according to the [docs](https://mmpretrain.readthedocs.io/en/latest/user_guides/dataset_prepare.html#prepare-dataset).

Test:

```shell
python tools/test.py configs/xcit/xcit-nano-12-p16_8xb128_in1k.py https://download.openmmlab.com/mmclassification/v0/xcit/xcit-nano-12-p16_3rdparty_in1k_20230213-ed776c38.pth
```

<!-- [TABS-END] -->

## Models and results

### Pretrained models

| Model                                           | Params (M) | Flops (G) |                      Config                       |                                       Download                                        |
| :---------------------------------------------- | :--------: | :-------: | :-----------------------------------------------: | :-----------------------------------------------------------------------------------: |
| `xcit-nano-12-p16_3rdparty_in1k`\*              |    3.05    |   0.56    |     [config](xcit-nano-12-p16_8xb128_in1k.py)     | [model](https://download.openmmlab.com/mmclassification/v0/xcit/xcit-nano-12-p16_3rdparty_in1k_20230213-ed776c38.pth) |
| `xcit-nano-12-p16_3rdparty-dist_in1k`\*         |    3.05    |   0.56    |     [config](xcit-nano-12-p16_8xb128_in1k.py)     | [model](https://download.openmmlab.com/mmclassification/v0/xcit/xcit-nano-12-p16_3rdparty-dist_in1k_20230213-fb247f7b.pth) |
| `xcit-tiny-12-p16_3rdparty_in1k`\*              |    6.72    |   1.24    |     [config](xcit-tiny-12-p16_8xb128_in1k.py)     | [model](https://download.openmmlab.com/mmclassification/v0/xcit/xcit-tiny-12-p16_3rdparty_in1k_20230213-82c547ca.pth) |
| `xcit-tiny-12-p16_3rdparty-dist_in1k`\*         |    6.72    |   1.24    |     [config](xcit-tiny-12-p16_8xb128_in1k.py)     | [model](https://download.openmmlab.com/mmclassification/v0/xcit/xcit-tiny-12-p16_3rdparty-dist_in1k_20230213-d5fde0a3.pth) |
| `xcit-nano-12-p16_3rdparty-dist_in1k-384px`\*   |    3.05    |   1.64    |  [config](xcit-nano-12-p16_8xb128_in1k-384px.py)  | [model](https://download.openmmlab.com/mmclassification/v0/xcit/xcit-nano-12-p16_3rdparty-dist_in1k-384px_20230213-712db4d4.pth) |
| `xcit-nano-12-p8_3rdparty_in1k`\*               |    3.05    |   2.16    |     [config](xcit-nano-12-p8_8xb128_in1k.py)      | [model](https://download.openmmlab.com/mmclassification/v0/xcit/xcit-nano-12-p8_3rdparty_in1k_20230213-3370c293.pth) |
| `xcit-nano-12-p8_3rdparty-dist_in1k`\*          |    3.05    |   2.16    |     [config](xcit-nano-12-p8_8xb128_in1k.py)      | [model](https://download.openmmlab.com/mmclassification/v0/xcit/xcit-nano-12-p8_3rdparty-dist_in1k_20230213-2f87d2b3.pth) |
| `xcit-tiny-24-p16_3rdparty_in1k`\*              |   12.12    |   2.34    |     [config](xcit-tiny-24-p16_8xb128_in1k.py)     | [model](https://download.openmmlab.com/mmclassification/v0/xcit/xcit-tiny-24-p16_3rdparty_in1k_20230213-366c1cd0.pth) |
| `xcit-tiny-24-p16_3rdparty-dist_in1k`\*         |   12.12    |   2.34    |     [config](xcit-tiny-24-p16_8xb128_in1k.py)     | [model](https://download.openmmlab.com/mmclassification/v0/xcit/xcit-tiny-24-p16_3rdparty-dist_in1k_20230213-b472e80a.pth) |
| `xcit-tiny-12-p16_3rdparty-dist_in1k-384px`\*   |    6.72    |   3.64    |  [config](xcit-tiny-12-p16_8xb128_in1k-384px.py)  | [model](https://download.openmmlab.com/mmclassification/v0/xcit/xcit-tiny-12-p16_3rdparty-dist_in1k-384px_20230213-00a20023.pth) |
| `xcit-tiny-12-p8_3rdparty_in1k`\*               |    6.71    |   4.81    |     [config](xcit-tiny-12-p8_8xb128_in1k.py)      | [model](https://download.openmmlab.com/mmclassification/v0/xcit/xcit-tiny-12-p8_3rdparty_in1k_20230213-8b02f8f5.pth) |
| `xcit-tiny-12-p8_3rdparty-dist_in1k`\*          |    6.71    |   4.81    |     [config](xcit-tiny-12-p8_8xb128_in1k.py)      | [model](https://download.openmmlab.com/mmclassification/v0/xcit/xcit-tiny-12-p8_3rdparty-dist_in1k_20230213-f3f9b44f.pth) |
| `xcit-small-12-p16_3rdparty_in1k`\*             |   26.25    |   4.81    |    [config](xcit-small-12-p16_8xb128_in1k.py)     | [model](https://download.openmmlab.com/mmclassification/v0/xcit/xcit-small-12-p16_3rdparty_in1k_20230213-d36779d2.pth) |
| `xcit-small-12-p16_3rdparty-dist_in1k`\*        |   26.25    |   4.81    |    [config](xcit-small-12-p16_8xb128_in1k.py)     | [model](https://download.openmmlab.com/mmclassification/v0/xcit/xcit-small-12-p16_3rdparty-dist_in1k_20230213-c95bbae1.pth) |
| `xcit-nano-12-p8_3rdparty-dist_in1k-384px`\*    |    3.05    |   6.34    |  [config](xcit-nano-12-p8_8xb128_in1k-384px.py)   | [model](https://download.openmmlab.com/mmclassification/v0/xcit/xcit-nano-12-p8_3rdparty-dist_in1k-384px_20230213-09d925ef.pth) |
| `xcit-tiny-24-p16_3rdparty-dist_in1k-384px`\*   |   12.12    |   6.87    |  [config](xcit-tiny-24-p16_8xb128_in1k-384px.py)  | [model](https://download.openmmlab.com/mmclassification/v0/xcit/xcit-tiny-24-p16_3rdparty-dist_in1k-384px_20230213-20e13917.pth) |
| `xcit-small-24-p16_3rdparty_in1k`\*             |   47.67    |   9.10    |    [config](xcit-small-24-p16_8xb128_in1k.py)     | [model](https://download.openmmlab.com/mmclassification/v0/xcit/xcit-small-24-p16_3rdparty_in1k_20230213-40febe38.pth) |
| `xcit-small-24-p16_3rdparty-dist_in1k`\*        |   47.67    |   9.10    |    [config](xcit-small-24-p16_8xb128_in1k.py)     | [model](https://download.openmmlab.com/mmclassification/v0/xcit/xcit-small-24-p16_3rdparty-dist_in1k_20230213-130d7262.pth) |
| `xcit-tiny-24-p8_3rdparty_in1k`\*               |   12.11    |   9.21    |     [config](xcit-tiny-24-p8_8xb128_in1k.py)      | [model](https://download.openmmlab.com/mmclassification/v0/xcit/xcit-tiny-24-p8_3rdparty_in1k_20230213-4b9ba392.pth) |
| `xcit-tiny-24-p8_3rdparty-dist_in1k`\*          |   12.11    |   9.21    |     [config](xcit-tiny-24-p8_8xb128_in1k.py)      | [model](https://download.openmmlab.com/mmclassification/v0/xcit/xcit-tiny-24-p8_3rdparty-dist_in1k_20230213-ad9c44b0.pth) |
| `xcit-tiny-12-p8_3rdparty-dist_in1k-384px`\*    |    6.71    |   14.13   |  [config](xcit-tiny-12-p8_8xb128_in1k-384px.py)   | [model](https://download.openmmlab.com/mmclassification/v0/xcit/xcit-tiny-12-p8_3rdparty-dist_in1k-384px_20230213-a072174a.pth) |
| `xcit-small-12-p16_3rdparty-dist_in1k-384px`\*  |   26.25    |   14.14   | [config](xcit-small-12-p16_8xb128_in1k-384px.py)  | [model](https://download.openmmlab.com/mmclassification/v0/xcit/xcit-small-12-p16_3rdparty-dist_in1k-384px_20230213-ba36c982.pth) |
| `xcit-medium-24-p16_3rdparty_in1k`\*            |   84.40    |   16.13   |    [config](xcit-medium-24-p16_8xb128_in1k.py)    | [model](https://download.openmmlab.com/mmclassification/v0/xcit/xcit-medium-24-p16_3rdparty_in1k_20230213-ad0aa92e.pth) |
| `xcit-medium-24-p16_3rdparty-dist_in1k`\*       |   84.40    |   16.13   |    [config](xcit-medium-24-p16_8xb128_in1k.py)    | [model](https://download.openmmlab.com/mmclassification/v0/xcit/xcit-medium-24-p16_3rdparty-dist_in1k_20230213-aca5cd0c.pth) |
| `xcit-small-12-p8_3rdparty_in1k`\*              |   26.21    |   18.69   |     [config](xcit-small-12-p8_8xb128_in1k.py)     | [model](https://download.openmmlab.com/mmclassification/v0/xcit/xcit-small-12-p8_3rdparty_in1k_20230213-9e364ce3.pth) |
| `xcit-small-12-p8_3rdparty-dist_in1k`\*         |   26.21    |   18.69   |     [config](xcit-small-12-p8_8xb128_in1k.py)     | [model](https://download.openmmlab.com/mmclassification/v0/xcit/xcit-small-12-p8_3rdparty-dist_in1k_20230213-71886580.pth) |
| `xcit-small-24-p16_3rdparty-dist_in1k-384px`\*  |   47.67    |   26.72   | [config](xcit-small-24-p16_8xb128_in1k-384px.py)  | [model](https://download.openmmlab.com/mmclassification/v0/xcit/xcit-small-24-p16_3rdparty-dist_in1k-384px_20230213-28fa2d0e.pth) |
| `xcit-tiny-24-p8_3rdparty-dist_in1k-384px`\*    |   12.11    |   27.05   |  [config](xcit-tiny-24-p8_8xb128_in1k-384px.py)   | [model](https://download.openmmlab.com/mmclassification/v0/xcit/xcit-tiny-24-p8_3rdparty-dist_in1k-384px_20230213-30d5e5ec.pth) |
| `xcit-small-24-p8_3rdparty_in1k`\*              |   47.63    |   35.81   |     [config](xcit-small-24-p8_8xb128_in1k.py)     | [model](https://download.openmmlab.com/mmclassification/v0/xcit/xcit-small-24-p8_3rdparty_in1k_20230213-280ebcc7.pth) |
| `xcit-small-24-p8_3rdparty-dist_in1k`\*         |   47.63    |   35.81   |     [config](xcit-small-24-p8_8xb128_in1k.py)     | [model](https://download.openmmlab.com/mmclassification/v0/xcit/xcit-small-24-p8_3rdparty-dist_in1k_20230213-f2773c78.pth) |
| `xcit-large-24-p16_3rdparty_in1k`\*             |   189.10   |   35.86   |    [config](xcit-large-24-p16_8xb128_in1k.py)     | [model](https://download.openmmlab.com/mmclassification/v0/xcit/xcit-large-24-p16_3rdparty_in1k_20230214-d29d2529.pth) |
| `xcit-large-24-p16_3rdparty-dist_in1k`\*        |   189.10   |   35.86   |    [config](xcit-large-24-p16_8xb128_in1k.py)     | [model](https://download.openmmlab.com/mmclassification/v0/xcit/xcit-large-24-p16_3rdparty-dist_in1k_20230214-4fea599c.pth) |
| `xcit-medium-24-p16_3rdparty-dist_in1k-384px`\* |   84.40    |   47.39   | [config](xcit-medium-24-p16_8xb128_in1k-384px.py) | [model](https://download.openmmlab.com/mmclassification/v0/xcit/xcit-medium-24-p16_3rdparty-dist_in1k-384px_20230214-6c23a201.pth) |
| `xcit-small-12-p8_3rdparty-dist_in1k-384px`\*   |   26.21    |   54.92   |  [config](xcit-small-12-p8_8xb128_in1k-384px.py)  | [model](https://download.openmmlab.com/mmclassification/v0/xcit/xcit-small-12-p8_3rdparty-dist_in1k-384px_20230214-9f2178bc.pth) |
| `xcit-medium-24-p8_3rdparty_in1k`\*             |   84.32    |   63.52   |    [config](xcit-medium-24-p8_8xb128_in1k.py)     | [model](https://download.openmmlab.com/mmclassification/v0/xcit/xcit-medium-24-p8_3rdparty_in1k_20230214-c362850b.pth) |
| `xcit-medium-24-p8_3rdparty-dist_in1k`\*        |   84.32    |   63.52   |    [config](xcit-medium-24-p8_8xb128_in1k.py)     | [model](https://download.openmmlab.com/mmclassification/v0/xcit/xcit-medium-24-p8_3rdparty-dist_in1k_20230214-625c953b.pth) |
| `xcit-small-24-p8_3rdparty-dist_in1k-384px`\*   |   47.63    |  105.24   |  [config](xcit-small-24-p8_8xb128_in1k-384px.py)  | [model](https://download.openmmlab.com/mmclassification/v0/xcit/xcit-small-24-p8_3rdparty-dist_in1k-384px_20230214-57298eca.pth) |
| `xcit-large-24-p16_3rdparty-dist_in1k-384px`\*  |   189.10   |  105.35   | [config](xcit-large-24-p16_8xb128_in1k-384px.py)  | [model](https://download.openmmlab.com/mmclassification/v0/xcit/xcit-large-24-p16_3rdparty-dist_in1k-384px_20230214-bd515a34.pth) |
| `xcit-large-24-p8_3rdparty_in1k`\*              |   188.93   |  141.23   |     [config](xcit-large-24-p8_8xb128_in1k.py)     | [model](https://download.openmmlab.com/mmclassification/v0/xcit/xcit-large-24-p8_3rdparty_in1k_20230214-08f2f664.pth) |
| `xcit-large-24-p8_3rdparty-dist_in1k`\*         |   188.93   |  141.23   |     [config](xcit-large-24-p8_8xb128_in1k.py)     | [model](https://download.openmmlab.com/mmclassification/v0/xcit/xcit-large-24-p8_3rdparty-dist_in1k_20230214-8c092b34.pth) |
| `xcit-medium-24-p8_3rdparty-dist_in1k-384px`\*  |   84.32    |  186.67   | [config](xcit-medium-24-p8_8xb128_in1k-384px.py)  | [model](https://download.openmmlab.com/mmclassification/v0/xcit/xcit-medium-24-p8_3rdparty-dist_in1k-384px_20230214-5db925e0.pth) |
| `xcit-large-24-p8_3rdparty-dist_in1k-384px`\*   |   188.93   |  415.00   |  [config](xcit-large-24-p8_8xb128_in1k-384px.py)  | [model](https://download.openmmlab.com/mmclassification/v0/xcit/xcit-large-24-p8_3rdparty-dist_in1k-384px_20230214-9f718b1a.pth) |

*Models with * are converted from the [official repo](https://github.com/facebookresearch/xcit). The config files of these models are only for inference. We haven't reproduce the training results.*

## Citation

```bibtex
@article{el2021xcit,
  title={XCiT: Cross-Covariance Image Transformers},
  author={El-Nouby, Alaaeldin and Touvron, Hugo and Caron, Mathilde and Bojanowski, Piotr and Douze, Matthijs and Joulin, Armand and Laptev, Ivan and Neverova, Natalia and Synnaeve, Gabriel and Verbeek, Jakob and others},
  journal={arXiv preprint arXiv:2106.09681},
  year={2021}
}
```
