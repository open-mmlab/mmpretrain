# Reversible Vision Transformers

> [Reversible Vision Transformers](https://openaccess.thecvf.com/content/CVPR2022/papers/Mangalam_Reversible_Vision_Transformers_CVPR_2022_paper.pdf)

<!-- [ALGORITHM] -->

## Introduction

**RevViT** is initially described in [Reversible Vision Tranformers](https://openaccess.thecvf.com/content/CVPR2022/papers/Mangalam_Reversible_Vision_Transformers_CVPR_2022_paper.pdf), which introduce the reversible idea into vision transformer, to reduce the GPU memory footprint required for training.

<!-- [IMAGE] -->

<div align=center>
<img src="https://github.com/facebookresearch/SlowFast/raw/main/projects/rev/teaser.png" width="70%"/>
</div>

## Abstract

<details>

<summary>Show the paper's abstract</summary>

<br>
We present Reversible Vision Transformers, a memory efficient architecture design for visual recognition. By decoupling the GPU memory footprint from the depth of the model, Reversible Vision Transformers enable memory efficient scaling of transformer architectures. We adapt two popular models, namely Vision Transformer and Multiscale Vision Transformers, to reversible variants and benchmark extensively across both model sizes and tasks of image classification, object detection and video classification. Reversible Vision Transformers achieve a reduced memory footprint of up to 15.5× at identical model complexity, parameters and accuracy, demonstrating the promise of reversible vision transformers as an efficient backbone for resource limited training regimes. Finally, we find that the additional computational burden of recomputing activations is more than overcome for deeper models, where throughput can increase up to 3.9× over their non-reversible counterparts.
</br>

</details>

## How to use it?

<!-- [TABS-BEGIN] -->

**Predict image**

```python
from mmpretrain import inference_model

predict = inference_model('revvit-small_3rdparty_in1k', 'demo/bird.JPEG')
print(predict['pred_class'])
print(predict['pred_score'])
```

**Use the model**

```python
import torch
from mmpretrain import get_model

model = get_model('revvit-small_3rdparty_in1k', pretrained=True)
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
python tools/test.py configs/revvit/revvit-small_8xb256_in1k.py https://download.openmmlab.com/mmclassification/v0/revvit/revvit-base_3rdparty_in1k_20221213-87a7b0a5.pth
```

<!-- [TABS-END] -->

## Models and results

### Image Classification on ImageNet-1k

| Model                          |   Pretrain   | Params (M) | Flops (G) | Top-1 (%) | Top-5 (%) |                Config                 |                                       Download                                       |
| :----------------------------- | :----------: | :--------: | :-------: | :-------: | :-------: | :-----------------------------------: | :----------------------------------------------------------------------------------: |
| `revvit-small_3rdparty_in1k`\* | From scratch |   22.44    |   4.58    |   79.87   |   94.90   | [config](revvit-small_8xb256_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/revvit/revvit-base_3rdparty_in1k_20221213-87a7b0a5.pth) |
| `revvit-base_3rdparty_in1k`\*  | From scratch |   87.34    |   17.49   |   81.81   |   95.56   | [config](revvit-base_8xb256_in1k.py)  | [model](https://download.openmmlab.com/mmclassification/v0/revvit/revvit-small_3rdparty_in1k_20221213-a3a34f5c.pth) |

*Models with * are converted from the [official repo](https://github.com/facebookresearch/SlowFast). The config files of these models are only for inference. We haven't reproduce the training results.*

## Citation

```bibtex
@inproceedings{mangalam2022reversible,
  title={Reversible Vision Transformers},
  author={Mangalam, Karttikeya and Fan, Haoqi and Li, Yanghao and Wu, Chao-Yuan and Xiong, Bo and Feichtenhofer, Christoph and Malik, Jitendra},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={10830--10840},
  year={2022}
}
```
