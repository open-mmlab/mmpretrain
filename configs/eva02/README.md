# EVA-02

> [EVA-02: A Visual Representation for Neon Genesis](https://arxiv.org/abs/2303.11331)

<!-- [ALGORITHM] -->

## Abstract

We launch EVA-02, a next-generation Transformer-based visual representation pre-trained to reconstruct strong and robust language-aligned vision features via masked image modeling. With an updated plain Transformer architecture as well as extensive pre-training from an open & accessible giant CLIP vision encoder, EVA-02 demonstrates superior performance compared to prior state-of-the-art approaches across various representative vision tasks, while utilizing significantly fewer parameters and compute budgets. Notably, using exclusively publicly accessible training data, EVA-02 with only 304M parameters achieves a phenomenal 90.0 fine-tuning top-1 accuracy on ImageNet-1K val set.  Additionally, our EVA-02-CLIP can reach up to 80.4 zero-shot top-1 on ImageNet-1K, outperforming the previous largest & best open-sourced CLIP with only ~1/6 parameters and ~1/6 image-text training data. We offer four EVA-02 variants in various model sizes, ranging from 6M to 304M parameters, all with impressive performance. To facilitate open accessand open research, we release the complete suite of EVA-02 to the community.

<div align=center>
<img src="https://user-images.githubusercontent.com/40905160/229037980-b83dceb5-41d6-406c-a20b-63b83c80136d.png" width="70%" alt="TrV builds upon the original plain ViT architecture and includes several enhancements: SwinGLU FFN, sub-LN, 2D RoPE, and JAX weight initialization. To keep the parameter & FLOPs consistent with the baseline, the FFN hidden dim of SwiGLU is 2/3× of the typical MLP counterpart."/>
</div>

## How to use it?

<!-- [TABS-BEGIN] -->

**Predict image**

```python
from mmpretrain import inference_model

predict = inference_model('vit-tiny-p14_eva02-in21k-pre_3rdparty_in1k-336px', 'demo/bird.JPEG')
print(predict['pred_class'])
print(predict['pred_score'])
```

**Use the model**

```python
import torch
from mmpretrain import get_model

model = get_model('vit-tiny-p14_eva02-in21k-pre_3rdparty_in1k-336px', pretrained=True)
inputs = torch.rand(1, 3, 336, 336)
out = model(inputs)
print(type(out))
# To extract features.
feats = model.extract_feat(inputs)
print(type(feats))
```

**Train/Test Command**

Prepare your dataset according to the [docs](https://mmpretrain.readthedocs.io/en/latest/user_guides/dataset_prepare.html#prepare-dataset).

Train:

```shell
python tools/train.py configs/eva02/eva02-tiny-p14_in1k.py
```

Test:

```shell
python tools/test.py configs/eva02/eva02-tiny-p14_in1k.py /path/to/eva02-tiny-p14_in1k.pth
```

<!-- [TABS-END] -->

## Models and results

### Pretrained models

| Model                             | Params (M) | Flops (G) |                Config                 |                                                   Download                                                    |
| :-------------------------------- | :--------: | :-------: | :-----------------------------------: | :-----------------------------------------------------------------------------------------------------------: |
| `vit-tiny-p14_eva02-pre_in21k`\*  |    5.50    |   1.70    | [config](eva02-tiny-p14_headless.py)  | [model](https://download.openmmlab.com/mmpretrain/v1.0/eva02/eva02-tiny-p14_pre_in21k_20230505-d703e7b1.pth)  |
| `vit-small-p14_eva02-pre_in21k`\* |   21.62    |   6.14    | [config](eva02-small-p14_headless.py) | [model](https://download.openmmlab.com/mmpretrain/v1.0/eva02/eva02-small-p14_pre_in21k_20230505-3175f463.pth) |
| `vit-base-p14_eva02-pre_in21k`\*  |   85.77    |   23.22   | [config](eva02-base-p14_headless.py)  | [model](https://download.openmmlab.com/mmpretrain/v1.0/eva02/eva02-base-p14_pre_in21k_20230505-2f2d4d3c.pth)  |
| `vit-large-p14_eva02-pre_in21k`\* |   303.29   |   81.15   | [config](eva02-large-p14_headless.py) | [model](https://download.openmmlab.com/mmpretrain/v1.0/eva02/eva02-large-p14_pre_in21k_20230505-9072de5d.pth) |
| `vit-large-p14_eva02-pre_m38m`\*  |   303.29   |   81.15   | [config](eva02-large-p14_headless.py) | [model](https://download.openmmlab.com/mmpretrain/v1.0/eva02/eva02-large-p14_pre_m38m_20230505-b8a1a261.pth)  |

- The input size / patch size of MIM pre-trained EVA-02 is `224x224` / `14x14`.

*Models with * are converted from the [official repo](https://github.com/baaivision/EVA).*

### Image Classification on ImageNet-1k

#### (*w/o* IN-21K intermediate fine-tuning)

| Model                                                 |      Pretrain      | Params (M) | Flops (G) | Top-1 (%) | Top-5 (%) |               Config                |                         Download                          |
| :---------------------------------------------------- | :----------------: | :--------: | :-------: | :-------: | :-------: | :---------------------------------: | :-------------------------------------------------------: |
| `vit-tiny-p14_eva02-in21k-pre_3rdparty_in1k-336px`\*  | EVA02 ImageNet-21k |    5.76    |   4.68    |   80.69   |   95.54   | [config](./eva02-tiny-p14_in1k.py)  | [model](https://download.openmmlab.com/mmpretrain/v1.0/eva02/eva02-tiny-p14_in21k-pre_3rdparty_in1k-336px_20230505-a4e8708a.pth) |
| `vit-small-p14_eva02-in21k-pre_3rdparty_in1k-336px`\* | EVA02 ImageNet-21k |   22.13    |   15.48   |   85.78   |   97.60   | [config](./eva02-small-p14_in1k.py) | [model](https://download.openmmlab.com/mmpretrain/v1.0/eva02/eva02-small-p14_in21k-pre_3rdparty_in1k-336px_20230505-9c5b0e85.pth) |
| `vit-base-p14_eva02-in21k-pre_3rdparty_in1k-448px`\*  | EVA02 ImageNet-21k |   87.13    |  107.11   |   88.29   |   98.53   | [config](./eva02-base-p14_in1k.py)  | [model](https://download.openmmlab.com/mmpretrain/v1.0/eva02/eva02-base-p14_in21k-pre_3rdparty_in1k-448px_20230505-8ad211c5.pth) |

*Models with * are converted from the  [official repo](https://github.com/baaivision/EVA/tree/master/EVA-02). The config files of these models are only for inference. We haven't reproduce the training results.*

#### (*w* IN-21K intermediate fine-tuning)

| Model                                                 |      Pretrain      | Params (M) | Flops (G) | Top-1 (%) | Top-5 (%) |               Config                |                         Download                          |
| :---------------------------------------------------- | :----------------: | :--------: | :-------: | :-------: | :-------: | :---------------------------------: | :-------------------------------------------------------: |
| `vit-base-p14_eva02-in21k-pre_in21k-medft_3rdparty_in1k-448px`\* | EVA02 ImageNet-21k |   87.13    |  107.11   |   88.47   |   98.62   | [config](./eva02-base-p14_in1k.py)  | [model](https://download.openmmlab.com/mmpretrain/v1.0/eva02/eva02-base-p14_in21k-pre_in21k-medft_3rdparty_in1k-448px_20230505-5cd4d87f.pth) |
| `vit-large-p14_eva02-in21k-pre_in21k-medft_3rdparty_in1k-448px`\* | EVA02 ImageNet-21k |   305.08   |  362.33   |   89.65   |   98.95   | [config](./eva02-large-p14_in1k.py) | [model](https://download.openmmlab.com/mmpretrain/v1.0/eva02/eva02-large-p14_in21k-pre_in21k-medft_3rdparty_in1k-448px_20230505-926d1599.pth) |
| `vit-large-p14_eva02_m38m-pre_in21k-medft_3rdparty_in1k-448px`\* |  EVA02 Merged-38M  |   305.10   |  362.33   |   89.83   |   99.00   | [config](./eva02-large-p14_in1k.py) | [model](https://download.openmmlab.com/mmpretrain/v1.0/eva02/eva02-large-p14_m38m-pre_in21k-medft_3rdparty_in1k-448px_20230505-150dc5ed.pth) |

*Models with * are converted from the  [official repo](https://github.com/baaivision/EVA/tree/master/EVA-02). The config files of these models are only for inference. We haven't reproduce the training results.*

## Citation

```bibtex
@article{EVA-02,
  title={EVA-02: A Visual Representation for Neon Genesis},
  author={Yuxin Fang and Quan Sun and Xinggang Wang and Tiejun Huang and Xinlong Wang and Yue Cao},
  journal={arXiv preprint arXiv:2303.11331},
  year={2023}
}
```
