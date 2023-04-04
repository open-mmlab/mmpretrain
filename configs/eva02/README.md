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

predict = inference_model('eva02-tiny-p14_in21k-pre_3rdparty_in1k', 'demo/bird.JPEG')
print(predict['pred_class'])
print(predict['pred_score'])
```

**Use the model**

```python
import torch
from mmpretrain import get_model

model = get_model('eva02-tiny-p14_in21k-pre_3rdparty_in1k', pretrained=True)
inputs = torch.rand(1, 3, 336, 336)
out = model(inputs)
print(type(out))
# To extract features.
feats = model.extract_feat(inputs)
print(type(feats))
```

**Train/Test Command**

Prepare your dataset according to the [docs](https://mmclassification.readthedocs.io/en/1.x/user_guides/dataset_prepare.html#prepare-dataset).

Train:

```shell
python tools/train.py configs/eva02/eva02_tiny_p14_8xb16_in1k.py
```

Test:

```shell
python tools/test.py configs/eva02/eva02_tiny_p14_8xb16_in1k.py /path/to/eva02_tiny_p14_8xb16_in1k.pt
```

<!-- [TABS-END] -->

## Models and results

### Pretrained models

| Model                                        | Params (M) | Flops (G) |                 Config                  |                       Download                       |
| :------------------------------------------- | :--------: | :-------: | :-------------------------------------: | :--------------------------------------------------: |
| `eva02-tiny-p14_in21k-pre_3rdparty_in21k`\*  |    5.50    |   1.70    | [config](./eva02_tiny_p14_headless.py)  | [model](eva02-tiny-p14_in21k-pre_3rdparty_in21k.pt)  |
| `eva02-small-p14_in21k-pre_3rdparty_in21k`\* |   21.62    |   6.14    | [config](./eva02_small_p14_headless.py) | [model](eva02-small-p14_in21k-pre_3rdparty_in21k.pt) |
| `eva02-base-p14_in21k-pre_3rdparty_in21k`\*  |   85.37    |   23.22   | [config](./eva02_base_p14_headless.py)  | [model](eva02-base-p14_in21k-pre_3rdparty_in21k.pt)  |
| ` eva02-base-p16_in21k-pre_3rdparty_in21k`\* |   85.86    |   17.61   | [config](./eva02_base_p16_headless.py)  | [model](eva02-base-p16_in21k-pre_3rdparty_in21k.pt)  |
| `eva02-large-p14_in21k-pre_3rdparty_in21k`\* |   303.29   |   81.15   | [config](./eva02_large_p14_headless.py) | [model](eva02-large-p14_in21k-pre_3rdparty_in21k.pt) |
| `eva02-large-p16_in21k-pre_3rdparty_in21k`\* |   303.41   |   61.66   | [config](./eva02_large_p16_headless.py) | [model](eva02-large-p16_in21k-pre_3rdparty_in21k.pt) |
| `eva02-large-p14_m38m-pre_3rdparty_m38m`\*   |   303.29   |   81.15   | [config](./eva02_large_p14_headless.py) |  [model](eva02-large-p14_m38m-pre_3rdparty_m38m.pt)  |
| `eva02-large-p16_m38m-pre_3rdparty_m38m`\*   |   303.41   |   61.66   | [config](./eva02_large_p16_headless.py) |  [model](eva02-large-p16_m38m-pre_3rdparty_m38m.pt)  |

- The input size / patch size of MIM pre-trained EVA-02 is `224x224` / `14x14`.

*Models with * are converted from the [official repo](https://github.com/baaivision/EVA).*

### Image Classification on ImageNet-1k

#### (*w/o* IN-21K intermediate fine-tuning)

| Model                                       |      Pretrain      | Params (M) | Flops (G) | Top-1 (%) | Top-5 (%) |                  Config                   |                      Download                       |
| :------------------------------------------ | :----------------: | :--------: | :-------: | :-------: | :-------: | :---------------------------------------: | :-------------------------------------------------: |
| `eva02-tiny-p14_in21k-pre_3rdparty_in1k`\*  | EVA02 ImageNet-21k |    5.76    |   4.68    |   80.69   |   95.54   | [config](./eva02_tiny_p14_8xb16_in1k.py)  | [model](eva02-tiny-p14_in21k-pre_3rdparty_in1k.pt)  |
| `eva02-small-p14_in21k-pre_3rdparty_in1k`\* | EVA02 ImageNet-21k |   22.13    |   15.48   |   85.77   |   97.60   | [config](./eva02_small_p14_8xb16_in1k.py) | [model](eva02-small-p14_in21k-pre_3rdparty_in1k.pt) |
| `eva02-base-p14_in21k-pre_3rdparty_in1k`\*  | EVA02 ImageNet-21k |   87.13    |  107.11   |   88.29   |   98.53   | [config](./eva02_base_p14_8xb16_in1k.py)  | [model](eva02-base-p14_in21k-pre_3rdparty_in1k.pt)  |
| `eva02-large-p14_in21k-pre_3rdparty_in1k`\* | EVA02 ImageNet-21k |   305.10   |  362.33   |   89.51   |   98.86   | [config](./eva02_large_p14_8xb16_in1k.py) | [model](eva02-large-p14_in21k-pre_3rdparty_in1k.pt) |
| `eva02-large-p14_m38m-pre_3rdparty_in1k`\*  |  EVA02 Merged-38M  |   305.10   |  362.33   |   89.39   |   98.80   | [config](./eva02_large_p14_8xb16_in1k.py) | [model](eva02-large-p14_m38m-pre_3rdparty_in1k.pt)  |

*Models with * are converted from the  [official repo](https://github.com/baaivision/EVA/tree/master/EVA-02). The config files of these models are only for inference. We haven't reprodcue the training results.*

#### (*w* IN-21K intermediate fine-tuning)

| Model                                              |      Pretrain      | Params (M) | Flops (G) | Top-1 (%) | Top-5 (%) |                  Config                   |                        Download                        |
| :------------------------------------------------- | :----------------: | :--------: | :-------: | :-------: | :-------: | :---------------------------------------: | :----------------------------------------------------: |
| `eva02-base-p14_in21k-pre_in21k-medft_3rdparty_in1k`\* | EVA02 ImageNet-21k |   87.13    |  107.11   |   88.47   |   98.62   | [config](./eva02_base_p14_8xb16_in1k.py)  | [model](eva02-base-p14_in21k-pre_in21k-medft_3rdparty_in1k.pt) |
| `eva02-large-p14_in21k-pre_in21k-medft_3rdparty_in1k`\* | EVA02 ImageNet-21k |   305.08   |  362.33   |   89.66   |   98.95   | [config](./eva02_large_p14_8xb16_in1k.py) | [model](eva02-large-p14_in21k-pre_in21k-medft_3rdparty_in1k.pt) |
| `eva02-large-p14_m38m-pre_in21k-medft_3rdparty_in1k`\* |  EVA02 Merged-38M  |   305.10   |  362.33   |   89.83   |   99.00   | [config](./eva02_large_p14_8xb16_in1k.py) | [model](eva02-large-p14_m38m-pre_in21k-medft_3rdparty_in1k.pt) |

*Models with * are converted from the  [official repo](https://github.com/baaivision/EVA/tree/master/EVA-02). The config files of these models are only for inference. We haven't reprodcue the training results.*

## Citation

```bibtex
@article{EVA-02,
  title={EVA-02: A Visual Representation for Neon Genesis},
  author={Yuxin Fang and Quan Sun and Xinggang Wang and Tiejun Huang and Xinlong Wang and Yue Cao},
  journal={arXiv preprint arXiv:2303.11331},
  year={2023}
}
```
