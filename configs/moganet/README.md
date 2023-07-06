# Efficient Multi-order Gated Aggregation Network

> [Efficient Multi-order Gated Aggregation Network](https://arxiv.org/abs/2211.03295)

<!-- [ALGORITHM] -->

## Abstract

Since the recent success of Vision Transformers (ViTs), explorations toward ViT-style architectures have triggered the resurgence of ConvNets. In this work, we explore the representation ability of modern ConvNets from a novel view of multi-order game-theoretic interaction, which reflects inter-variable interaction effects w.r.t.~contexts of different scales based on game theory. Within the modern ConvNet framework, we tailor the two feature mixers with conceptually simple yet effective depthwise convolutions to facilitate middle-order information across spatial and channel spaces respectively. In this light, a new family of pure ConvNet architecture, dubbed MogaNet, is proposed, which shows excellent scalability and attains competitive results among state-of-the-art models with more efficient use of parameters on ImageNet and multifarious typical vision benchmarks, including COCO object detection, ADE20K semantic segmentation, 2D\&3D human pose estimation, and video prediction. Typically, MogaNet hits 80.0\% and 87.8\% top-1 accuracy with 5.2M and 181M parameters on ImageNet, outperforming ParC-Net-S and ConvNeXt-L while saving 59\% FLOPs and 17M parameters. The source code is available at https://github.com/Westlake-AI/MogaNet.

<div align=center>
<img src="https://user-images.githubusercontent.com/44519745/200625735-86bd2237-5bbe-43c1-ab37-049810b8d8a1.jpg" width="100%"/>
</div>

## How to use it?

<!-- [TABS-BEGIN] -->

**Predict image**

```python
from mmpretrain import inference_model

predict = inference_model('moganet-tiny_3rdparty_8xb128_in1k', 'demo/bird.JPEG')
print(predict['pred_class'])
print(predict['pred_score'])
```

**Use the model**

```python
import torch
from mmpretrain import get_model

model = get_model('moganet-tiny_3rdparty_8xb128_in1k', pretrained=True)
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
python tools/test.py configs/moganet/moganet-tiny_8xb128_in1k.py https://github.com/Westlake-AI/openmixup/releases/download/moganet-in1k-weights/moga_tiny_sz224_8xb128_fp16_ep300.pth
```

<!-- [TABS-END] -->

## Models and results

### Image Classification on ImageNet-1k

| Model                                   |   Pretrain   | Params (M) | Flops (G) | Top-1 (%) | Top-5 (%) |  Config  |  Download  |
| :-------------------------------------- | :----------: | :--------: | :-------: | :-------: | :-------: | :-------: | :-------: |
| `moganet-xtiny_3rdparty_8xb128_in1k`\*  | From scratch |    2.97    |   0.79    |   76.48   |   93.49   | [config](moganet-xtiny_8xb128_in1k.py) | [model](https://github.com/Lupin1998/mmpretrain/releases/download/moganet-in1k-weights/moganet-xtiny_3rdparty_8xb128_in1k.pth) |
| `moganet-tiny_3rdparty_8xb128_in1k`\*   | From scratch |    5.20    |   1.09    |   77.24   |   93.51   | [config](moganet-tiny_8xb128_in1k.py) | [model](https://github.com/Lupin1998/mmpretrain/releases/download/moganet-in1k-weights/moganet-tiny_3rdparty_8xb128_in1k.pth) |
| `moganet-small_3rdparty_8xb128_in1k`\*  | From scratch |    4.94    |   25.35   |   83.38   |   96.58   | [config](moganet-small_8xb128_in1k.py) | [model](https://github.com/Lupin1998/mmpretrain/releases/download/moganet-in1k-weights/moganet-small_3rdparty_8xb128_in1k.pth) |
| `moganet-base_3rdparty_8xb128_in1k`\*   | From scratch |    9.88    |   43.72   |   84.20   |   96.77   | [config](moganet-base_8xb128_in1k.py) | [model](https://github.com/Lupin1998/mmpretrain/releases/download/moganet-in1k-weights/moganet-base_3rdparty_8xb128_in1k.pth) |
| `moganet-large_3rdparty_8xb128_in1k`\*  | From scratch |    15.84   |   82.48   |   84.76   |   97.15   | [config](moganet-large_8xb128_in1k.py) | [model](https://github.com/Lupin1998/mmpretrain/releases/download/moganet-in1k-weights/moganet-large_3rdparty_8xb128_in1k.pth) |
| `moganet-xlarge_3rdparty_16xb32_in1k`\* | From scratch |    34.43   |   180.8   |   85.11   |   97.38   | [config](moganet-xlarge_16xb32_in1k.py) | [model](https://github.com/Lupin1998/mmpretrain/releases/download/moganet-in1k-weights/moganet-xlarge_3rdparty_16xb32_in1k.pth) |

*Models with * are converted from the [official repo](https://github.com/Westlake-AI/MogaNet). The config files of these models are only for inference. We haven't reproduce the training results.*

## Citation

```bibtex
@article{Li2022MogaNet,
  title={Efficient Multi-order Gated Aggregation Network},
  author={Siyuan Li and Zedong Wang and Zicheng Liu and Cheng Tan and Haitao Lin and Di Wu and Zhiyuan Chen and Jiangbin Zheng and Stan Z. Li},
  journal={ArXiv},
  year={2022},
  volume={abs/2211.03295}
}
```
