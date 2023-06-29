# Visual-Attention-Network

> [Visual Attention Network](https://arxiv.org/abs/2202.09741)

<!-- [ALGORITHM] -->

## Abstract

While originally designed for natural language processing (NLP) tasks, the self-attention mechanism has recently taken various computer vision areas by storm. However, the 2D nature of images brings three challenges for applying self-attention in computer vision. (1) Treating images as 1D sequences neglects their 2D structures. (2) The quadratic complexity is too expensive for high-resolution images. (3) It only captures spatial adaptability but ignores channel adaptability. In this paper, we propose a novel large kernel attention (LKA) module to enable self-adaptive and long-range correlations in self-attention while avoiding the above issues. We further introduce a novel neural network based on LKA, namely Visual Attention Network (VAN). While extremely simple and efficient, VAN outperforms the state-of-the-art vision transformers and convolutional neural networks with a large margin in extensive experiments, including image classification, object detection, semantic segmentation, instance segmentation, etc.

<div align=center>
<img src="https://user-images.githubusercontent.com/24734142/157409411-2f622ba7-553c-4702-91be-eba03f9ea04f.png" width="80%"/>
</div>

## How to use it?

<!-- [TABS-BEGIN] -->

**Predict image**

```python
from mmpretrain import inference_model

predict = inference_model('van-tiny_3rdparty_in1k', 'demo/bird.JPEG')
print(predict['pred_class'])
print(predict['pred_score'])
```

**Use the model**

```python
import torch
from mmpretrain import get_model

model = get_model('van-tiny_3rdparty_in1k', pretrained=True)
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
python tools/test.py configs/van/van-tiny_8xb128_in1k.py https://download.openmmlab.com/mmclassification/v0/van/van-tiny_8xb128_in1k_20220501-385941af.pth
```

<!-- [TABS-END] -->

## Models and results

### Image Classification on ImageNet-1k

| Model                       |   Pretrain   | Params (M) | Flops (G) | Top-1 (%) | Top-5 (%) |               Config               |                                          Download                                          |
| :-------------------------- | :----------: | :--------: | :-------: | :-------: | :-------: | :--------------------------------: | :----------------------------------------------------------------------------------------: |
| `van-tiny_3rdparty_in1k`\*  | From scratch |    4.11    |   0.88    |   75.41   |   93.02   | [config](van-tiny_8xb128_in1k.py)  | [model](https://download.openmmlab.com/mmclassification/v0/van/van-tiny_8xb128_in1k_20220501-385941af.pth) |
| `van-small_3rdparty_in1k`\* | From scratch |   13.86    |   2.52    |   81.01   |   95.63   | [config](van-small_8xb128_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/van/van-small_8xb128_in1k_20220501-17bc91aa.pth) |
| `van-base_3rdparty_in1k`\*  | From scratch |   26.58    |   5.03    |   82.80   |   96.21   | [config](van-base_8xb128_in1k.py)  | [model](https://download.openmmlab.com/mmclassification/v0/van/van-base_8xb128_in1k_20220501-6a4cc31b.pth) |
| `van-large_3rdparty_in1k`\* | From scratch |   44.77    |   8.99    |   83.86   |   96.73   | [config](van-large_8xb128_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/van/van-large_8xb128_in1k_20220501-f212ba21.pth) |

*Models with * are converted from the [official repo](https://github.com/Visual-Attention-Network/VAN-Classification). The config files of these models are only for inference. We haven't reproduce the training results.*

## Citation

```bibtex
@article{guo2022visual,
  title={Visual Attention Network},
  author={Guo, Meng-Hao and Lu, Cheng-Ze and Liu, Zheng-Ning and Cheng, Ming-Ming and Hu, Shi-Min},
  journal={arXiv preprint arXiv:2202.09741},
  year={2022}
}
```
