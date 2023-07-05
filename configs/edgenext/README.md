# EdgeNeXt

> [EdgeNeXt: Efficiently Amalgamated CNN-Transformer Architecture for Mobile Vision Applications](https://arxiv.org/abs/2206.10589)

<!-- [ALGORITHM] -->

## Abstract

In the pursuit of achieving ever-increasing accuracy, large and complex neural networks are usually developed. Such models demand high computational resources and therefore cannot be deployed on edge devices. It is of great interest to build resource-efficient general purpose networks due to their usefulness in several application areas. In this work, we strive to effectively combine the strengths of both CNN and Transformer models and propose a new efficient hybrid architecture EdgeNeXt. Specifically in EdgeNeXt, we introduce split depth-wise transpose attention (SDTA) encoder that splits input tensors into multiple channel groups and utilizes depth-wise convolution along with self-attention across channel dimensions to implicitly increase the receptive field and encode multi-scale features. Our extensive experiments on classification, detection and segmentation tasks, reveal the merits of the proposed approach, outperforming state-of-the-art methods with comparatively lower compute requirements. Our EdgeNeXt model with 1.3M parameters achieves 71.2% top-1 accuracy on ImageNet-1K, outperforming MobileViT with an absolute gain of 2.2% with 28% reduction in FLOPs. Further, our EdgeNeXt model with 5.6M parameters achieves 79.4% top-1 accuracy on ImageNet-1K.

<div align=center>
<img src="https://github.com/mmaaz60/EdgeNeXt/raw/main/images/EdgeNext.png" width="100%"/>
</div>

## How to use it?

<!-- [TABS-BEGIN] -->

**Predict image**

```python
from mmpretrain import inference_model

predict = inference_model('edgenext-xxsmall_3rdparty_in1k', 'demo/bird.JPEG')
print(predict['pred_class'])
print(predict['pred_score'])
```

**Use the model**

```python
import torch
from mmpretrain import get_model

model = get_model('edgenext-xxsmall_3rdparty_in1k', pretrained=True)
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
python tools/test.py configs/edgenext/edgenext-xxsmall_8xb256_in1k.py https://download.openmmlab.com/mmclassification/v0/edgenext/edgenext-xxsmall_3rdparty_in1k_20220801-7ca8a81d.pth
```

<!-- [TABS-END] -->

## Models and results

### Image Classification on ImageNet-1k

| Model                                |   Pretrain   | Params (M) | Flops (G) | Top-1 (%) | Top-5 (%) |                   Config                    |                                 Download                                 |
| :----------------------------------- | :----------: | :--------: | :-------: | :-------: | :-------: | :-----------------------------------------: | :----------------------------------------------------------------------: |
| `edgenext-xxsmall_3rdparty_in1k`\*   | From scratch |    1.33    |   0.26    |   71.20   |   89.91   |  [config](edgenext-xxsmall_8xb256_in1k.py)  | [model](https://download.openmmlab.com/mmclassification/v0/edgenext/edgenext-xxsmall_3rdparty_in1k_20220801-7ca8a81d.pth) |
| `edgenext-xsmall_3rdparty_in1k`\*    | From scratch |    2.34    |   0.53    |   74.86   |   92.31   |  [config](edgenext-xsmall_8xb256_in1k.py)   | [model](https://download.openmmlab.com/mmclassification/v0/edgenext/edgenext-xsmall_3rdparty_in1k_20220801-974f9fe7.pth) |
| `edgenext-small_3rdparty_in1k`\*     | From scratch |    5.59    |   1.25    |   79.41   |   94.53   |   [config](edgenext-small_8xb256_in1k.py)   | [model](https://download.openmmlab.com/mmclassification/v0/edgenext/edgenext-small_3rdparty_in1k_20220801-d00db5f8.pth) |
| `edgenext-small-usi_3rdparty_in1k`\* | From scratch |    5.59    |   1.25    |   81.06   |   95.34   | [config](edgenext-small_8xb256-usi_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/edgenext/edgenext-small_3rdparty-usi_in1k_20220801-ae6d8dd3.pth) |
| `edgenext-base_3rdparty_in1k`\*      | From scratch |   18.51    |   3.81    |   82.48   |   96.20   |   [config](edgenext-base_8xb256_in1k.py)    | [model](https://download.openmmlab.com/mmclassification/v0/edgenext/edgenext-base_3rdparty_in1k_20220801-9ade408b.pth) |
| `edgenext-base_3rdparty-usi_in1k`\*  | From scratch |   18.51    |   3.81    |   83.67   |   96.70   | [config](edgenext-base_8xb256-usi_in1k.py)  | [model](https://download.openmmlab.com/mmclassification/v0/edgenext/edgenext-base_3rdparty-usi_in1k_20220801-909e8939.pth) |

*Models with * are converted from the [official repo](https://github.com/mmaaz60/EdgeNeXt). The config files of these models are only for inference. We haven't reproduce the training results.*

## Citation

```bibtex
@article{Maaz2022EdgeNeXt,
    title={EdgeNeXt: Efficiently Amalgamated CNN-Transformer Architecture for Mobile Vision Applications},
    author={Muhammad Maaz and Abdelrahman Shaker and Hisham Cholakkal and Salman Khan and Syed Waqas Zamir and Rao Muhammad Anwer and Fahad Shahbaz Khan},
    journal={2206.10589},
    year={2022}
}
```
