# Vision Transformer

> [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)

<!-- [ALGORITHM] -->

## Introduction

**Vision Transformer**, known as **ViT**, succeeded in using a full transformer to outperform previous works that based on convolutional networks in vision field. ViT splits image into patches to feed the multi-head attentions, concatenates a learnable class token for final prediction and adds a learnable position embeddings for relative positional message between patches. Based on these three techniques with attentions, ViT provides a brand-new pattern to build a basic structure in vision field.

The strategy works even better when coupled with large datasets pre-trainings. Because of its simplicity and effectiveness, some after works in classification field are originated from ViT. And even in recent multi-modality field, ViT-based method still plays a role in it.

<div align=center>
<img src="https://user-images.githubusercontent.com/26739999/142579081-b5718032-6581-472b-8037-ea66aaa9e278.png" width="70%"/>
</div>

## Abstract

<details>

<summary>Show the paper's abstract</summary>

<br>

While the Transformer architecture has become the de-facto standard for natural language processing tasks, its applications to computer vision remain limited. In vision, attention is either applied in conjunction with convolutional networks, or used to replace certain components of convolutional networks while keeping their overall structure in place. We show that this reliance on CNNs is not necessary and a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks. When pre-trained on large amounts of data and transferred to multiple mid-sized or small image recognition benchmarks (ImageNet, CIFAR-100, VTAB, etc.), Vision Transformer (ViT) attains excellent results compared to state-of-the-art convolutional networks while requiring substantially fewer computational resources to train.
</br>

</details>

## How to use it?

<!-- [TABS-BEGIN] -->

**Predict image**

```python
from mmpretrain import inference_model

predict = inference_model('vit-base-p32_in21k-pre_3rdparty_in1k-384px', 'demo/bird.JPEG')
print(predict['pred_class'])
print(predict['pred_score'])
```

**Use the model**

```python
import torch
from mmpretrain import get_model

model = get_model('vit-base-p32_in21k-pre_3rdparty_in1k-384px', pretrained=True)
inputs = torch.rand(1, 3, 224, 224)
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
python tools/train.py configs/vision_transformer/vit-base-p16_32xb128-mae_in1k.py
```

Test:

```shell
python tools/test.py configs/vision_transformer/vit-base-p32_64xb64_in1k-384px.py https://download.openmmlab.com/mmclassification/v0/vit/finetune/vit-base-p32_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-9cea8599.pth
```

<!-- [TABS-END] -->

## Models and results

### Image Classification on ImageNet-1k

| Model                                           |   Pretrain   | Params (M) | Flops (G) | Top-1 (%) | Top-5 (%) |                    Config                    |                           Download                           |
| :---------------------------------------------- | :----------: | :--------: | :-------: | :-------: | :-------: | :------------------------------------------: | :----------------------------------------------------------: |
| `vit-base-p32_in21k-pre_3rdparty_in1k-384px`\*  | ImageNet-21k |   88.30    |   13.06   |   84.01   |   97.08   | [config](vit-base-p32_64xb64_in1k-384px.py)  | [model](https://download.openmmlab.com/mmclassification/v0/vit/finetune/vit-base-p32_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-9cea8599.pth) |
| `vit-base-p16_32xb128-mae_in1k`                 | From scratch |   86.57    |   17.58   |   82.37   |   96.15   |  [config](vit-base-p16_32xb128-mae_in1k.py)  | [model](https://download.openmmlab.com/mmclassification/v0/vit/vit-base-p16_pt-32xb128-mae_in1k_20220623-4c544545.pth) \| [log](https://download.openmmlab.com/mmclassification/v0/vit/vit-base-p16_pt-32xb128-mae_in1k_20220623-4c544545.log) |
| `vit-base-p16_in21k-pre_3rdparty_in1k-384px`\*  | ImageNet-21k |   86.86    |   55.54   |   85.43   |   97.77   | [config](vit-base-p16_64xb64_in1k-384px.py)  | [model](https://download.openmmlab.com/mmclassification/v0/vit/finetune/vit-base-p16_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-98e8652b.pth) |
| `vit-large-p16_in21k-pre_3rdparty_in1k-384px`\* | ImageNet-21k |   304.72   |  191.21   |   85.63   |   97.63   | [config](vit-large-p16_64xb64_in1k-384px.py) | [model](https://download.openmmlab.com/mmclassification/v0/vit/finetune/vit-large-p16_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-b20ba619.pth) |

*Models with * are converted from the [official repo](https://github.com/google-research/vision_transformer/blob/88a52f8892c80c10de99194990a517b4d80485fd/vit_jax/models.py#L208). The config files of these models are only for inference. We haven't reproduce the training results.*

## Citation

```bibtex
@inproceedings{
  dosovitskiy2021an,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Alexey Dosovitskiy and Lucas Beyer and Alexander Kolesnikov and Dirk Weissenborn and Xiaohua Zhai and Thomas Unterthiner and Mostafa Dehghani and Matthias Minderer and Georg Heigold and Sylvain Gelly and Jakob Uszkoreit and Neil Houlsby},
  booktitle={International Conference on Learning Representations},
  year={2021},
  url={https://openreview.net/forum?id=YicbFdNTTy}
}
```
