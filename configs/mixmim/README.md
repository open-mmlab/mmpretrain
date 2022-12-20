# MixMIM

> [MixMIM: Mixed and Masked Image Modeling for Efficient Visual Representation Learning](https://arxiv.org/abs/2205.13137)

<!-- [ALGORITHM] -->

## Abstract

In this study, we propose Mixed and Masked Image Modeling (MixMIM), a
simple but efficient MIM method that is applicable to various hierarchical Vision
Transformers. Existing MIM methods replace a random subset of input tokens with
a special [MASK] symbol and aim at reconstructing original image tokens from
the corrupted image. However, we find that using the [MASK] symbol greatly
slows down the training and causes training-finetuning inconsistency, due to the
large masking ratio (e.g., 40% in BEiT). In contrast, we replace the masked tokens
of one image with visible tokens of another image, i.e., creating a mixed image.
We then conduct dual reconstruction to reconstruct the original two images from
the mixed input, which significantly improves efficiency. While MixMIM can
be applied to various architectures, this paper explores a simpler but stronger
hierarchical Transformer, and scales with MixMIM-B, -L, and -H. Empirical
results demonstrate that MixMIM can learn high-quality visual representations
efficiently. Notably, MixMIM-B with 88M parameters achieves 85.1% top-1
accuracy on ImageNet-1K by pretraining for 600 epochs, setting a new record for
neural networks with comparable model sizes (e.g., ViT-B) among MIM methods.
Besides, its transferring performances on the other 6 datasets show MixMIM has
better FLOPs / performance tradeoff than previous MIM methods

<div align=center>
<img src="https://user-images.githubusercontent.com/56866854/202853730-d26fb3d7-e5e8-487a-aad5-e3d4600cef87.png"/>
</div>

## How to use it?

### Inference

<!-- [TABS-BEGIN] -->

**Predict image**

```python
>>> import torch
>>> import mmcls
>>> model = mmcls.get_model('mixmim-base_3rdparty_in1k', pretrained=True)
>>> predict = mmcls.inference_model(model, 'demo/demo.JPEG')
>>> print(predict['pred_class'])
sea snake
>>> print(predict['pred_score'])
0.865431010723114
```

**Use the model**

```python
>>> import torch
>>> import mmcls
>>>
>>> model = mmcls.get_model('mixmim-base_3rdparty_in1k', pretrained=True)
>>> inputs = torch.rand(1, 3, 224, 224)
>>> # To get classification scores.
>>> out = model(inputs)
>>> print(out.shape)
torch.Size([1, 1000])
>>> # To extract features.
>>> outs = model.extract_feat(inputs)
>>> print(outs[0].shape)
torch.Size([1, 1024])
```

<!-- [TABS-END] -->

## Models

|            Model            | Params(M) | Pretrain Epochs | Flops(G) | Top-1 (%) | Top-5 (%) |                Config                 |                                        Download                                        |
| :-------------------------: | :-------: | :-------------: | :------: | :-------: | :-------: | :-----------------------------------: | :------------------------------------------------------------------------------------: |
| mixmim-base_3rdparty_in1k\* |    88     |       300       |   16.3   |   84.6    |   97.0    | [config](./mixmim-base_8xb64_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/mixmim/mixmim-base_3rdparty_in1k_20221206-e40e2c8c.pth) |

*Models with * are converted from the [official repo](https://github.com/Sense-X/MixMIM). The config files of these models are only for inference.*

For MixMIM self-supervised learning algorithm, welcome to [MMSelfSup page](https://github.com/open-mmlab/mmselfsup/tree/dev-1.x/configs/selfsup/mixmim) to get more information.

## Citation

```bibtex
@article{MixMIM2022,
  author  = {Jihao Liu, Xin Huang, Yu Liu, Hongsheng Li},
  journal = {arXiv:2205.13137},
  title   = {MixMIM: Mixed and Masked Image Modeling for Efficient Visual Representation Learning},
  year    = {2022},
}
```
