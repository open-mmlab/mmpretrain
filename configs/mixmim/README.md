# MixMIM

> [MixMIM: Mixed and Masked Image Modeling for Efficient Visual Representation Learning](https://arxiv.org/abs/2205.13137)

## Abstract

In this study, we propose Mixed and Masked Image Modeling (MixMIM), a
simple but efficient MIM method that is applicable to various hierarchical Vision
Transformers. Existing MIM methods replace a random subset of input tokens with
a special \[MASK\] symbol and aim at reconstructing original image tokens from
the corrupted image. However, we find that using the \[MASK\] symbol greatly
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

## Models

|    Model    | Params(M) | Pretrain Epochs | Flops(G) | Top-1 (%) | Top-5 (%) |                      Config                      |         Download          |
| :---------: | :-------: | :-------------: | :------: | :-------: | :-------: | :----------------------------------------------: | :-----------------------: |
| MixMIM-Base |    88     |      1.08       |   300    |   84.6    |   97.0    | [config](./mixmim-base-p16_8xb64-pt300e-in1k.py) | [model](<>)  \| [log](<>) |

## How to use it?

### Inference

```python
python ./tools/test.py configs/mixmim/mixmim-base-p16_8xb64-pt300e-in1k.py  mixmim-base-p16_8xb64-pt300e-in1k_checkpoint.pth

```

## Citation

```
@article{MixMIM2022,
  author  = {Jihao Liu, Xin Huang, Yu Liu, Hongsheng Li},
  journal = {arXiv:2205.13137},
  title   = {MixMIM: Mixed and Masked Image Modeling for Efficient Visual Representation Learning},
  year    = {2022},
}
```
