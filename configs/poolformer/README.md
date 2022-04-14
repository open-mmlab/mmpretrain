# PoolFormer

> [MetaFormer is Actually What You Need for Vision](https://arxiv.org/abs/2111.11418)
<!-- [ALGORITHM] -->

## Abstract

Transformers have shown great potential in computer vision tasks. A common belief is their attention-based token mixer module contributes most to their competence. However, recent works show the attention-based module in transformers can be replaced by spatial MLPs and the resulted models still perform quite well. Based on this observation, we hypothesize that the general architecture of the transformers, instead of the specific token mixer module, is more essential to the model's performance. To verify this, we deliberately replace the attention module in transformers with an embarrassingly simple spatial pooling operator to conduct only basic token mixing. Surprisingly, we observe that the derived model, termed as PoolFormer, achieves competitive performance on multiple computer vision tasks. For example, on ImageNet-1K, PoolFormer achieves 82.1% top-1 accuracy, surpassing well-tuned vision transformer/MLP-like baselines DeiT-B/ResMLP-B24 by 0.3%/1.1% accuracy with 35%/52% fewer parameters and 49%/61% fewer MACs. The effectiveness of PoolFormer verifies our hypothesis and urges us to initiate the concept of "MetaFormer", a general architecture abstracted from transformers without specifying the token mixer. Based on the extensive experiments, we argue that MetaFormer is the key player in achieving superior results for recent transformer and MLP-like models on vision tasks. This work calls for more future research dedicated to improving MetaFormer instead of focusing on the token mixer modules. Additionally, our proposed PoolFormer could serve as a starting baseline for future MetaFormer architecture design.

<div align=center>
<img src="https://user-images.githubusercontent.com/15921929/144710761-1635f59a-abde-4946-984c-a2c3f22a19d2.png" width="100%"/>
</div>

## Results and models

### ImageNet-1k

|         Model         | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) | Config | Download |
|:---------------------:|:---------:|:--------:|:---------:|:---------:|:------:|:--------:|
| PoolFormer-S12\*      | 11.92     | 1.87     | 77.24     | 93.51     | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/poolformer/poolformer-s12_32xb128_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/poolformer/poolformer-s12_3rdparty_32xb128_in1k_20220414-f8d83051.pth) |
| PoolFormer-S24\*      | 21.39     | 3.51     | 80.33     | 95.05     | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/poolformer/poolformer-s24_32xb128_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/poolformer/poolformer-s24_3rdparty_32xb128_in1k_20220414-d7055904.pth) |
| PoolFormer-S36\*      | 30.86     | 5.15     | 81.43     | 95.45     | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/poolformer/poolformer-s36_32xb128_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/poolformer/poolformer-s36_3rdparty_32xb128_in1k_20220414-d78ff3e8.pth) |
| PoolFormer-M36\*      | 56.17     | 8.96     | 82.14     | 95.71     | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/poolformer/poolformer-m36_32xb128_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/poolformer/poolformer-m36_3rdparty_32xb128_in1k_20220414-c55e0949.pth) |
| PoolFormer-M48\*      | 73.47     | 11.80    | 82.51     | 95.95     | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/poolformer/poolformer-m48_32xb128_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/poolformer/poolformer-m48_3rdparty_32xb128_in1k_20220414-9378f3eb.pth) |

*Models with \* are converted from the [official repo](https://github.com/sail-sg/poolformer). The config files of these models are only for inference. We don't ensure these config files' training accuracy and welcome you to contribute your reproduction results.*

## Citation

```bibtex
@article{yu2021metaformer,
  title={MetaFormer is Actually What You Need for Vision},
  author={Yu, Weihao and Luo, Mi and Zhou, Pan and Si, Chenyang and Zhou, Yichen and Wang, Xinchao and Feng, Jiashi and Yan, Shuicheng},
  journal={arXiv preprint arXiv:2111.11418},
  year={2021}
}
```
