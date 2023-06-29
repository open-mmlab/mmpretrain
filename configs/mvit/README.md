# MViT V2

> [MViTv2: Improved Multiscale Vision Transformers for Classification and Detection](http://openaccess.thecvf.com//content/CVPR2022/papers/Li_MViTv2_Improved_Multiscale_Vision_Transformers_for_Classification_and_Detection_CVPR_2022_paper.pdf)

<!-- [ALGORITHM] -->

## Abstract

In this paper, we study Multiscale Vision Transformers (MViTv2) as a unified architecture for image and video
classification, as well as object detection. We present an improved version of MViT that incorporates
decomposed relative positional embeddings and residual pooling connections. We instantiate this architecture
in five sizes and evaluate it for ImageNet classification, COCO detection and Kinetics video recognition where
it outperforms prior work. We further compare MViTv2s' pooling attention to window attention mechanisms where
it outperforms the latter in accuracy/compute. Without bells-and-whistles, MViTv2 has state-of-the-art
performance in 3 domains: 88.8% accuracy on ImageNet classification, 58.7 boxAP on COCO object detection as
well as 86.1% on Kinetics-400 video classification.

<div align=center>
<img src="https://user-images.githubusercontent.com/26739999/180376227-755243fa-158e-4068-940a-416036519665.png" width="50%"/>
</div>

## How to use it?

<!-- [TABS-BEGIN] -->

**Predict image**

```python
from mmpretrain import inference_model

predict = inference_model('mvitv2-tiny_3rdparty_in1k', 'demo/bird.JPEG')
print(predict['pred_class'])
print(predict['pred_score'])
```

**Use the model**

```python
import torch
from mmpretrain import get_model

model = get_model('mvitv2-tiny_3rdparty_in1k', pretrained=True)
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
python tools/test.py configs/mvit/mvitv2-tiny_8xb256_in1k.py https://download.openmmlab.com/mmclassification/v0/mvit/mvitv2-tiny_3rdparty_in1k_20220722-db7beeef.pth
```

<!-- [TABS-END] -->

## Models and results

### Image Classification on ImageNet-1k

| Model                          |   Pretrain   | Params (M) | Flops (G) | Top-1 (%) | Top-5 (%) |                Config                 |                                       Download                                       |
| :----------------------------- | :----------: | :--------: | :-------: | :-------: | :-------: | :-----------------------------------: | :----------------------------------------------------------------------------------: |
| `mvitv2-tiny_3rdparty_in1k`\*  | From scratch |   24.17    |   4.70    |   82.33   |   96.15   | [config](mvitv2-tiny_8xb256_in1k.py)  | [model](https://download.openmmlab.com/mmclassification/v0/mvit/mvitv2-tiny_3rdparty_in1k_20220722-db7beeef.pth) |
| `mvitv2-small_3rdparty_in1k`\* | From scratch |   34.87    |   7.00    |   83.63   |   96.51   | [config](mvitv2-small_8xb256_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/mvit/mvitv2-small_3rdparty_in1k_20220722-986bd741.pth) |
| `mvitv2-base_3rdparty_in1k`\*  | From scratch |   51.47    |   10.16   |   84.34   |   96.86   | [config](mvitv2-base_8xb256_in1k.py)  | [model](https://download.openmmlab.com/mmclassification/v0/mvit/mvitv2-base_3rdparty_in1k_20220722-9c4f0a17.pth) |
| `mvitv2-large_3rdparty_in1k`\* | From scratch |   217.99   |   43.87   |   85.25   |   97.14   | [config](mvitv2-large_8xb256_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/mvit/mvitv2-large_3rdparty_in1k_20220722-2b57b983.pth) |

*Models with * are converted from the [official repo](https://github.com/facebookresearch/mvit). The config files of these models are only for inference. We haven't reproduce the training results.*

## Citation

```bibtex
@inproceedings{li2021improved,
  title={MViTv2: Improved multiscale vision transformers for classification and detection},
  author={Li, Yanghao and Wu, Chao-Yuan and Fan, Haoqi and Mangalam, Karttikeya and Xiong, Bo and Malik, Jitendra and Feichtenhofer, Christoph},
  booktitle={CVPR},
  year={2022}
}
```
