# SimMIM

> [SimMIM: A Simple Framework for Masked Image Modeling](https://arxiv.org/abs/2111.09886)

<!-- [ALGORITHM] -->

## Abstract

This paper presents SimMIM, a simple framework for masked image modeling. We simplify recently proposed related approaches without special designs such as blockwise masking and tokenization via discrete VAE or clustering. To study what let the masked image modeling task learn good representations, we systematically study the major components in our framework, and find that simple designs of each component have revealed very strong representation learning performance: 1) random masking of the input image with a moderately large masked patch size (e.g., 32) makes a strong pre-text task; 2) predicting raw pixels of RGB values by direct regression performs no worse than the patch classification approaches with complex designs; 3) the prediction head can be as light as a linear layer, with no worse performance than heavier ones. Using ViT-B, our approach achieves 83.8% top-1 fine-tuning accuracy on ImageNet-1K by pre-training also on this dataset, surpassing previous best approach by +0.6%. When applied on a larger model of about 650 million parameters, SwinV2H, it achieves 87.1% top-1 accuracy on ImageNet-1K using only ImageNet-1K data. We also leverage this approach to facilitate the training of a 3B model (SwinV2-G), that by 40× less data than that in previous practice, we achieve the state-of-the-art on four representative vision benchmarks. The code and models will be publicly available at https: //github.com/microsoft/SimMIM .

<div align=center>
<img src="https://user-images.githubusercontent.com/30762564/159404597-ac6d3a44-ee59-4cdc-8f6f-506a7d1b18b6.png" width="70%"/>
</div>

## How to use it?

<!-- [TABS-BEGIN] -->

**Predict image**

```python
from mmpretrain import inference_model

predict = inference_model('swin-base-w6_simmim-100e-pre_8xb256-coslr-100e_in1k-192px', 'demo/bird.JPEG')
print(predict['pred_class'])
print(predict['pred_score'])
```

**Use the model**

```python
import torch
from mmpretrain import get_model

model = get_model('simmim_swin-base-w6_8xb256-amp-coslr-100e_in1k-192px', pretrained=True)
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
python tools/train.py configs/simmim/simmim_swin-base-w6_8xb256-amp-coslr-100e_in1k-192px.py
```

Test:

```shell
python tools/test.py configs/simmim/benchmarks/swin-base-w6_8xb256-coslr-100e_in1k-192px.py https://download.openmmlab.com/mmselfsup/1.x/simmim/simmim_swin-base_8xb256-amp-coslr-100e_in1k-192/swin-base_ft-8xb256-coslr-100e_in1k/swin-base_ft-8xb256-coslr-100e_in1k_20220829-9cf23aa1.pth
```

<!-- [TABS-END] -->

## Models and results

### Pretrained models

| Model                                                     | Params (M) | Flops (G) |                            Config                             |                            Download                             |
| :-------------------------------------------------------- | :--------: | :-------: | :-----------------------------------------------------------: | :-------------------------------------------------------------: |
| `simmim_swin-base-w6_8xb256-amp-coslr-100e_in1k-192px`    |   89.87    |   18.83   | [config](simmim_swin-base-w6_8xb256-amp-coslr-100e_in1k-192px.py) | [model](https://download.openmmlab.com/mmselfsup/1.x/simmim/simmim_swin-base_8xb256-amp-coslr-100e_in1k-192/simmim_swin-base_8xb256-amp-coslr-100e_in1k-192_20220829-0e15782d.pth) \| [log](https://download.openmmlab.com/mmselfsup/1.x/simmim/simmim_swin-base_8xb256-amp-coslr-100e_in1k-192/simmim_swin-base_8xb256-amp-coslr-100e_in1k-192_20220829-0e15782d.json) |
| `simmim_swin-base-w6_16xb128-amp-coslr-800e_in1k-192px`   |   89.87    |   18.83   | [config](simmim_swin-base-w6_16xb128-amp-coslr-800e_in1k-192px.py) | [model](https://download.openmmlab.com/mmselfsup/1.x/simmim/simmim_swin-base_16xb128-amp-coslr-800e_in1k-192/simmim_swin-base_16xb128-amp-coslr-800e_in1k-192_20220916-a0e931ac.pth) \| [log](https://download.openmmlab.com/mmselfsup/1.x/simmim/simmim_swin-base_16xb128-amp-coslr-800e_in1k-192/simmim_swin-base_16xb128-amp-coslr-800e_in1k-192_20220916-a0e931ac.json) |
| `simmim_swin-large-w12_16xb128-amp-coslr-800e_in1k-192px` |   199.92   |   55.85   | [config](simmim_swin-large-w12_16xb128-amp-coslr-800e_in1k-192px.py) | [model](https://download.openmmlab.com/mmselfsup/1.x/simmim/simmim_swin-large_16xb128-amp-coslr-800e_in1k-192/simmim_swin-large_16xb128-amp-coslr-800e_in1k-192_20220916-4ad216d3.pth) \| [log](https://download.openmmlab.com/mmselfsup/1.x/simmim/simmim_swin-large_16xb128-amp-coslr-800e_in1k-192/simmim_swin-large_16xb128-amp-coslr-800e_in1k-192_20220916-4ad216d3.json) |

### Image Classification on ImageNet-1k

| Model                                     |                   Pretrain                   | Params (M) | Flops (G) | Top-1 (%) |                   Config                   |                   Download                    |
| :---------------------------------------- | :------------------------------------------: | :--------: | :-------: | :-------: | :----------------------------------------: | :-------------------------------------------: |
| `swin-base-w6_simmim-100e-pre_8xb256-coslr-100e_in1k-192px` | [SIMMIM 100-Epochs](https://download.openmmlab.com/mmselfsup/1.x/simmim/simmim_swin-base_8xb256-amp-coslr-100e_in1k-192/simmim_swin-base_8xb256-amp-coslr-100e_in1k-192_20220829-0e15782d.pth) |   87.75    |   11.30   |   82.70   | [config](benchmarks/swin-base-w6_8xb256-coslr-100e_in1k-192px.py) | [model](https://download.openmmlab.com/mmselfsup/1.x/simmim/simmim_swin-base_8xb256-amp-coslr-100e_in1k-192/swin-base_ft-8xb256-coslr-100e_in1k/swin-base_ft-8xb256-coslr-100e_in1k_20220829-9cf23aa1.pth) \| [log](https://download.openmmlab.com/mmselfsup/1.x/simmim/simmim_swin-base_8xb256-amp-coslr-100e_in1k-192/swin-base_ft-8xb256-coslr-100e_in1k/swin-base_ft-8xb256-coslr-100e_in1k_20220829-9cf23aa1.json) |
| `swin-base-w7_simmim-100e-pre_8xb256-coslr-100e_in1k` | [SIMMIM 100-Epochs](https://download.openmmlab.com/mmselfsup/1.x/simmim/simmim_swin-base_8xb256-amp-coslr-100e_in1k-192/simmim_swin-base_8xb256-amp-coslr-100e_in1k-192_20220829-0e15782d.pth) |   87.77    |   15.47   |   83.50   | [config](benchmarks/swin-base-w7_8xb256-coslr-100e_in1k.py) |                      N/A                      |
| `swin-base-w6_simmim-800e-pre_8xb256-coslr-100e_in1k-192px` | [SIMMIM 800-Epochs](https://download.openmmlab.com/mmselfsup/1.x/simmim/simmim_swin-base_16xb128-amp-coslr-800e_in1k-192/simmim_swin-base_16xb128-amp-coslr-800e_in1k-192_20220916-a0e931ac.pth) |   87.77    |   15.47   |   83.80   | [config](benchmarks/swin-base-w7_8xb256-coslr-100e_in1k.py) | [model](https://download.openmmlab.com/mmselfsup/1.x/simmim/simmim_swin-base_16xb128-amp-coslr-800e_in1k-192/swin-base_ft-8xb256-coslr-100e_in1k-224/swin-base_ft-8xb256-coslr-100e_in1k-224_20221208-155cc6e6.pth) \| [log](https://download.openmmlab.com/mmselfsup/1.x/simmim/simmim_swin-base_16xb128-amp-coslr-800e_in1k-192/swin-base_ft-8xb256-coslr-100e_in1k-224/swin-base_ft-8xb256-coslr-100e_in1k-224_20221208-155cc6e6.json) |
| `swin-large-w14_simmim-800e-pre_8xb256-coslr-100e_in1k` | [SIMMIM 800-Epochs](https://download.openmmlab.com/mmselfsup/1.x/simmim/simmim_swin-large_16xb128-amp-coslr-800e_in1k-192/simmim_swin-large_16xb128-amp-coslr-800e_in1k-192_20220916-4ad216d3.pth) |   196.85   |   38.85   |   84.80   | [config](benchmarks/swin-large-w14_8xb256-coslr-100e_in1k.py) | [model](https://download.openmmlab.com/mmselfsup/1.x/simmim/simmim_swin-large_16xb128-amp-coslr-800e_in1k-192/swin-large_ft-8xb256-coslr-ws14-100e_in1k-224/swin-large_ft-8xb256-coslr-ws14-100e_in1k-224_20220916-d4865790.pth) \| [log](https://download.openmmlab.com/mmselfsup/1.x/simmim/simmim_swin-large_16xb128-amp-coslr-800e_in1k-192/swin-large_ft-8xb256-coslr-ws14-100e_in1k-224/swin-large_ft-8xb256-coslr-ws14-100e_in1k-224_20220916-d4865790.json) |

## Citation

```bibtex
@inproceedings{xie2021simmim,
  title={SimMIM: A Simple Framework for Masked Image Modeling},
  author={Xie, Zhenda and Zhang, Zheng and Cao, Yue and Lin, Yutong and Bao, Jianmin and Yao, Zhuliang and Dai, Qi and Hu, Han},
  booktitle={International Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022}
}
```
