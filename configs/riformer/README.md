# RIFormer

> [RIFormer: Keep Your Vision Backbone Effective But Removing Token Mixer](https://arxiv.org/abs/2304.05659)

<!-- [ALGORITHM] -->

## Introduction

RIFormer is a way to keep a vision backbone effective while removing token mixers in its basic building blocks. Equipped with our proposed optimization strategy, we are able to build an extremely simple vision backbone with encouraging performance, while enjoying the high efficiency during inference. RIFormer shares nearly the same macro and micro design as MetaFormer, but safely removing all token mixers. The quantitative results show that our networks outperform many prevailing backbones with faster inference speed on ImageNet-1K.

<div align=center>
<img src="https://user-images.githubusercontent.com/48375204/223930120-dc075c8e-0513-42eb-9830-469a45c1d941.png" width="65%"/>
</div>

## Abstract

<details>

<summary>Show the paper's abstract</summary>

<br>
This paper studies how to keep a vision backbone effective while removing token mixers in its basic building blocks. Token mixers, as self-attention for vision transformers (ViTs), are intended to perform information communication between different spatial tokens but suffer from considerable computational cost and latency. However, directly removing them will lead to an incomplete model structure prior, and thus brings a significant accuracy drop. To this end, we first develop an RepIdentityFormer base on the re-parameterizing idea, to study the token mixer free model architecture. And we then explore the improved learning paradigm to break the limitation of simple token mixer free backbone, and summarize the empirical practice into 5 guidelines. Equipped with the proposed optimization strategy, we are able to build an extremely simple vision backbone with encouraging performance, while enjoying the high efficiency during inference. Extensive experiments and ablative analysis also demonstrate that the inductive bias of network architecture, can be incorporated into simple network structure with appropriate optimization strategy. We hope this work can serve as a starting point for the exploration of optimization-driven efficient network design.
</br>

</details>

## How to use

The checkpoints provided are all `training-time` models. Use the reparameterize tool or `switch_to_deploy` interface to switch them to more efficient `inference-time` architecture, which not only has fewer parameters but also less calculations.

<!-- [TABS-BEGIN] -->

**Predict image**

Use `classifier.backbone.switch_to_deploy()` interface to switch the RIFormer models into inference mode.

```python
>>> import torch
>>> from mmpretrain import get_model, inference_model
>>>
>>> model = get_model("riformer-s12_in1k", pretrained=True)
>>> results = inference_model(model, 'demo/demo.JPEG')
>>> print( (results['pred_class'], results['pred_score']) )
('sea snake', 0.7827484011650085)
>>>
>>> # switch to deploy mode
>>> model.backbone.switch_to_deploy()
>>> results = inference_model(model, 'demo/demo.JPEG')
>>> print( (results['pred_class'], results['pred_score']) )
('sea snake', 0.7827480435371399)
```

**Use the model**

```python
>>> import torch
>>>
>>> model = get_model("riformer-s12_in1k", pretrained=True)
>>> model.eval()
>>> inputs = torch.rand(1, 3, 224, 224).to(model.data_preprocessor.device)
>>> # To get classification scores.
>>> out = model(inputs)
>>> print(out.shape)
torch.Size([1, 1000])
>>> # To extract features.
>>> outs = model.extract_feat(inputs)
>>> print(outs[0].shape)
torch.Size([1, 512])
>>>
>>> # switch to deploy mode
>>> model.backbone.switch_to_deploy()
>>> out_deploy = model(inputs)
>>> print(out.shape)
torch.Size([1, 1000])
>>> assert torch.allclose(out, out_deploy, rtol=1e-4, atol=1e-5) # pass without error
```

**Test Command**

Place the ImageNet dataset to the `data/imagenet/` directory, or prepare datasets according to the [docs](https://mmpretrain.readthedocs.io/en/latest/user_guides/dataset_prepare.html#prepare-dataset).

*224×224*

Download Checkpoint:

```shell
wget https://download.openmmlab.com/mmclassification/v1/riformer/riformer-s12_32xb128_in1k_20230406-6741ce71.pth
```

Test use unfused model:

```shell
python tools/test.py configs/riformer/riformer-s12_8xb128_in1k.py riformer-s12_32xb128_in1k_20230406-6741ce71.pth
```

Reparameterize checkpoint:

```shell
python tools/model_converters/reparameterize_model.py configs/riformer/riformer-s12_8xb128_in1k.py riformer-s12_32xb128_in1k_20230406-6741ce71.pth riformer-s12_deploy.pth
```

Test use fused model:

```shell
python tools/test.py configs/riformer/deploy/riformer-s12-deploy_8xb128_in1k.py riformer-s12_deploy.pth
```

<!-- [TABS-END] -->

For more configurable parameters, please refer to the [API](https://mmpretrain.readthedocs.io/en/latest/api/generated/mmpretrain.models.backbones.RIFormer.html#mmpretrain.models.backbones.RIFormer).

<details>

<summary><b>How to use the reparameterization tool</b>(click to show)</summary>

<br>

Use provided tool to reparameterize the given model and save the checkpoint:

```bash
python tools/convert_models/reparameterize_model.py ${CFG_PATH} ${SRC_CKPT_PATH} ${TARGET_CKPT_PATH}
```

`${CFG_PATH}` is the config file path, `${SRC_CKPT_PATH}` is the source chenpoint file path, `${TARGET_CKPT_PATH}` is the target deploy weight file path.

For example:

```shell
# download the weight
wget https://download.openmmlab.com/mmclassification/v1/riformer/riformer-s12_32xb128_in1k_20230406-6741ce71.pth

# reparameterize unfused weight to fused weight
python tools/model_converters/reparameterize_model.py configs/riformer/riformer-s12_8xb128_in1k.py riformer-s12_32xb128_in1k_20230406-6741ce71.pth riformer-s12_deploy.pth
```

To use reparameterized weights, you can use the deploy model config file such as the [s12_deploy example](./deploy/riformer-s12-deploy_8xb128_in1k.py):

```text
# in riformer-s12-deploy_8xb128_in1k.py
_base_ = '../deploy/riformer-s12-deploy_8xb128_in1k.py'  # basic s12 config

model = dict(backbone=dict(deploy=True))  # switch model into deploy mode
```

```shell
python tools/test.py configs/riformer/deploy/riformer-s12-deploy_8xb128_in1k.py riformer-s12_deploy.pth
```

</br>

</details>

## Results and models

### ImageNet-1k

|         Model         | resolution | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) |                    Config                     |                                         Download                                          |
| :-------------------: | :--------: | :-------: | :------: | :-------: | :-------: | :-------------------------------------------: | :---------------------------------------------------------------------------------------: |
|   riformer-s12_in1k   |  224x224   |   11.92   |   1.82   |   76.90   |   93.06   |    [config](./riformer-s12_8xb128_in1k.py)    | [model](https://download.openmmlab.com/mmclassification/v1/riformer/riformer-s12_32xb128_in1k_20230406-6741ce71.pth) |
|   riformer-s24_in1k   |  224x224   |   21.39   |   3.41   |   80.28   |   94.80   |    [config](./riformer-s24_8xb128_in1k.py)    | [model](https://download.openmmlab.com/mmclassification/v1/riformer/riformer-s24_32xb128_in1k_20230406-fdab072a.pth) |
|   riformer-s36_in1k   |  224x224   |   30.86   |   5.00   |   81.29   |   95.41   |    [config](./riformer-s36_8xb128_in1k.py)    | [model](https://download.openmmlab.com/mmclassification/v1/riformer/riformer-s36_32xb128_in1k_20230406-fdfcd3b0.pth) |
|   riformer-m36_in1k   |  224x224   |   56.17   |   8.80   |   82.57   |   95.99   |    [config](./riformer-m36_8xb128_in1k.py)    | [model](https://download.openmmlab.com/mmclassification/v1/riformer/riformer-m36_32xb128_in1k_20230406-2fcb9d9b.pth) |
|   riformer-m48_in1k   |  224x224   |   73.47   |  11.59   |   82.75   |   96.11   |    [config](./riformer-m48_8xb64_in1k.py)     | [model](https://download.openmmlab.com/mmclassification/v1/riformer/riformer-m48_32xb128_in1k_20230406-2b9d1abf.pth) |
| riformer-s12_384_in1k |  384x384   |   11.92   |   5.36   |   78.29   |   93.93   | [config](./riformer-s12_8xb128_in1k-384px.py) | [model](https://download.openmmlab.com/mmclassification/v1/riformer/riformer-s12_32xb128_in1k-384px_20230406-145eda4c.pth) |
| riformer-s24_384_in1k |  384x384   |   21.39   |  10.03   |   81.36   |   95.40   | [config](./riformer-s24_8xb128_in1k-384px.py) | [model](https://download.openmmlab.com/mmclassification/v1/riformer/riformer-s24_32xb128_in1k-384px_20230406-bafae7ab.pth) |
| riformer-s36_384_in1k |  384x384   |   30.86   |  14.70   |   82.22   |   95.95   | [config](./riformer-s36_8xb64_in1k-384px.py)  | [model](https://download.openmmlab.com/mmclassification/v1/riformer/riformer-s36_32xb128_in1k-384px_20230406-017ed3c4.pth) |
| riformer-m36_384_in1k |  384x384   |   56.17   |  25.87   |   83.39   |   96.40   | [config](./riformer-m36_8xb64_in1k-384px.py)  | [model](https://download.openmmlab.com/mmclassification/v1/riformer/riformer-m36_32xb128_in1k-384px_20230406-66a6f764.pth) |
| riformer-m48_384_in1k |  384x384   |   73.47   |  34.06   |   83.70   |   96.60   | [config](./riformer-m48_8xb64_in1k-384px.py)  | [model](https://download.openmmlab.com/mmclassification/v1/riformer/riformer-m48_32xb128_in1k-384px_20230406-2e874826.pth) |

The config files of these models are only for inference.

## Citation

```bibtex
@inproceedings{wang2023riformer,
  title={RIFormer: Keep Your Vision Backbone Effective But Removing Token Mixer},
  author={Wang, Jiahao and Zhang, Songyang and Liu, Yong and Wu, Taiqiang and Yang, Yujiu and Liu, Xihui and Chen, Kai and Luo, Ping and Lin, Dahua},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023}
}
```
