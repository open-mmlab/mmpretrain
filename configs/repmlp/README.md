# RepMLP

> [RepMLP: Re-parameterizing Convolutions into Fully-connected Layers for Image Recognition](https://arxiv.org/abs/2105.01883)

<!-- [ALGORITHM] -->

## Abstract

We propose RepMLP, a multi-layer-perceptron-style neural network building block for image recognition, which is composed of a series of fully-connected (FC) layers. Compared to convolutional layers, FC layers are more efficient, better at modeling the long-range dependencies and positional patterns, but worse at capturing the local structures, hence usually less favored for image recognition. We propose a structural re-parameterization technique that adds local prior into an FC to make it powerful for image recognition. Specifically, we construct convolutional layers inside a RepMLP during training and merge them into the FC for inference. On CIFAR, a simple pure-MLP model shows performance very close to CNN. By inserting RepMLP in traditional CNN, we improve ResNets by 1.8% accuracy on ImageNet, 2.9% for face recognition, and 2.3% mIoU on Cityscapes with lower FLOPs. Our intriguing findings highlight that combining the global representational capacity and positional perception of FC with the local prior of convolution can improve the performance of neural network with faster speed on both the tasks with translation invariance (e.g., semantic segmentation) and those with aligned images and positional patterns (e.g., face recognition).

<div align=center>
<img src="https://user-images.githubusercontent.com/18586273/155455288-a17a5c48-11af-4b74-995a-cf7183f0e2d2.png" width="80%"/>
</div>

## How to use it?

<!-- [TABS-BEGIN] -->

**Predict image**

```python
from mmpretrain import inference_model, get_model

model = get_model('repmlp-base_3rdparty_8xb64_in1k', pretrained=True)
model.backbone.switch_to_deploy()
predict = inference_model(model, 'demo/bird.JPEG')
print(predict['pred_class'])
print(predict['pred_score'])
```

**Use the model**

```python
import torch
from mmpretrain import get_model

model = get_model('repmlp-base_3rdparty_8xb64_in1k', pretrained=True)
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
python tools/test.py configs/repmlp/repmlp-base_8xb64_in1k.py https://download.openmmlab.com/mmclassification/v0/repmlp/repmlp-base_3rdparty_8xb64_in1k_20220330-1cb1f11b.pth
```

**Reparameterization**

The checkpoints provided are all `training-time` models. Use the reparameterize tool to switch them to more efficient `inference-time` architecture, which not only has fewer parameters but also less calculations.

```bash
python tools/convert_models/reparameterize_model.py ${CFG_PATH} ${SRC_CKPT_PATH} ${TARGET_CKPT_PATH}
```

`${CFG_PATH}` is the config file, `${SRC_CKPT_PATH}` is the source chenpoint file, `${TARGET_CKPT_PATH}` is the target deploy weight file path.

To use reparameterized weights, the config file must switch to the deploy config files.

```bash
python tools/test.py ${deploy_cfg} ${deploy_checkpoint} --metrics accuracy
```

You can also use `backbone.switch_to_deploy()` to switch to the deploy mode in Python code. For example:

```python
from mmpretrain.models import RepMLPNet

backbone = RepMLPNet(arch='B', img_size=224, reparam_conv_kernels=(1, 3))
backbone.switch_to_deploy()
```

<!-- [TABS-END] -->

## Models and results

### Image Classification on ImageNet-1k

| Model                                     |   Pretrain   | Params (M) | Flops (G) | Top-1 (%) | Top-5 (%) |                  Config                   |                               Download                                |
| :---------------------------------------- | :----------: | :--------: | :-------: | :-------: | :-------: | :---------------------------------------: | :-------------------------------------------------------------------: |
| `repmlp-base_3rdparty_8xb64_in1k`\*       | From scratch |   68.24    |   6.71    |   80.41   |   95.14   |    [config](repmlp-base_8xb64_in1k.py)    | [model](https://download.openmmlab.com/mmclassification/v0/repmlp/repmlp-base_3rdparty_8xb64_in1k_20220330-1cb1f11b.pth) |
| `repmlp-base_3rdparty_8xb64_in1k-256px`\* | From scratch |   96.45    |   9.69    |   81.11   |   95.50   | [config](repmlp-base_8xb64_in1k-256px.py) | [model](https://download.openmmlab.com/mmclassification/v0/repmlp/repmlp-base_3rdparty_8xb64_in1k-256px_20220330-7c5a91ce.pth) |

*Models with * are converted from the [official repo](https://github.com/DingXiaoH/RepMLP/blob/072d8516beba83d75dfe6ebb12f625abad4b53d5/repmlpnet.py#L278). The config files of these models are only for inference. We haven't reproduce the training results.*

## Citation

```bibtex
@article{ding2021repmlp,
  title={Repmlp: Re-parameterizing convolutions into fully-connected layers for image recognition},
  author={Ding, Xiaohan and Xia, Chunlong and Zhang, Xiangyu and Chu, Xiaojie and Han, Jungong and Ding, Guiguang},
  journal={arXiv preprint arXiv:2105.01883},
  year={2021}
}
```
