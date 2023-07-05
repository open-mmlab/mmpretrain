# RepLKNet

> [Scaling Up Your Kernels to 31x31: Revisiting Large Kernel Design in CNNs](https://arxiv.org/abs/2203.06717)

<!-- [ALGORITHM] -->

## Abstract

We revisit large kernel design in modern convolutional neural networks (CNNs). Inspired by recent advances in vision transformers (ViTs), in this paper, we demonstrate that using a few large convolutional kernels instead of a stack of small kernels could be a more powerful paradigm. We suggested five guidelines, e.g., applying re-parameterized large depth-wise convolutions, to design efficient highperformance large-kernel CNNs. Following the guidelines, we propose RepLKNet, a pure CNN architecture whose kernel size is as large as 31×31, in contrast to commonly used 3×3. RepLKNet greatly closes the performance gap between CNNs and ViTs, e.g., achieving comparable or superior results than Swin Transformer on ImageNet and a few typical downstream tasks, with lower latency. RepLKNet also shows nice scalability to big data and large models, obtaining 87.8% top-1 accuracy on ImageNet and 56.0% mIoU on ADE20K, which is very competitive among the state-of-the-arts with similar model sizes. Our study further reveals that, in contrast to small-kernel CNNs, large kernel CNNs have much larger effective receptive fields and higher shape bias rather than texture bias.

<div align=center>
<img src="https://user-images.githubusercontent.com/48375204/197546040-cdf078c3-7fbd-400f-8b27-01668c8dfebf.png" width="60%"/>
</div>

## How to use it?

<!-- [TABS-BEGIN] -->

**Predict image**

```python
from mmpretrain import inference_model, get_model

model = get_model('replknet-31B_3rdparty_in1k', pretrained=True)
model.backbone.switch_to_deploy()
predict = inference_model(model, 'demo/bird.JPEG')
print(predict['pred_class'])
print(predict['pred_score'])
```

**Use the model**

```python
import torch
from mmpretrain import get_model

model = get_model('replknet-31B_3rdparty_in1k', pretrained=True)
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
python tools/test.py configs/replknet/replknet-31B_32xb64_in1k.py https://download.openmmlab.com/mmclassification/v0/replknet/replknet-31B_3rdparty_in1k_20221118-fd08e268.pth
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
from mmpretrain.models import RepLKNet

backbone = RepLKNet(arch='31B')
backbone.switch_to_deploy()
```

<!-- [TABS-END] -->

## Models and results

### Image Classification on ImageNet-1k

| Model                                          |   Pretrain   | Params (M) | Flops (G) | Top-1 (%) | Top-5 (%) |                   Config                    |                            Download                            |
| :--------------------------------------------- | :----------: | :--------: | :-------: | :-------: | :-------: | :-----------------------------------------: | :------------------------------------------------------------: |
| `replknet-31B_3rdparty_in1k`\*                 | From scratch |   79.86    |   15.64   |   83.48   |   96.57   |    [config](replknet-31B_32xb64_in1k.py)    | [model](https://download.openmmlab.com/mmclassification/v0/replknet/replknet-31B_3rdparty_in1k_20221118-fd08e268.pth) |
| `replknet-31B_3rdparty_in1k-384px`\*           | From scratch |   79.86    |   45.95   |   84.84   |   97.34   | [config](replknet-31B_32xb64_in1k-384px.py) | [model](https://download.openmmlab.com/mmclassification/v0/replknet/replknet-31B_3rdparty_in1k-384px_20221118-03a170ce.pth) |
| `replknet-31B_in21k-pre_3rdparty_in1k`\*       | ImageNet-21k |   79.86    |   15.64   |   85.20   |   97.56   |    [config](replknet-31B_32xb64_in1k.py)    | [model](https://download.openmmlab.com/mmclassification/v0/replknet/replknet-31B_in21k-pre_3rdparty_in1k_20221118-54ed5c46.pth) |
| `replknet-31B_in21k-pre_3rdparty_in1k-384px`\* | ImageNet-21k |   79.86    |   45.95   |   85.99   |   97.75   | [config](replknet-31B_32xb64_in1k-384px.py) | [model](https://download.openmmlab.com/mmclassification/v0/replknet/replknet-31B_in21k-pre_3rdparty_in1k-384px_20221118-76c92b24.pth) |
| `replknet-31L_in21k-pre_3rdparty_in1k-384px`\* | ImageNet-21k |   172.67   |   97.24   |   86.63   |   98.00   | [config](replknet-31L_32xb64_in1k-384px.py) | [model](https://download.openmmlab.com/mmclassification/v0/replknet/replknet-31L_in21k-pre_3rdparty_in1k-384px_20221118-dc3fc07c.pth) |
| `replknet-XL_meg73m-pre_3rdparty_in1k-320px`\* |    MEG73M    |   335.44   |  129.57   |   87.57   |   98.39   | [config](replknet-XL_32xb64_in1k-320px.py)  | [model](https://download.openmmlab.com/mmclassification/v0/replknet/replknet-XL_meg73m-pre_3rdparty_in1k-320px_20221118-88259b1d.pth) |

*Models with * are converted from the [official repo](https://github.com/DingXiaoH/RepLKNet-pytorch/blob/main/replknet.py). The config files of these models are only for inference. We haven't reproduce the training results.*

## Citation

```bibtex
@inproceedings{ding2022scaling,
  title={Scaling up your kernels to 31x31: Revisiting large kernel design in cnns},
  author={Ding, Xiaohan and Zhang, Xiangyu and Han, Jungong and Ding, Guiguang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11963--11975},
  year={2022}
}
```
