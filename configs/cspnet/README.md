# CSPNet

> [CSPNet: A New Backbone that can Enhance Learning Capability of CNN](https://arxiv.org/abs/1911.11929)

<!-- [ALGORITHM] -->

## Abstract

Neural networks have enabled state-of-the-art approaches to achieve incredible results on computer vision tasks such as object detection. However, such success greatly relies on costly computation resources, which hinders people with cheap devices from appreciating the advanced technology. In this paper, we propose Cross Stage Partial Network (CSPNet) to mitigate the problem that previous works require heavy inference computations from the network architecture perspective. We attribute the problem to the duplicate gradient information within network optimization. The proposed networks respect the variability of the gradients by integrating feature maps from the beginning and the end of a network stage, which, in our experiments, reduces computations by 20% with equivalent or even superior accuracy on the ImageNet dataset, and significantly outperforms state-of-the-art approaches in terms of AP50 on the MS COCO object detection dataset. The CSPNet is easy to implement and general enough to cope with architectures based on ResNet, ResNeXt, and DenseNet. Source code is at this https URL.

<div align=center>
<img src="https://user-images.githubusercontent.com/18586273/159420842-6147c687-a488-460c-8bb2-4ea5276c26c7.png" width="60%"/>
</div>

## How to use it?

<!-- [TABS-BEGIN] -->

**Predict image**

```python
from mmpretrain import inference_model

predict = inference_model('cspdarknet50_3rdparty_8xb32_in1k', 'demo/bird.JPEG')
print(predict['pred_class'])
print(predict['pred_score'])
```

**Use the model**

```python
import torch
from mmpretrain import get_model

model = get_model('cspdarknet50_3rdparty_8xb32_in1k', pretrained=True)
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
python tools/test.py configs/cspnet/cspdarknet50_8xb32_in1k.py https://download.openmmlab.com/mmclassification/v0/cspnet/cspdarknet50_3rdparty_8xb32_in1k_20220329-bd275287.pth
```

<!-- [TABS-END] -->

## Models and results

### Image Classification on ImageNet-1k

| Model                                |   Pretrain   | Params (M) | Flops (G) | Top-1 (%) | Top-5 (%) |                Config                |                                    Download                                     |
| :----------------------------------- | :----------: | :--------: | :-------: | :-------: | :-------: | :----------------------------------: | :-----------------------------------------------------------------------------: |
| `cspdarknet50_3rdparty_8xb32_in1k`\* | From scratch |   27.64    |   5.04    |   80.05   |   95.07   | [config](cspdarknet50_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/cspnet/cspdarknet50_3rdparty_8xb32_in1k_20220329-bd275287.pth) |
| `cspresnet50_3rdparty_8xb32_in1k`\*  | From scratch |   21.62    |   3.48    |   79.55   |   94.68   | [config](cspresnet50_8xb32_in1k.py)  | [model](https://download.openmmlab.com/mmclassification/v0/cspnet/cspresnet50_3rdparty_8xb32_in1k_20220329-dd6dddfb.pth) |
| `cspresnext50_3rdparty_8xb32_in1k`\* | From scratch |   20.57    |   3.11    |   79.96   |   94.96   | [config](cspresnext50_8xb32_in1k.py) | [model](https://download.openmmlab.com/mmclassification/v0/cspnet/cspresnext50_3rdparty_8xb32_in1k_20220329-2cc84d21.pth) |

*Models with * are converted from the [official repo](https://github.com/rwightman/pytorch-image-models). The config files of these models are only for inference. We haven't reproduce the training results.*

## Citation

```bibtex
@inproceedings{wang2020cspnet,
  title={CSPNet: A new backbone that can enhance learning capability of CNN},
  author={Wang, Chien-Yao and Liao, Hong-Yuan Mark and Wu, Yueh-Hua and Chen, Ping-Yang and Hsieh, Jun-Wei and Yeh, I-Hau},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops},
  pages={390--391},
  year={2020}
}
```
