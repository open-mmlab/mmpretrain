# Tutorial 0: Create Backbones in mmcls

In the neural network, the backbone network is an important way to extract features. In tasks such as classification, detection, segmentation, video understanding and etc., the backbone network is an essential component.Since image feature extraction is more general for various visual tasks, it can "borrow" the backbone network and corresponding model weights pre-trained in classification tasks. Because the classification task is relatively simple, the huge ImageNet data set can be used for pre-training, and further training the detection network on this basis can not only improve the convergence speed of the model, but also improve the accuracy.

In the OpenMMLab project, MMClassification not only acts as a toolbox and benchmark for classification tasks, but also acts as a backbone network library. This document mainly describes how to use these backbone networks. Part of the code in the document can only be run after configuring the required environment. For details, see [Installation](https://mmclassification.readthedocs.io/en/latest/install.html).

<!-- TOC -->

- [Instantiate Backbones](#instantiate-backbones)
- [Using Backbones in the downstream OpenMMLab libraries](#using-backbones-in-the-downstream-openmmlab-libraries)
- [Using TIMMBackbone](#Using-TIMMBackbone)

<!-- TOC -->

## Instantiate Backbones

MMClassification provides the `build_backbone` interface to instantiate the backbone network. The incoming parameter is a dictionary, including the backbone network type parameter `type` and other configurable parameters. The backbone network currently supported by MMClassification can be found in [here](https://github.com/open-mmlab/mmclassification/blob/master/mmcls/models/backbones/__init__.py), and the relevant parameters of the backbone network can be found in [API ](https://mmclassification.readthedocs.io/en/latest/api.html#module-mmcls.models.backbones).

### Basic interface `build_backbone`

***ResNet50***
```
import torch
from mmcls.models import build_backbone
x = torch.randn( (1, 3, 224, 224) )

# instantiate a resnet backbone, the depth resnet can be 18, 34, 50, 101, 152, refer the backbone API
backbone_cfg = dict(type='ResNet', depth=50)
model = build_backbone(backbone_cfg)
model.init_weights()

outs = model(x)   # output a tuple of torch.Tensor
for out in outs:
    print(out.size())
```

Output
```
torch.Size([1, 2048, 7, 7])
```

***ViT***

in ViT, using parameter `arch` to specify bancbone architecture, such as 'tiny', 'base' and 'lagre'.

```
import torch
from mmcls.models import build_backbone
x = torch.randn( (1, 3, 224, 224) )

backbone_cfg = dict(
    type='VisionTransformer',
    arch='base',
    output_cls_token=False,  # whether to put cls_token
    final_norm=False)
model = build_backbone(backbone_cfg)
model.init_weights()

outs = model(x)   # output a tuple of `torch.Tensor`
for out in outs:
    print(out.size())
```

Output:
```
torch.Size([1, 768, 14, 14])
```

### Output multi-feature map

In detection and segmentation tasks, the backbone network is often required to output multiple feature maps for multi-scale processing. In order to meet this requirement, all backbone networks of MMClassification support multi-feature map output, which is configured by the parameter `out_indices`, generally a ` Sequence[int]`, such as `tuple` or `list`.

***ResNet***
```
import torch
from mmcls.models import build_backbone
x = torch.randn( (1, 3, 224, 224) )

backbone_cfg = dict(type='ResNet', depth=50, out_indices=(0, 1, 2, 3))
model = build_backbone(backbone_cfg)
model.init_weights()

outs = model(x)
for out in outs:
    print(out.size())
```

Output:
```
torch.Size([1, 256, 56, 56])
torch.Size([1, 512, 28, 28])
torch.Size([1, 1024, 14, 14])
torch.Size([1, 2048, 7, 7])
```

### Using Pre-trained checkpoints

MMClassification initializes the weights of the model through `init_cfg`, which includes loading the already trained weights. `init_cfg` needs to be set to a dictionary data type. When loading pre-training, 'type' needs to be set to 'Pretrained'; 'checkpoint' is set to the weight path to be loaded, or it can be a link; 'prefix' means to select a certain part of the weight to load, the pre-trained models provided by MMClassification are all is the `classifier` model, here only the 'backbone' part of the weights are loaded.

***ResNet50 Example***

MMClassification provides several checkpoints of `ResNet50`, including:

|         model         |  Link  |  Top-1 Acc  |  info |
|:---------------------:|:--------:| :--------:| :--------:|
| ResNet50 |  [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth) | 76.55 | Baseline of ResNet50 in ImagNet-1k |
| ResNet50(rsb-a1) | [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb256-rsb-a1-600e_in1k_20211228-20e21305.pth) | 80.12 | Elite model of ResNet50-rsb in ImagNet-1k |
| ResNet50-miil |  [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_mill_3rdparty_in21k_20220307-bdb3a68b.pth) | - | Pre-trained model of ResNet50 with mutil label in ImagNet-21k |

```
import torch
from mmcls.models import build_backbone
x = torch.randn( (1, 3, 224, 224) )

checkpoint_path = 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth'
backbone_cfg = dict(
    type='ResNet',
    depth=50,
    out_indices=(0, 1, 2, 3),
    init_cfg = dict(
        type='Pretrained',   # 'Pretrained' mean loading from provided weights
        checkpoint=checkpoint_path,
        prefix='backbone')         # only load 'backbone' weights
)
model = build_backbone(backbone_cfg)
model.init_weights()

outs = model(x)   # 输出为一个 tuple
for out in outs:
    print(out.size())
```

Output
```
torch.Size([1, 256, 56, 56])
torch.Size([1, 512, 28, 28])
torch.Size([1, 1024, 14, 14])
torch.Size([1, 2048, 7, 7])
```


### Frozen stages

In some downstream tasks, especially tasks with very small datasets, after loading the pre-trained model, in order to reduce the effect of overfitting, it is necessary to freeze the pre-supplemented network layers (including BN) in the backbone network. MMClassification provides `frozen_stage` to configure this setting, which requires an `int` data to be passed in. The default is '-1', which means not to freeze any network layer; '0' means to freeze ***`stage0` and its previous network layers* ** (part of the algorithm is `layer0`).


**Example:**

```
import torch
from mmcls.models import build_backbone
x = torch.randn( (1, 3, 224, 224) )
from mmcv.utils.parrots_wrapper import _BatchNorm

checkpoint_path = 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth'
frozen_stages = 2
backbone_cfg = dict(
    type='ResNet',
    depth=50,
    out_indices=(0, 1, 2, 3),
    frozen_stages=frozen_stages,
    init_cfg = dict(
        type='Pretrained',
        checkpoint=checkpoint_path,
        prefix='backbone')
)
model = build_backbone(backbone_cfg)
model.init_weights()

for i in range(1, frozen_stages + 1):
    layer = getattr(model, f'layer{i}')
    for mod in layer.modules():
        if isinstance(mod, _BatchNorm):
            assert mod.training is False
    for param in layer.parameters():
        assert param.requires_grad is False
```

It should run successfully and end without assertion errors.

## Using Backbones in the downstream OpenMMLab libraries

MMClassification plays the role of **backbone network library** in the OpenMMLab project. There will be more types of backbone networks, and the latest backbone networks will be added in time. And mmdet, mmseg, mmtracking can directly use the backbone network of mmcls. For related project examples, please refer to [backbone-example](https://github.com/mzr1996/backbone-example).

**Using mmcls Backbone in mmdet**

If a backbone network has been implemented in the classification code base MMClassification, but has not been implemented in `mmdet`, we can directly call it across libraries by modifying `model.backbone` in the configuration file, note that you need to add ** `custom_imports=dict(imports='mmcls.models')`** Add domain information when registering the backbone network into the registrar and specifying the backbone network**`type=mmcls.{BACKBONE_CLASSNAME}`**, when you need to pay attention to the backbone network The number of input channels `in_channels` for `model.neck` may change when modified. For example, we want to change the backbone network in the above YoloX from CSPDarkNet (Yolov5 version) to CSPResNeXt, but mmdet has no implementation of CSPResNeXt, you can use the following configuration:

```
# Directly inherit the original config of ``yoloX``,
# link: https://github.com/open-mmlab/mmdetection/tree/master/configs/yolox
_base_ = "./yolox_s_8x8_300e_coco.py"

# Because there is no import mmcls in mmdet
# Therefore, the backbone network in it will not be registered with the manager
# Here we need to specify additional imports manually using custom_imports
# thereby registering the backbone network in mmcls
custom_imports=dict(imports='mmcls.models', allow_failed_imports=False)
model = dict(
    backbone=dict(
        # Use the "scope.type" syntax to specify the required modules from mmcls
        type='mmcls.CSPResNeXt',
        # CSPResNeXt50 setting
        arch=50,
        out_indices=(1, 2, 3),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='mmcls://cspresnext50'),
        # The dictionary of the same fields in the config and the inherited config will be merged by default
        # `_delete_` is used here to delete other configs in the inherited config file
        _delete_=True),
    # When the backbone network changes, other corresponding configurations also need to be changed.
    # Pay special attention to the change in the number of output channels of the backbone network.
    neck=dict(in_channels=[512, 1024, 2048])
)
```

**Using mmcls Backbone in mmsegmentation**

Backbone using `mmcls` in `mmsegmentation` is the same as in `mmdet`. As in `RegNet` using `mmcls` in `mmsegmentation`.

```
# Directly inherit the original config of ``uppernet``,
# Link: https://github.com/open-mmlab/mmsegmentation/tree/master/configs/upernet
_base_ = "./upernet_r50_512x512_80k_ade20k.py"

# Because there is no import mmcls in mmsegmentation
# Therefore, the backbone network in it will not be registered with the manager
# Here we need to specify additional imports manually using custom_imports
# thereby registering the backbone network in mmcls
custom_imports=dict(imports='mmcls.models', allow_failed_imports=False)
model = dict(
    backbone=dict(
        # Use the "scope.type" syntax to specify the required modules from mmcls
        type='mmcls.RegNet',
        # RegNet setting
        arch='regnetx_4.0gf',
        out_indices=(0, 1, 2, 3),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='mmcls://regnet-4g'),
        # The dictionary of the same fields in the config and the inherited config will be merged by default
        # `_delete_` is used here to delete other configs in the inherited config file
        _delete_=True),
    # When the backbone network changes, other corresponding configurations also need to be changed.
    # Pay special attention to the change in the number of output channels of the backbone network.
    decode_head=dict(in_channels=[80, 240, 560, 1360]),
    auxiliary_head=dict(in_channels=560),
)
```

## Using TIMMBackbone

[`TIMM`](https://github.com/rwightman/pytorch-image-models/) is a very popular PyTorch-based algorithm library, including a rich backbone network, widely used in academia and industry . Compared to MMClassification, `TIMM` has a very large advantage in model diversity, but the MMClassification code is more flexible for training and adding new components. MMCclassification implements the [`TIMMBackbone`](https://github.com/open-mmlab/mmclassification/blob/f0ee5dcb2aca434e972c7969a5cd6edc4d56c97a/mmcls/models/backbones/timm_backbone.py#L39) wrapper for more convenient use of the TIMM backbone. Note 1. If you want to output multiple feature maps, you need to add `features_only=True`; 2. If you want to load a pretrained model, you need to add `pretrained=True`.

For example, `EfficientNetv2` is not yet implemented in MMCclassification, but is implemented in `TIMM`, you can use `EfficientNetv2` in `mmcls` by:

```
import torch
from mmcls.models import build_backbone
x = torch.randn( (1, 3, 224, 224) )

backbone_cfg=dict(
    type="TIMMBackbone",
    pretrained=False,
    model_name="tf_efficientnetv2_s_in21k",
    features_only=True,
    out_indices=(2, 3, 4))
model = build_backbone(backbone_cfg)
model.init_weights()

outs = model(x)   # output a tuple
for out in outs:
    print(out.size())
```

Output:

```
torch.Size([1, 64, 28, 28])
torch.Size([1, 160, 14, 14])
torch.Size([1, 256, 7, 7])
```
