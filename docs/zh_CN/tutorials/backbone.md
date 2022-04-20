# 教程 0：如何调用MMClassification主干网络

在神经网络中，主干网络是一种重要的提取特征方式，在分类，检测，分割以及视频理解等等视觉任务中，主干网络都是必不可少的组成部分。
由于图片特征提取较为通用，因而可以“借用”在分类任务中预训练的主干网络和相应的模型权重。因为分类任务比较简单，故而可以利用庞大的 ImageNet 数据集进行预训练，而在此基础上进一步训练检测网络、分割网络以及其他下游任务中，既能够提高模型收敛速度，又能够提高精度。

MMClassification 在 OpenMMLab 项目中，不仅作为分类任务的工具箱以及基准，还充当**主干网络库**的角色，本文档主要介绍如何调用 MMClassification 的主干网络。文档中的示例，需要配置所需环境后才可运行，详见[安装](https://mmclassification.readthedocs.io/zh_CN/latest/install.html)。

<!-- TOC -->

- [通过接口构建主干网](#通过接口构建主干网)
- [在 OpenMMLab 下游算法库中调用](#在-openmmlab-下游算法库中调用)
- [调用 TIMM 中的主干网](#调用-timm-中的主干网)

<!-- TOC -->

## 通过接口构建主干网

MMClassification 提供了 `build_backbone` 接口实例化主干网络。其传入参数为一个字典或者 `mmcv.ConfigDict` 数据类型，包括主干网类别参数 'type' 以及其他可配置参数。目前 MMClassification 已支持的主干网络及相关参数可参考 [API](https://mmclassification.readthedocs.io/ch_CH/latest/api/models.html#backbones) 文档。

```note
使用 MMClassification 提供的相关工具，比如 train、test 等工具时会隐式调用 build_backbone 接口构造主干网络！
```

### 基本调用方法

- "type" : 主干网络算法，已支持算法可参考 [主干网络列表](https://mmclassification.readthedocs.io/ch_CH/latest/api/models.html#backbones)；
- "depth" 或者 "arch" ： 配置主干模型架构。

**ResNet50 示例**

在 `ResNet` 中， 由 'depth' 配置模型的架构， 可选项有 18, 34, 50, 101。

```
import torch
from mmcls.models import build_backbone
x = torch.randn( (1, 3, 224, 224) )

# 实例化一个resnet, resnet的depth可以18, 34, 50, 101, 152, 可参看 API
backbone_cfg = dict(type='ResNet', depth=50)
model = build_backbone(backbone_cfg)
model.init_weights()

outs = model(x)   # 输出为一个 tuple
for out in outs:
    print(out.size())
```

Output
```
torch.Size([1, 2048, 7, 7])
```

**ViT 示例**

在 `VisionTransformer` 中， 通过 'arch' 配置模型的架构， 可选项有 'tiny', 'base' 和 'lagre'。

```
import torch
from mmcls.models import build_backbone
x = torch.randn( (1, 3, 224, 224) )

backbone_cfg = dict(
    type='VisionTransformer',
    arch='base',
    output_cls_token=False,
    final_norm=False)
model = build_backbone(backbone_cfg)
model.init_weights()

outs = model(x)   # 输出为一个 tuple
for out in outs:
    print(out.size())
```

Output
```
torch.Size([1, 768, 14, 14])
```

### 输出多个特征图

在检测，分割等任务中，往往需要主干网络输出多个特征图进行多尺度的处理，为了满足这一需求 MMClassification 的所有主干网络都支持多特征图输出，由参数 `out_indices` 配置，一般为一个 `Sequence[int]`，如 `tuple` 或者 `list`。

**Swin-base 示例**

```
import torch
from mmcls.models import build_backbone
x = torch.randn( (1, 3, 224, 224) )

backbone_cfg = dict(type='SwinTransformer', arch="base", out_indices=(0, 1, 2, 3))
model = build_backbone(backbone_cfg)
model.init_weights()

outs = model(x)
for out in outs:
    print(out.size())
```

Output
```
torch.Size([1, 512, 28, 28])
torch.Size([1, 1024, 14, 14])
torch.Size([1, 2048, 7, 7])
torch.Size([1, 2048, 7, 7])
```

### 加载预训练模型

MMClassification 通过 `init_cfg` 初始化模型的权重，可参考 mmcv [相关教程](https://mmcv.readthedocs.io/zh_CN/latest/understand_mmcv/cnn.html#weight-initialization)，其中就包括了加载已经训练好的权重。 `init_cfg` 需要设置为一个字典数据类型。 加载预训练时，需要设置：

- `type`: 需设置为 'Pretrained';
- `checkpoint`: 设置为准备加载的权重路径，也可以为一个链接;
- `prefix`: 表示选取权重某个部分加载， MMClassification 提供的预训练模型都是 `classifier` 的模型(一般包括backbone、neck以及head)，这里只加载其中的 'backbone' 部分权重。


**ResNet50 示例**

MMClassification 提供了 `ResNet50` 的多个预训练模型，包括:

|         模型         |  链接  |  Top-1 Acc  |  备注 |
|:---------------------:|:--------:| :--------:| :--------:|
| ResNet50 |  [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth) | 76.55 | ResNet50在ImagNet-1k上的baseline |
| ResNet50(rsb-a1) | [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb256-rsb-a1-600e_in1k_20211228-20e21305.pth) | 80.12 | ResNet50-rsb在ImagNet-1k上的高精度模型 |
| ResNet50-miil |  [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_mill_3rdparty_in21k_20220307-bdb3a68b.pth) | - | ResNet50在ImagNet-21k上多标签模型的预训练模型 |

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
        type='Pretrained',   # 初始化类型，这里使用 'Pretrained'， 表示预训练
        checkpoint=checkpoint_path,
        prefix='backbone')          # 只需要加载预训练中 backbone 部分的权重\
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

### 冻结部分网络层

在一些下游任务，特别是数据集非常小的任务中，加载好预训练模型后，为了减小过拟合效应，需要冻结主干网络中前补的网络层（包括 BN ）。 MMClassification 提供 `frozen_stages` 配置该设置，要求传入一个 `int` 数据， 默认为 '-1'，表示不冻结任何网络层； '0' 表示冻结 **`stage0` 以及其之前的网络层**（部分算法为 `layer0`）。


**ConvNeXt 示例**

```
import torch
from mmcls.models import build_backbone
x = torch.randn( (1, 3, 224, 224) )

checkpoint_path = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-base_3rdparty_32xb128-noema_in1k_20220301-2a0ee547.pth'
backbone_cfg = dict(
    type='ConvNeXt',
    arch='base',
    frozen_stages=2,
    out_indices=(0, 1, 2, 3),
    gap_before_final_norm=False,   # 非分类任务需要设置为 False
    init_cfg=dict(
        type='Pretrained', checkpoint=checkpoint_path, prefix='backbone.')
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

应该顺利运行结束无输出，没有报出断言错误。

## 在 OpenMMLab 下游算法库中调用

MMClassification 在 OpenMMLab 项目中充当**主干网络库**的角色，在主干网络的种类会更多，也会及时添加最新的主干网络。且mmdet, mmseg，mmtracking中可直接使用mmcls的主干网络，具体的技术实现可以参考[知乎技术文章](https://zhuanlan.zhihu.com/p/436865195)。相关项目实例可参考 [backbone-example](https://github.com/mzr1996/backbone-example)。

**在mmdet中使用mmcls的主干网络**

如果某个主干网络在分类代码库 MMClassification 中已经实现了，但在 `mmdet` 并没有被实现，我们是可以直接通过修改配置文件中 `model.backbone` 来跨库调用的, 需要注意以下三点：
1. 添加 **`custom_imports=dict(imports='mmcls.models')`** 注册主干网络入注册器；
2. 指定主干网络时添加域信息 **`type=mmcls.{BACKBONE_CLASSNAME}`**;
3. 需要注意时当主干网络修改时，`model.neck` 的输入通道数 `in_channels` 可能发生变化。

比如我们想把 `MMDet` 中 `YoloX` 中的主干网络从 `CSPDarkNet` 换成 `CSPResNeXt`，但 `MMDet` 还没有 `CSPResNeXt` 的实现，可以使用如下配置：

```
# 直接继承 yoloX 的原始配置，路径 https://github.com/open-mmlab/mmdetection/tree/master/configs/yolox
_base_ = "./yolox_s_8x8_300e_coco.py"

# 因为 mmdet 中没有 import mmcls
# 因而其中的主干网络并不会被注册到管理器中
# 这里我们需要手动用 custom_imports 来指定额外的导入
# 从而注册 mmcls 中的主干网络
custom_imports=dict(imports='mmcls.models', allow_failed_imports=False)
model = dict(
    backbone=dict(
        # 使用 "scope.type" 的语法，指定从 mmcls 中寻找需要的模块
        type='mmcls.CSPResNeXt',
        # MobileNet V3 的其他设置
        arch=50,
        out_indices=(1, 2, 3),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='mmcls://cspresnext50'),
        # 配置文件与继承的配置文件中相同字段的字典，默认会融合
        # 这里使用 `_delete_` 来删除继承的配置文件中的其他配置
        _delete_=True),
    # 主干网络发生变化，其他相应的配置也需要改变，特别注意主干网络输出通道数的变化
    neck=dict(in_channels=[512, 1024, 2048])
)
```

**在mmsegmentation中使用mmcls的主干网络**

在 `MMSegmentation` 中使用 `mmcls` 的主干网络与在 `MMDet` 的用法一致。 如在 `MMSegmentation` 中使用 `mmcls` 的 `RegNet`。

```
# 直接继承 uppernet 的原始配置，路径 https://github.com/open-mmlab/mmsegmentation/tree/master/configs/upernet
_base_ = "./upernet_r50_512x512_80k_ade20k.py"

# 因为 mmsegmentation 中没有 import mmcls
# 因而其中的主干网络并不会被注册到管理器中
# 这里我们需要手动用 custom_imports 来指定额外的导入
# 从而注册 mmcls 中的主干网络
custom_imports=dict(imports='mmcls.models', allow_failed_imports=False)
model = dict(
    backbone=dict(
        # 使用 "scope.type" 的语法，指定从 mmcls 中寻找需要的模块
        type='mmcls.RegNet',
        # RegNet 的其他设置
        arch='regnetx_4.0gf',
        out_indices=(0, 1, 2, 3),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='mmcls://regnet-4g'),
        # 配置文件与继承的配置文件中相同字段的字典，默认会融合
        # 这里使用 `_delete_` 来删除继承的配置文件中的其他配置
        _delete_=True),
    # 主干网络发生变化，其他相应的配置也需要改变, 特别注意主干网络输出通道数的变化
    decode_head=dict(in_channels=[80, 240, 560, 1360]),
    auxiliary_head=dict(in_channels=560),
)
```

## 调用 TIMM 中的主干网

[`TIMM`](https://github.com/rwightman/pytorch-image-models/) 是一个非常流行的基于 PyTorch 的算法库，包含丰富的主干网络，在学术界以及工业界都被广泛运用。相对于 MMClassification，`TIMM` 在模型多样性上有非常大的优势，但是 MMClassification 代码对于训练和添加新组件更加灵活。 MMCclassification 实现了 [`TIMMBackbone`](https://github.com/open-mmlab/mmclassification/blob/f0ee5dcb2aca434e972c7969a5cd6edc4d56c97a/mmcls/models/backbones/timm_backbone.py#L39) 包装器以更加方便的使用 TIMM 主干网络。需要注意:

1. 使用 'model_name' 指定算法名，架构以及其他训练信息，即 `timm.create_model` 的 'model_name';
2. 如果想输出多个特征图， 需要添加 `features_only=True`;
3. 如果想加载预训练模型，需要添加 `pretrained=True`。

比如，在 MMCclassification 还没有实现 `EfficientNetv2`, 但在 `TIMM` 中有实现却有实现，可以通过以下方式在 `mmcls` 中使用 `EfficientNetv2`：

```
import torch
from mmcls.models import build_backbone
x = torch.randn( (1, 3, 224, 224) )

backbone_cfg=dict(
    type="TIMMBackbone",    # 'type' 须为 `TIMMBackbone`
    pretrained=True,       # 加载预训练模型
    model_name="tf_efficientnetv2_s_in21k",  # 指定 efficientnetv2 在 imagnet21k 预训练好 的 small 结构模型。
    features_only=True,    # 输出多个特征图
    out_indices=(2, 3, 4))
model = build_backbone(backbone_cfg)
model.init_weights()

outs = model(x)   # 输出为一个 tuple
for out in outs:
    print(out.size())
```

输出：

```
torch.Size([1, 64, 28, 28])
torch.Size([1, 160, 14, 14])
torch.Size([1, 256, 7, 7])
```
