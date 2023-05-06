# 使用现有模型进行推理

本文将展示如何使用以下API：

1. [**`list_models`**](mmpretrain.apis.list_models) 和 [**`get_model`**](mmpretrain.apis.get_model) ：列出 MMPreTrain 中的模型并获取模型。
2. [**`ImageClassificationInferencer`**](mmpretrain.apis.ImageClassificationInferencer): 在给定图像上进行推理。
3. [**`FeatureExtractor`**](mmpretrain.apis.FeatureExtractor): 从图像文件直接提取特征。

## 列出模型和获取模型

列出 MMPreTrain 中的所有已支持的模型。

```
>>> from mmpretrain import list_models
>>> list_models()
['barlowtwins_resnet50_8xb256-coslr-300e_in1k',
 'beit-base-p16_beit-in21k-pre_3rdparty_in1k',
 .................]
```

`list_models` 支持模糊匹配，您可以使用 **\*** 匹配任意字符。

```
>>> from mmpretrain import list_models
>>> list_models("*convnext-b*21k")
['convnext-base_3rdparty_in21k',
 'convnext-base_in21k-pre-3rdparty_in1k-384px',
 'convnext-base_in21k-pre_3rdparty_in1k']
```

了解了已经支持了哪些模型后，你可以使用 `get_model` 获取特定模型。

```
>>> from mmpretrain import get_model

# 没有预训练权重的模型
>>> model = get_model("convnext-base_in21k-pre_3rdparty_in1k")

# 使用MMPreTrain中默认的权重
>>> model = get_model("convnext-base_in21k-pre_3rdparty_in1k", pretrained=True)

# 使用本地权重
>>> model = get_model("convnext-base_in21k-pre_3rdparty_in1k", pretrained="your_local_checkpoint_path")

# 您还可以做一些修改，例如修改 head 中的 num_classes。
>>> model = get_model("convnext-base_in21k-pre_3rdparty_in1k", head=dict(num_classes=10))

# 您可以获得没有 neck，head 的模型，并直接从 backbone 中的 stage 1, 2, 3 输出
>>> model_headless = get_model("resnet18_8xb32_in1k", head=None, neck=None, backbone=dict(out_indices=(1, 2, 3)))
```

得到模型后，你可以进行推理：

```
>>> import torch
>>> from mmpretrain import get_model
>>> model = get_model('convnext-base_in21k-pre_3rdparty_in1k', pretrained=True)
>>> x = torch.rand((1, 3, 224, 224))
>>> y = model(x)
>>> print(type(y), y.shape)
<class 'torch.Tensor'> torch.Size([1, 1000])
```

## 在给定图像上进行推理

这是一个使用 ImageNet-1k 预训练权重在给定图像上构建推理器的示例。

```
>>> from mmpretrain import ImageClassificationInferencer

>>> inferencer = ImageClassificationInferencer('resnet50_8xb32_in1k')
>>> results = inferencer('https://github.com/open-mmlab/mmpretrain/raw/main/demo/demo.JPEG')
>>> print(results[0]['pred_class'])
sea snake
```

result 是一个包含 pred_label、pred_score、pred_scores 和 pred_class 的字典，结果如下：

```{text}
{"pred_label":65,"pred_score":0.6649366617202759,"pred_class":"sea snake", "pred_scores": [..., 0.6649366617202759, ...]}
```

如果你想使用自己的配置和权重：

```
>>> from mmpretrain import ImageClassificationInferencer
>>> inferencer = ImageClassificationInferencer(
            model='configs/resnet/resnet50_8xb32_in1k.py',
            pretrained='https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth',
            device='cuda')
>>> inferencer('https://github.com/open-mmlab/mmpretrain/raw/main/demo/demo.JPEG')
```

你还可以在CUDA上通过批处理进行多个图像的推理：

```{python}
>>> from mmpretrain import ImageClassificationInferencer

>>> inferencer = ImageClassificationInferencer('resnet50_8xb32_in1k', device='cuda')
>>> imgs = ['https://github.com/open-mmlab/mmpretrain/raw/main/demo/demo.JPEG'] * 5
>>> results = inferencer(imgs, batch_size=2)
>>> print(results[1]['pred_class'])
sea snake
```

## 从图像中提取特征

与 `model.extract_feat` 相比，`FeatureExtractor` 用于直接从图像文件中提取特征，而不是从一批张量中提取特征。简单说，`model.extract_feat` 的输入是 `torch.Tensor`，`FeatureExtractor` 的输入是图像。

```
>>> from mmpretrain import FeatureExtractor, get_model
>>> model = get_model('resnet50_8xb32_in1k', backbone=dict(out_indices=(0, 1, 2, 3)))
>>> extractor = FeatureExtractor(model)
>>> features = extractor('https://github.com/open-mmlab/mmpretrain/raw/main/demo/demo.JPEG')[0]
>>> features[0].shape, features[1].shape, features[2].shape, features[3].shape
(torch.Size([256]), torch.Size([512]), torch.Size([1024]), torch.Size([2048]))
```
