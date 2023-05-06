# Inference with existing models

This note will show how to use the following APIs：

1. [**`list_models`**](mmpretrain.apis.list_models) & [**`get_model`**](mmpretrain.apis.get_model) ：list the model in MMPreTrain and get the model.
2. [**`ImageClassificationInferencer`**](mmpretrain.apis.ImageClassificationInferencer): inference on given images.
3. [**`FeatureExtractor`**](mmpretrain.apis.FeatureExtractor): extract features from the image files directly.

For more details about the pre-trained models in MMPretrain, you can refer to [Model Zoo](../modelzoo_statistics.md).

## List models and Get model

list all the models in MMPreTrain.

```
>>> from mmpretrain import list_models
>>> list_models()
['barlowtwins_resnet50_8xb256-coslr-300e_in1k',
 'beit-base-p16_beit-in21k-pre_3rdparty_in1k',
 .................]
```

`list_models` supports fuzzy matching, you can use **\*** to match any character.

```
>>> from mmpretrain import list_models
>>> list_models("*convnext-b*21k")
['convnext-base_3rdparty_in21k',
 'convnext-base_in21k-pre-3rdparty_in1k-384px',
 'convnext-base_in21k-pre_3rdparty_in1k']
```

you can use `get_model` get the model.

```
>>> from mmpretrain import get_model

# model without pre-trained weight
>>> model = get_model("convnext-base_in21k-pre_3rdparty_in1k")

# model with default weight in MMPreTrain
>>> model = get_model("convnext-base_in21k-pre_3rdparty_in1k", pretrained=True)

# model with weight in local
>>> model = get_model("convnext-base_in21k-pre_3rdparty_in1k", pretrained="your_local_checkpoint_path")

# you can also do some modification, like modify the num_classes in head.
>>> model = get_model("convnext-base_in21k-pre_3rdparty_in1k", head=dict(num_classes=10))

# you can get model without neck, head, and output from stage 1, 2, 3 in backbone
>>> model_headless = get_model("resnet18_8xb32_in1k", head=None, neck=None, backbone=dict(out_indices=(1, 2, 3)))
```

Then you can do the forward:

```
>>> import torch
>>> from mmpretrain import get_model
>>> model = get_model('convnext-base_in21k-pre_3rdparty_in1k', pretrained=True)
>>> x = torch.rand((1, 3, 224, 224))
>>> y = model(x)
>>> print(type(y), y.shape)
<class 'torch.Tensor'> torch.Size([1, 1000])
```

## Inference on a given image

Here is an example of building the inferencer on a [given image](https://github.com/open-mmlab/mmpretrain/raw/main/demo/demo.JPEG) by using ImageNet-1k pre-trained checkpoint.

```python
>>> from mmpretrain import ImageClassificationInferencer

>>> inferencer = ImageClassificationInferencer('resnet50_8xb32_in1k')
>>> results = inferencer('https://github.com/open-mmlab/mmpretrain/raw/main/demo/demo.JPEG')
>>> print(results[0]['pred_class'])
sea snake
```

`result` is a dictionary containing `pred_label`, `pred_score`, `pred_scores` and `pred_class`, the result is as follows:

```text
{"pred_label":65,"pred_score":0.6649366617202759,"pred_class":"sea snake", "pred_scores": [..., 0.6649366617202759, ...]}
```

If you want to use your own config and checkpoint:

```
>>> from mmpretrain import ImageClassificationInferencer
>>> inferencer = ImageClassificationInferencer(
            model='configs/resnet/resnet50_8xb32_in1k.py',
            pretrained='https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth',
            device='cuda')
>>> inferencer('https://github.com/open-mmlab/mmpretrain/raw/main/demo/demo.JPEG')
```

You can also inference multiple images by batch on CUDA:

```python
>>> from mmpretrain import ImageClassificationInferencer

>>> inferencer = ImageClassificationInferencer('resnet50_8xb32_in1k', device='cuda')
>>> imgs = ['https://github.com/open-mmlab/mmpretrain/raw/main/demo/demo.JPEG'] * 5
>>> results = inferencer(imgs, batch_size=2)
>>> print(results[1]['pred_class'])
sea snake
```

## Extract Features From Image

Compared with `model.extract_feat`, `FeatureExtractor` is used to extract features from the image files directly, instead of a batch of tensors.
In a word, the input of `model.extract_feat` is `torch.Tensor`, the input of `FeatureExtractor` is images.

```
>>> from mmpretrain import FeatureExtractor, get_model
>>> model = get_model('resnet50_8xb32_in1k', backbone=dict(out_indices=(0, 1, 2, 3)))
>>> extractor = FeatureExtractor(model)
>>> features = extractor('https://github.com/open-mmlab/mmpretrain/raw/main/demo/demo.JPEG')[0]
>>> features[0].shape, features[1].shape, features[2].shape, features[3].shape
(torch.Size([256]), torch.Size([512]), torch.Size([1024]), torch.Size([2048]))
```
