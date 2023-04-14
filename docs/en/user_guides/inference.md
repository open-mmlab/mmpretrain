# Inference with existing models

This note will show how to use the following APIs：

1. [**`list_models`**](mmpretrain.apis.list_models) & [**`get_model`**](<(mmpretrain.apis.get_model)>) ：list the model in MMPreTrain and get the model.
2. [**`ImageClassificationInferencer`**](mmpretrain.apis.ImageClassificationInferencer): inference on given images.
3. [**`FeatureExtractor`**](mmpretrain.apis.FeatureExtractor): extract features from the image files directly.
4. [**`ImageRetrievalInferencer`**](mmpretrain.apis.ImageRetrievalInferencer): retrieve images from a folder.

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

# get your owner pretrained model
>>> your_model = get_model("CONFIG_PATH", pretrained="CKPT_PATH")

# model without pre-trained weight
>>> model = get_model("convnext-base_in21k-pre_3rdparty_in1k")

# model with weight in MMPreTrain
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
>>> x = torch.rand((1, 3, 224, 224))
>>> y = model(x)
>>> print(type(y), y.shape)
<class 'torch.Tensor'> torch.Size([1, 1000])
```

## Inference on a given image

Here is an example of building the inferencer on a [given image](https://github.com/open-mmlab/mmpretrain/raw/main/demo/demo.JPEG) by using ImageNet-1k pre-trained checkpoint.

```python
>>> from mmpretrain import ImageClassificationInferencer

# inferencer = ImageClassificationInferencer('CONFIG_PATH', 'CKPT_PATH')
>>> inferencer = ImageClassificationInferencer('resnet50_8xb32_in1k')
>>> results = inferencer('https://github.com/open-mmlab/mmpretrain/raw/main/demo/demo.JPEG')
>>> print(results[0]['pred_class'])
sea snake
```

`result` is a dictionary containing `pred_label`, `pred_score`, `pred_scores` and `pred_class`, the result is as follows:

```text
{"pred_label":65,"pred_score":0.6649366617202759,"pred_class":"sea snake", "pred_scores": [..., 0.6649366617202759, ...]}
```

To inference multiple images by batch on CUDA

```python
>>> from mmpretrain import ImageClassificationInferencer

# inferencer = ImageClassificationInferencer('CONFIG_PATH', 'CKPT_PATH', device='cuda')
>>> inferencer = ImageClassificationInferencer('resnet50_8xb32_in1k', device='cuda')
>>> imgs = ['https://github.com/open-mmlab/mmpretrain/raw/main/demo/demo.JPEG'] * 5
>>> results = inferencer(imgs, batch_size=2)
>>> print(results[1]['pred_class'])
sea snake
```

An image demo can be found in [demo/image_demo.py](https://github.com/open-mmlab/mmpretrain/blob/main/demo/image_demo.py).
As for how to test existing models on standard datasets, please see this [guide](./test.md)

## Extract Features From Image

Compared with `model.extract_feat`, `FeatureExtractor` is used to extract features from the image files directly, instead of a batch of tensors.

```
>>> from mmpretrain import FeatureExtractor, get_model
>>> model = get_model('resnet50_8xb32_in1k', backbone=dict(out_indices=(0, 1, 2, 3)))
>>> extractor = FeatureExtractor(model)
>>> features = extractor('https://github.com/open-mmlab/mmpretrain/raw/main/demo/demo.JPEG')[0]
>>> features[0].shape, features[1].shape, features[2].shape, features[3].shape
(torch.Size([256]), torch.Size([512]), torch.Size([1024]), torch.Size([2048]))
```

## ImageRetrievalInferncer

```
>>> from mmpretrain import inference_model
>>> inference_model(
...     'resnet50-arcface_8xb32_inshop',
...     'demo/bird.JPEG',
...     prototype='data/imagenet/train/',       # The folder of images to retrieve.
...     prototype_vecs='proto.pkl')           # The path to save prototype vectors. And it will load this file in the next call.
```
