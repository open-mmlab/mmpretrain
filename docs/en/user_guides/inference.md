# Inference with existing models

This tutorial will show how to use the following APIs：

- [**`list_models`**](mmpretrain.apis.list_models): List available model names in MMPreTrain.
- [**`get_model`**](mmpretrain.apis.get_model): Get a model from model name or model config.
- [**`inference_model`**](mmpretrain.apis.inference_model): Inference a model with the correspondding
  inferencer. It's a shortcut for a quick start, and for advanced usage, please use the below inferencer
  directly.
- Inferencers:
  1. [**`ImageClassificationInferencer`**](mmpretrain.apis.ImageClassificationInferencer):
     Perform image classification on the given image.
  2. [**`ImageRetrievalInferencer`**](mmpretrain.apis.ImageRetrievalInferencer):
     Perform image-to-image retrieval from the given image on a given image set.
  3. [**`ImageCaptionInferencer`**](mmpretrain.apis.ImageCaptionInferencer):
     Generate a caption on the given image.
  4. [**`VisualQuestionAnsweringInferencer`**](mmpretrain.apis.VisualQuestionAnsweringInferencer):
     Answer a question according to the given image.
  5. [**`VisualGroundingInferencer`**](mmpretrain.apis.VisualGroundingInferencer):
     Locate an object from the description on the given image.
  6. [**`TextToImageRetrievalInferencer`**](mmpretrain.apis.TextToImageRetrievalInferencer):
     Perform text-to-image retrieval from the given description on a given image set.
  7. [**`ImageToTextRetrievalInferencer`**](mmpretrain.apis.ImageToTextRetrievalInferencer):
     Perform image-to-text retrieval from the given image on a series of text.
  8. [**`NLVRInferencer`**](mmpretrain.apis.NLVRInferencer):
     Perform Natural Language for Visual Reasoning on a given image-pair and text.
  9. [**`FeatureExtractor`**](mmpretrain.apis.FeatureExtractor):
     Extract features from the image files by a vision backbone.

## List available models

list all the models in MMPreTrain.

```python
>>> from mmpretrain import list_models
>>> list_models()
['barlowtwins_resnet50_8xb256-coslr-300e_in1k',
 'beit-base-p16_beit-in21k-pre_3rdparty_in1k',
 ...]
```

`list_models` supports Unix filename pattern matching, you can use \*\* * \*\* to match any character.

```python
>>> from mmpretrain import list_models
>>> list_models("*convnext-b*21k")
['convnext-base_3rdparty_in21k',
 'convnext-base_in21k-pre-3rdparty_in1k-384px',
 'convnext-base_in21k-pre_3rdparty_in1k']
```

You can use the `list_models` method of inferencers to get the available models of the correspondding tasks.

```python
>>> from mmpretrain import ImageCaptionInferencer
>>> ImageCaptionInferencer.list_models()
['blip-base_3rdparty_caption',
 'blip2-opt2.7b_3rdparty-zeroshot_caption',
 'flamingo_3rdparty-zeroshot_caption',
 'ofa-base_3rdparty-finetuned_caption']
```

## Get a model

you can use `get_model` get the model.

```python
>>> from mmpretrain import get_model

# Get model without loading pre-trained weight.
>>> model = get_model("convnext-base_in21k-pre_3rdparty_in1k")

# Get model and load the default checkpoint.
>>> model = get_model("convnext-base_in21k-pre_3rdparty_in1k", pretrained=True)

# Get model and load the specified checkpoint.
>>> model = get_model("convnext-base_in21k-pre_3rdparty_in1k", pretrained="your_local_checkpoint_path")

# Get model with extra initialization arguments, for example, modify the num_classes in head.
>>> model = get_model("convnext-base_in21k-pre_3rdparty_in1k", head=dict(num_classes=10))

# Another example, remove the neck and head, and output from stage 1, 2, 3 in backbone
>>> model_headless = get_model("resnet18_8xb32_in1k", head=None, neck=None, backbone=dict(out_indices=(1, 2, 3)))
```

The obtained model is a usual PyTorch module.

```python
>>> import torch
>>> from mmpretrain import get_model
>>> model = get_model('convnext-base_in21k-pre_3rdparty_in1k', pretrained=True)
>>> x = torch.rand((1, 3, 224, 224))
>>> y = model(x)
>>> print(type(y), y.shape)
<class 'torch.Tensor'> torch.Size([1, 1000])
```

## Inference on given images

Here is an example to inference an [image](https://github.com/open-mmlab/mmpretrain/raw/main/demo/demo.JPEG) by the ResNet-50 pre-trained classification model.

```python
>>> from mmpretrain import inference_model
>>> image = 'https://github.com/open-mmlab/mmpretrain/raw/main/demo/demo.JPEG'
>>> # If you have no graphical interface, please set `show=False`
>>> result = inference_model('resnet50_8xb32_in1k', image, show=True)
>>> print(result['pred_class'])
sea snake
```

The `inference_model` API is only for demo and cannot keep the model instance or inference on multiple
samples. You can use the inferencers for multiple calling.

```python
>>> from mmpretrain import ImageClassificationInferencer
>>> image = 'https://github.com/open-mmlab/mmpretrain/raw/main/demo/demo.JPEG'
>>> inferencer = ImageClassificationInferencer('resnet50_8xb32_in1k')
>>> # Note that the inferencer output is a list of result even if the input is a single sample.
>>> result = inferencer('https://github.com/open-mmlab/mmpretrain/raw/main/demo/demo.JPEG')[0]
>>> print(result['pred_class'])
sea snake
>>>
>>> # You can also use is for multiple images.
>>> image_list = ['demo/demo.JPEG', 'demo/bird.JPEG'] * 16
>>> results = inferencer(image_list, batch_size=8)
>>> print(len(results))
32
>>> print(results[1]['pred_class'])
house finch, linnet, Carpodacus mexicanus
```

Usually, the result for every sample is a dictionary. For example, the image classification result is a dictionary containing `pred_label`, `pred_score`, `pred_scores` and `pred_class` as follows:

```python
{
    "pred_label": 65,
    "pred_score": 0.6649366617202759,
    "pred_class":"sea snake",
    "pred_scores": array([..., 0.6649366617202759, ...], dtype=float32)
}
```

You can configure the inferencer by arguments, for example, use your own config file and checkpoint to
inference images by CUDA.

```python
>>> from mmpretrain import ImageClassificationInferencer
>>> image = 'https://github.com/open-mmlab/mmpretrain/raw/main/demo/demo.JPEG'
>>> config = 'configs/resnet/resnet50_8xb32_in1k.py'
>>> checkpoint = 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth'
>>> inferencer = ImageClassificationInferencer(model=config, pretrained=checkpoint, device='cuda')
>>> result = inferencer(image)[0]
>>> print(result['pred_class'])
sea snake
```

## Inference by a Gradio demo

We also provide a gradio demo for all supported tasks and you can find it in [projects/gradio_demo/launch.py](https://github.com/open-mmlab/mmpretrain/blob/main/projects/gradio_demo/launch.py).

Please install `gradio` by `pip install -U gradio` at first.

Here is the interface preview:

<img src="https://user-images.githubusercontent.com/26739999/236147750-90ccb517-92c0-44e9-905e-1473677023b1.jpg" width="100%"/>

## Extract Features From Image

Compared with `model.extract_feat`, `FeatureExtractor` is used to extract features from the image files directly, instead of a batch of tensors.
In a word, the input of `model.extract_feat` is `torch.Tensor`, the input of `FeatureExtractor` is images.

```python
>>> from mmpretrain import FeatureExtractor, get_model
>>> model = get_model('resnet50_8xb32_in1k', backbone=dict(out_indices=(0, 1, 2, 3)))
>>> extractor = FeatureExtractor(model)
>>> features = extractor('https://github.com/open-mmlab/mmpretrain/raw/main/demo/demo.JPEG')[0]
>>> features[0].shape, features[1].shape, features[2].shape, features[3].shape
(torch.Size([256]), torch.Size([512]), torch.Size([1024]), torch.Size([2048]))
```
