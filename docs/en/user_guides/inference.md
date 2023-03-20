# Inference with existing models

- [Inference with existing models](#inference-with-existing-models)
  - [Inference on a given image](#inference-on-a-given-image)

MMPretrain provides pre-trained models in [Model Zoo](../modelzoo_statistics.md).
This note will show **how to use existing models to inference on given images**.

As for how to test existing models on standard datasets, please see this [guide](./train_test.md#test)

## Inference on a given image

MMPretrain provides high-level Python APIs for inference on a given image:

- [`get_model`](mmpretrain.apis.get_model): Get a model with the model name.
- [`init_model`](mmpretrain.apis.init_model): Initialize a model with a config and checkpoint
- [`inference_model`](mmpretrain.apis.inference_model): Inference on a given image

Here is an example of building the model and inference on a given image by using ImageNet-1k pre-trained checkpoint.

```{note}
You can use `wget https://github.com/open-mmlab/mmclassification/raw/master/demo/demo.JPEG` to download the example image or use your own image.
```

```python
from mmpretrain import get_model, inference_model

img_path = 'demo.JPEG'   # you can specify your own picture path

# build the model from a config file and a checkpoint file
model = get_model('resnet50_8xb32_in1k', pretrained=True, device="cpu")  # device can be 'cuda:0'
# test a single image
result = inference_model(model, img_path)
```

`result` is a dictionary containing `pred_label`, `pred_score`, `pred_scores` and `pred_class`, the result is as follows:

```text
{"pred_label":65,"pred_score":0.6649366617202759,"pred_class":"sea snake", "pred_scores": [..., 0.6649366617202759, ...]}
```

An image demo can be found in [demo/image_demo.py](https://github.com/open-mmlab/mmclassification/blob/pretrain/demo/image_demo.py).
