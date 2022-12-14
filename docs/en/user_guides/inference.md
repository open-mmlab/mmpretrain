# Inference with existing models

MMClassification provides pre-trained models for classification in [Model Zoo](../modelzoo_statistics.md).
This note will show **how to use existing models to inference on given images**.

As for how to test existing models on standard datasets, please see this [guide](./train_test.md#test)

## Inference on a given image

MMClassification provides high-level Python APIs for inference on a given image:

- [init_model](mmcls.apis.init_model): Initialize a model with a config and checkpoint
- [inference_model](mmcls.apis.inference_model): Inference on a given image

Here is an example of building the model and inference on a given image by using ImageNet-1k pre-trained checkpoint.

```{note}
If you use mmcls as a 3rd-party package, you need to download the conifg and the demo image in the example.

Run 'mim download mmcls --config resnet50_8xb32_in1k --dest .' to download the required config.

Run 'wget https://github.com/open-mmlab/mmclassification/blob/master/demo/demo.JPEG' to download the desired demo image.
```

```python
from mmcls.apis import inference_model, init_model
from mmcls.utils import register_all_modules

config_path = './configs/resnet/resnet50_8xb32_in1k.py'
checkpoint_path = 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth' # can be a local path
img_path = 'demo/demo.JPEG'   # you can specify your own picture path

# register all modules and set mmcls as the default scope.
register_all_modules()
# build the model from a config file and a checkpoint file
model = init_model(config_path, checkpoint_path, device="cpu")  # device can be 'cuda:0'
# test a single image
result = inference_model(model, img_path)
```

`result` is a dictionary containing `pred_label`, `pred_score`, `pred_scores` and `pred_class`, the result is as follows:

```text
{"pred_label":65,"pred_score":0.6649366617202759,"pred_class":"sea snake", "pred_scores": [..., 0.6649366617202759, ...]}
```

An image demo can be found in [demo/image_demo.py](https://github.com/open-mmlab/mmclassification/blob/1.x/demo/image_demo.py).
