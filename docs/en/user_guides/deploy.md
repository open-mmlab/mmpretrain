# Deployment

The deployment of OpenMMLab codebases, including MMClassification, MMDetection and so on are supported by [MMDeploy](https://github.com/open-mmlab/mmdeploy).
The latest deployment guide for MMClassification can be found from [here](https://mmdeploy.readthedocs.io/en/1.x/04-supported-codebases/mmcls.html).

This tutorial is organized as follows:

- [Installation](#installation)
- [Convert model](#convert-model)
- [Model Specification](#model-specification)
- [Model inference](#model-inference)
  - [Backend model inference](#backend-model-inference)
  - [SDK model inference](#sdk-model-inference)
- [Supported models](#supported-models)

## Installation

Please follow the [quick guide](https://github.com/open-mmlab/mmclassification/tree/1.x#installation) to install mmcls. And then install mmdeploy from source by following [this](https://mmdeploy.readthedocs.io/en/1.x/get_started.html#installation) guide.

```{note}
If you install mmdeploy prebuilt package, please also clone its repository by 'git clone https://github.com/open-mmlab/mmdeploy.git --depth=1' to get the deployment config files.
```

## Convert model

Suppose mmclassification and mmdeploy repositories are in the same directory, and the working directory is the root path of mmclassification.

Take a pretrained [resnet18](https://github.com/open-mmlab/mmclassification/blob/1.x/configs/resnet/resnet18_8xb32_in1k.py) model on imagenet as an example.
You can download its checkpoint from [here](https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_8xb32_in1k_20210831-fbbb1da6.pth), and then convert it to onnx model as follows:

```python
from mmdeploy.apis import torch2onnx
from mmdeploy.backend.sdk.export_info import export2SDK

img = 'demo/demo.JPEG'
work_dir = 'mmdeploy_models/mmcls/onnx'
save_file = 'end2end.onnx'
deploy_cfg = '../mmdeploy/configs/mmcls/classification_onnxruntime_dynamic.py'
model_cfg = 'configs/resnet/resnet18_8xb32_in1k.py'
model_checkpoint = 'resnet18_8xb32_in1k_20210831-fbbb1da6.pth'
device = 'cpu'

# 1. convert model to onnx
torch2onnx(img, work_dir, save_file, deploy_cfg, model_cfg,
  model_checkpoint, device)

# 2. extract pipeline info for inference by MMDeploy SDK
export2SDK(deploy_cfg, model_cfg, work_dir, pth=model_checkpoint, device=device)
```

It is crucial to specify the correct deployment config during model conversion. MMDeploy has already provided builtin deployment config [files](https://github.com/open-mmlab/mmdeploy/tree/1.x/configs/mmcls) of all supported backends for mmclassification. The config filename pattern is:

```
classification_{backend}-{precision}_{static | dynamic}_{shape}.py
```

- **{backend}:** inference backend, such as onnxruntime, tensorrt, pplnn, ncnn, openvino, coreml and etc.
- **{precision}:** fp16, int8. When it's empty, it means fp32
- **{static | dynamic}:** static shape or dynamic shape
- **{shape}:** input shape or shape range of a model

Therefore, in the above example, you can also convert `resnet18` to other backend models by changing the deployment config file `classification_onnxruntime_dynamic.py` to [others](https://github.com/open-mmlab/mmdeploy/tree/1.x/configs/mmcls), e.g., converting to tensorrt-fp16 model by `classification_tensorrt-fp16_dynamic-224x224-224x224.py`.

```{tip}
When converting mmcls models to tensorrt models, --device should be set to "cuda"
```

## Model Specification

Before moving on to model inference chapter, let's know more about the converted model structure which is very important for model inference.

The converted model locates in the working directory like `mmdeploy_models/mmcls/onnx` in the previous example. It includes:

```
mmdeploy_models/mmcls/onnx
├── deploy.json
├── detail.json
├── end2end.onnx
└── pipeline.json
```

in which,

- **end2end.onnx**: backend model which can be inferred by ONNX Runtime
- ***xxx*.json**: the necessary information for mmdeploy SDK

The whole package **mmdeploy_models/mmcls/onnx** is defined as **mmdeploy SDK model**, i.e., **mmdeploy SDK model** includes both backend model and inference meta information.

## Model inference

### Backend model inference

Take the previous converted `end2end.onnx` model as an example, you can use the following code to inference the model.

```python
from mmdeploy.apis.utils import build_task_processor
from mmdeploy.utils import get_input_shape, load_config
import torch

deploy_cfg = '../mmdeploy/configs/mmcls/classification_onnxruntime_dynamic.py'
model_cfg = 'configs/resnet/resnet18_8xb32_in1k.py'
device = 'cpu'
backend_model = ['mmdeploy_models/mmcls/onnx/end2end.onnx']
image = 'demo/cat-dog.png'

# read deploy_cfg and model_cfg
deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)

# build task and backend model
task_processor = build_task_processor(model_cfg, deploy_cfg, device)
model = task_processor.build_backend_model(backend_model)

# process input image
input_shape = get_input_shape(deploy_cfg)
model_inputs, _ = task_processor.create_input(image, input_shape)

# do model inference
with torch.no_grad():
    result = model.test_step(model_inputs)

# visualize results
task_processor.visualize(
    image=image,
    model=model,
    result=result[0],
    window_name='visualize',
    output_file='output_classification.png')
```

### SDK model inference

You can also perform SDK model inference like following,

```python
from mmdeploy_python import Classifier
import cv2

img = cv2.imread('demo/cat-dog.png')

# create a classifier
classifier = Classifier(model_path='mmdeploy_models/mmcls/onnx', device_name='cpu', device_id=0)
# perform inference
result = classifier(img)
# show inference result
for label_id, score in result:
    print(label_id, score)
```

Besides python API, MMDeploy SDK also provides other FFI (Foreign Function Interface), such as C, C++, C#, Java and so on. You can learn their usage from [demos](https://github.com/open-mmlab/mmdeploy/tree/1.x/demo).

## Supported models

Please refer to [here](https://mmdeploy.readthedocs.io/en/1.x/04-supported-codebases/mmcls.html#supported-models) for the supported model list.
