# 使用现有模型推理

MMClassification 在 [Model Zoo](../modelzoo_statistics.md) 中提供了用于分类的预训练模型。
本说明将展示**如何使用现有模型对给定图像进行推理**。

至于如何在标准数据集上测试现有模型，请看这个[指南](./train_test.md#测试)

## 推理单张图片

MMClassification 为图像推理提供高级 Python API：

- [init_model](mmcls.apis.init_model): 初始化一个模型。
- [inference_model](mmcls.apis.inference_model)：对给定图片进行推理。

下面是一个示例，如何使用一个 ImageNet-1k 预训练权重初始化模型并推理给定图像。

```{note}
如果您将 mmcls 当作第三方库使用，需要下载样例中的配置文件以及样例图片。

运行 'mim download mmcls --config resnet50_8xb32_in1k --dest .' 下载所需配置文件。

运行 'wget https://github.com/open-mmlab/mmclassification/blob/master/demo/demo.JPEG' 下载所需图片。
```

```python
from mmcls.apis import inference_model, init_model
from mmcls.utils import register_all_modules

config_path = './configs/resnet/resnet50_8xb32_in1k.py'
checkpoint_path = 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth' # 也可以设置为一个本地的路径
img_path = 'demo/demo.JPEG'   # 可以指定自己的图片路径

# 注册
register_all_modules()                 # 将所有模块注册在默认 mmcls 域中
# 构建模型
model = init_model(config_path, checkpoint_path, device="cpu") # `device` 可以为 'cuda:0'
# 执行推理
result = inference_model(model, img_path)
print(result)
```

`result` 为一个包含了 `pred_label`, `pred_score`, `pred_scores` 和 `pred_class`的字典，结果如下:

```text
{"pred_label":65,"pred_score":0.6649366617202759,"pred_class":"sea snake", "pred_scores": [..., 0.6649366617202759, ...]}
```

演示可以在 [demo/image_demo.py](https://github.com/open-mmlab/mmclassification/blob/1.x/demo/image_demo.py) 中找到。
