# 使用现有模型推理

MMPretrain 在 [Model Zoo](../modelzoo_statistics.md) 中提供了预训练模型。
本说明将展示**如何使用现有模型对给定图像进行推理**。

至于如何在标准数据集上测试现有模型，请看这个[指南](./test.md)

## 推理单张图片

MMPretrain 为图像推理提供高级 Python API：

- [`get_model`](mmpretrain.apis.get_model): 根据名称获取一个模型。
- [`inference_model`](mmpretrain.apis.inference_model)：对给定图片进行推理。

下面是一个示例，如何使用一个 ImageNet-1k 预训练权重初始化模型并推理给定图像。

```{note}
可以运行 `wget https://github.com/open-mmlab/mmclassification/raw/master/demo/demo.JPEG` 下载样例图片，或使用其他图片。
```

```python
from mmpretrain import get_model, inference_model

img_path = 'demo.JPEG'   # 可以指定自己的图片路径

# 构建模型
model = get_model('resnet50_8xb32_in1k', pretrained=True, device="cpu")  # `device` 可以为 'cuda:0'
# 执行推理
result = inference_model(model, img_path)
```

`result` 为一个包含了 `pred_label`, `pred_score`, `pred_scores` 和 `pred_class`的字典，结果如下：

```text
{"pred_label":65,"pred_score":0.6649366617202759,"pred_class":"sea snake", "pred_scores": [..., 0.6649366617202759, ...]}
```

演示可以在 [demo/image_demo.py](https://github.com/open-mmlab/mmpretrain/blob/main/demo/image_demo.py) 中找到。
