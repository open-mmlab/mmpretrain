# 混淆矩阵

MMPretrain 提供 `tools/analysis_tools/confusion_matrix.py` 工具来分析预测结果的混淆矩阵。关于混淆矩阵的介绍，可参考[链接](https://zh.wikipedia.org/zh-cn/%E6%B7%B7%E6%B7%86%E7%9F%A9%E9%98%B5)。

## 命令行使用

**命令行**：

```shell
python tools/analysis_tools/confusion_matrix.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT} \
    [--show] \
    [--show-path] \
    [--include-values] \
    [--cmap ${CMAP}] \
    [--cfg-options ${CFG-OPTIONS}]
```

**所有参数的说明**：

- `config`：模型配置文件的路径。
- `checkpoint`：权重路径。
- `--show`：是否展示混淆矩阵的 matplotlib 可视化结果，默认不展示。
- `--show-path`：如果 `show` 为 True，可视化结果的保存路径。
- `--include-values`：是否在可视化结果上添加数值。
- `--cmap`：可视化结果使用的颜色映射图，即 `cmap`，默认为 `viridis`。
- `--cfg-options`：对配置文件的修改，参考[学习配置文件](../user_guides/config.md)。

**使用示例**：

```shell
python tools/analysis_tools/confusion_matrix.py \
    configs/resnet/resnet50_8xb16_cifar10.py \
    https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_b16x8_cifar10_20210528-f54bfad9.pth \
    --show
```

**输出图片**：

<div align=center><img src="https://user-images.githubusercontent.com/26739999/210298124-49ae00f7-c8fd-488a-a4da-58c285e9c1f1.png" style=" width: auto; height: 40%; "></div>

## 基础用法

```python
>>> import torch
>>> from mmpretrain.evaluation import ConfusionMatrix
>>> y_pred = [0, 1, 1, 3]
>>> y_true = [0, 2, 1, 3]
>>> ConfusionMatrix.calculate(y_pred, y_true, num_classes=4)
tensor([[1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]])
>>> # plot the confusion matrix
>>> import matplotlib.pyplot as plt
>>> y_score = torch.rand((1000, 10))
>>> y_true = torch.randint(10, (1000, ))
>>> matrix = ConfusionMatrix.calculate(y_score, y_true)
>>> ConfusionMatrix().plot(matrix)
>>> plt.show()
```

## 结合评估器使用

```python
>>> import torch
>>> from mmpretrain.evaluation import ConfusionMatrix
>>> from mmpretrain.structures import DataSample
>>> from mmengine.evaluator import Evaluator
>>> data_samples = [
...     DataSample().set_gt_label(i%5).set_pred_score(torch.rand(5))
...     for i in range(1000)
... ]
>>> evaluator = Evaluator(metrics=ConfusionMatrix())
>>> evaluator.process(data_samples)
>>> evaluator.evaluate(1000)
{'confusion_matrix/result': tensor([[37, 37, 48, 43, 35],
         [35, 51, 32, 46, 36],
         [45, 28, 39, 42, 46],
         [42, 40, 40, 35, 43],
         [40, 39, 41, 37, 43]])}
```
