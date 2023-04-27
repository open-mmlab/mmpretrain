# Confusion Matrix

MMPretrain provides `tools/analysis_tools/confusion_matrix.py` tool to calculate and visualize the confusion matrix. For an introduction to the confusion matrix, see [link](https://en.wikipedia.org/wiki/Confusion_matrix).

## Command-line Usage

**Command**：

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

**Description of all arguments**：

- `config`: The path of the model config file.
- `checkpoint`: The path of the checkpoint.
- `--show`: If or not to show the matplotlib visualization result of the confusion matrix, the default is `False`.
- `--show-path`: If `show` is True, the path where the results are saved is visualized.
- `--include-values`: Whether to add values to the visualization results.
- `--cmap`: The color map used for visualization results, `cmap`, which defaults to `viridis`.

* `--cfg-options`: Modifications to the configuration file, refer to [Learn about Configs](../user_guides/config.md).

**Examples of use**:

```shell
python tools/analysis_tools/confusion_matrix.py \
    configs/resnet/resnet50_8xb16_cifar10.py \
    https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_b16x8_cifar10_20210528-f54bfad9.pth \
    --show
```

**output image**:

<div align=center><img src="https://user-images.githubusercontent.com/26739999/210298124-49ae00f7-c8fd-488a-a4da-58c285e9c1f1.png" style=" width: auto; height: 40%; "></div>

## **Basic Usage**

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

## **Use with Evalutor**

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
