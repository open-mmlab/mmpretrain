# 自定义评估指标

## 使用 MMPretrain 中的指标

在 MMPretrain 中，我们为单标签分类和多标签分类提供了多种指标：

**单标签分类**:

- [`Accuracy`](mmpretrain.evaluation.Accuracy)
- [`SingleLabelMetric`](mmpretrain.evaluation.SingleLabelMetric)，包括精度、召回率、f1-score 和支持度。

**多标签分类**:

- [`AveragePrecision`](mmpretrain.evaluation.AveragePrecision)， 或 AP (mAP)。
- [`MultiLabelMetric`](mmpretrain.evaluation.MultiLabelMetric)，包括精度、召回率、f1-score 和支持度。

要在验证和测试期间使用这些指标，我们需要修改配置文件中的 `val_evaluator` 和 `test_evaluator` 字段。

以下为几个例子：

1. 在验证和测试期间计算 top-1 和 top-5 准确率。

   ```python
   val_evaluator = dict(type='Accuracy', topk=(1, 5))
   test_evaluator = val_evaluator
   ```

2. 在验证和测试期间计算 top-1 准确率、top-5 准确度、精确度和召回率。

   ```python
   val_evaluator = [
     dict(type='Accuracy', topk=(1, 5)),
     dict(type='SingleLabelMetric', items=['precision', 'recall']),
   ]
   test_evaluator = val_evaluator
   ```

3. 计算 mAP（平均平均精度）、CP（类别平均精度）、CR（类别平均召回率）、CF（类别平均 F1 分数）、OP（总体平均精度）、OR（总体平均召回率）和 OF1（总体平均 F1 分数）。

   ```python
   val_evaluator = [
     dict(type='AveragePrecision'),
     dict(type='MultiLabelMetric', average='macro'),  # class-wise mean
     dict(type='MultiLabelMetric', average='micro'),  # overall mean
   ]
   test_evaluator = val_evaluator
   ```

## 添加新的指标

MMPretrain 支持为追求更高定制化的用户实现定制化的评估指标。

您需要在 `mmpretrain/evaluation/metrics` 下创建一个新文件，并在该文件中实现新的指标，例如，在 `mmpretrain/evaluation/metrics/my_metric.py` 中。并创建一个自定义的评估指标类 `MyMetric` 继承 [MMEngine 中的 BaseMetric](mmengine.evaluator.BaseMetric)。

需要分别覆盖数据格式处理方法`process`和度量计算方法`compute_metrics`。 将其添加到“METRICS”注册表以实施任何自定义评估指标。

```python
from mmengine.evaluator import BaseMetric
from mmpretrain.registry import METRICS

@METRICS.register_module()
class MyMetric(BaseMetric):

    def process(self, data_batch: Sequence[Dict], data_samples: Sequence[Dict]):
    """ The processed results should be stored in ``self.results``, which will
        be used to computed the metrics when all batches have been processed.
        `data_batch` stores the batch data from dataloader,
        and `data_samples` stores the batch outputs from model.
    """
        ...

    def compute_metrics(self, results: List):
    """ Compute the metrics from processed results and returns the evaluation results.
    """
        ...
```

然后，将其导入 `mmpretrain/evaluation/metrics/__init__.py` 以将其添加到 `mmpretrain.evaluation` 包中。

```python
# In mmpretrain/evaluation/metrics/__init__.py
...
from .my_metric import MyMetric

__all__ = [..., 'MyMetric']
```

最后，在配置文件的 `val_evaluator` 和 `test_evaluator` 字段中使用 `MyMetric`。

```python
val_evaluator = dict(type='MyMetric', ...)
test_evaluator = val_evaluator
```

```{note}
更多的细节可以参考 {external+mmengine:doc}`MMEngine 文档: Evaluation <design/evaluation>`.
```
