# 自定义评估指标

## 使用 MMClassification 中的指标

在 MMClassification 中，我们为单标签分类和多标签分类提供了多个指标:

**单标签分类**

- [`Accuracy`](mmcls.evaluation.Accuracy)
- [`SingleLabelMetric`](mmcls.evaluation.SingleLabelMetric), 包含 `precision,recall,f1-score` 和 `support` 指标

**多标签分类**

- [`AveragePrecision`](mmcls.evaluation.AveragePrecision), 或者 AP (mAP).
- [`MultiLabelMetric`](mmcls.evaluation.MultiLabelMetric), 包含 `precision,recall,f1-score` 和 `support` 指标

为了在验证和测试期间使用这些指标，我们需要修改配置文件中的 `val_evaluator` 和 `test_evaluator` 字段。

下面是几个例子:

1. 在验证和测试期间计算 top-1 和 top-5 的准确率。

   ```python
   val_evaluator = dict(type='Accuracy', topk=(1, 5))
   test_evaluator = val_evaluator
   ```

2. 在验证和测试期间计算 `top-1` 和 `top-5` 的准确率，以及精确率和召回率。

   ```python
   val_evaluator = [
     dict(type='Accuracy', topk=(1, 5)),
     dict(type='SingleLabelMetric', items=['precision', 'recall']),
   ]
   test_evaluator = val_evaluator
   ```

3. 计算 mAP(平均平均精度)，CP(各类别平均精度)，CR(各类别平均召回率)，CF(各类别平均 F1-score)， OP(总体平均精确率)， OR(总体平均召回率) 和 OF1(总体平均
   F1-score)。

   ```python
   val_evaluator = [
     dict(type='AveragePrecision'),
     dict(type='MultiLabelMetric', average='macro'),  # class-wise mean
     dict(type='MultiLabelMetric', average='micro'),  # overall mean
   ]
   test_evaluator = val_evaluator
   ```

## 添加新指标

MMClassification 支持为追求更高自定义性的用户实现自定义评估指标。

用户需要在 `mmcls/evaluation/metrics` 下创建一个新的文件，并在该文件中实现新的指标，例如在 `mmcls/evaluation/metrics/my_metric.py` 中实现。并创建一个自定义评估指标类 `MyMetric`，该类继承了 MMEngine 中的 [`BaseMetric in MMEngine`](mmengine.evaluator.BaseMetric)。

类中需要分别重写数据格式处理方法 `process` 和指标计算方法 `compute_metrics`，并将其添加到 `METRICS` 注册表中，以实现任何自定义评估指标。

```python
from mmengine.evaluator import BaseMetric
from mmcls.registry import METRICS

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

然后，将其导入到 `mmcls/evaluation/metrics/__init__.py` 中，并将其添加到 `mmcls.evaluation`中。

```python
# In mmcls/evaluation/metrics/__init__.py
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
更多细节可以查看 {external+mmengine:doc}`MMEngine Documentation: Evaluation <design/evaluation>`
```
