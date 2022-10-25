# Custom Evaluation Metrics

In MMClassification, metrics are provided based on [`MMEngine: BaseMetric`](mmengine.evaluator.BaseMetric) and should be wrapped in [`MMEngine: Evaluator`](mmengine.evaluator.Evaluator) when starting a runner. To specify the desired metrics used in validation and testing, we have to modify the `val_evaluator` and `test_evaluator` fields in the config file.

Here is an example for using basic `Accuracy` metric to evaluate the performance of single-label classification task.

```python
val_evaluator = dict(type='Accuracy', topk=(1, 5))
test_evaluator = val_evaluator
```

For single-label classification task, metrics are relatively unified and simple. Accuracy is commonly used for most cases. Furthermore, precision, recall, f1-scores are computed as well for detailed analysis.

When it comes for multi-label classification task, additional average precision(AP) might be used.

Therefore, four metrics are supported in MMClassification and descrided as below:

## Accuracy

`Accuracy` is only used for single-label task. For binary classification, given the true positive(TP), false positive(FP), true negative(TN), false negative(FN), accuracy can be calculated by the following equation:

```{math}
Acc = \frac{TP+TN}{TP+FP+TN+FN}
```

For multi-class classification, it is just the fraction of correct classifications in all classfications:

```{math}
Acc = \frac{\text{correct classifications}}{\text{all classifications}}
```

Besides the `topk` option for topk-accuracy, `Accuracy` also accept `thrs` options to calculate metric based on one or multiple threshold scores.

```python
# use one thr
val_evaluator = dict(type='Accuracy', topk=(1, 5), thrs=0.5)
test_evaluator = val_evaluator
# or use sequence of thrs
val_evaluator = dict(type='Accuracy', topk=(1, 5), thrs=(0.3, 0.8, None))
test_evaluator = val_evaluator
```

## SingleLabelMetric

`SingleLabelMetric` is only used for single-label task obviously, and calculates precision(P), recall(R), f1-score and support simultaneously. Except for the support which means the total number of occurrences of each category in the targets, the rest metrics can be formulated use variables above:

```{math}
P = \frac{TP}{TP+FP}, R = \frac{TP}{TP+FN}, \text{F1-score} = \frac{2*R*P}{R+P}
```

In other words, precision is the fraction of correct predictions in all predictions, recall is the fraction of correct predictions in all targets.

Here is an example of using `SingleLabelMetric` which only outputs precision and recall:

```python
val_evaluator = dict(
    type='SingleLabelMetric', items=('precision, recall'))
test_evaluator = val_evaluator
```

```{tip}
`SingleLabelMetric` does not support `topk` option, and more options refers [the documentation](mmcls.evaluation.metrics.SingleLabelMetric) for more details.
```

## MultiLabelMetric

`MultiLabelMetric` is used for multi-label task, and the detailed collection of metrics are the same as `SingleLabelMetric`. Refers [the documentation](mmcls.evaluation.metrics.MultiLabelMetric) for more details of implementation differences.

Here is an example of using `MultiLabelMetric` which outputs precision, recall and f1-score as defaults, and with the `mirco` average method which refers for globally average rather than class-wise average:

```python
val_evaluator = dict(
    type='MultiLabelMetric', average='mirco')
test_evaluator = val_evaluator
```

## AveragePrecision

`AveragePrecision` is used for multi-label task. Refers [the documentation](mmcls.evaluation.metrics.AveragePrecision) for more details of implementation.

{math}`P_n` and {math}`R_n` are the precision and recall at cut-off `n` in the list respectively. Average precision is calculated by the following equation:

```{math}
\text{AP} = \sum_n (R_n - R_{n-1}) P_n
```

Here is an example of using `AveragePrecision` metric with the `mirco` average method.

```python
val_evaluator = dict(type='AveragePrecision', average='mirco')
test_evaluator = val_evaluator
```

In addition, evaluator also supports multiple metrics, such as use `MultiLabelMetric` and `AveragePrecision` at the same time for better evaluation.

```python
val_evaluator = [dict(type='MultiLabelMetric'), dict(type='AveragePrecision')]
test_evaluator = val_evaluator
```

## Customized Metric

MMClassification supports the implementation of customized evaluation metrics for users who pursue higher customization.

You need to create a new file under `mmcls/evaluation/metrics`, and implement the new metric in the file, for example, in `mmcls/evaluation/metrics/my_metric.py`. And create a customized evaluation metric class `MyMetric` which inherits [`MMEngine: BaseMetric`](mmengine.evaluator.metrics.BaseMetric).

The data format processing method `process` and the metric calculation method `compute_metrics` need to be overwritten respectively. Add it to the `METRICS` registry to implement any customized evaluation metric.

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

Then, import it in the `mmcls/evaluation/metrics/__init__.py` to add it into the `mmcls.evaluation` package.

```python
# In mmcls/evaluation/metrics/__init__.py
...
from .my_metric import MyMetric

__all__ = [..., 'MyMetric']
```

Finally, use `MyMetric` in the `val_evaluator` and `test_evaluator` field of config files.

```python
val_evaluator = dict(type='MyMetric', ...)
test_evaluator = val_evaluator
```

```{note}
More details can be found in {external+mmengine:doc}`MMEngine Documentation: BaseMetric <design/evaluation>`.
```
