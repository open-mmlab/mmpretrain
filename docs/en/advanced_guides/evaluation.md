# Customize Evaluation Metrics

## Use metrics in MMPretrain

In MMPretrain, we have provided multiple metrics for both single-label classification and multi-label
classification:

**Single-label Classification**:

- [`Accuracy`](mmpretrain.evaluation.Accuracy)
- [`SingleLabelMetric`](mmpretrain.evaluation.SingleLabelMetric), including precision, recall, f1-score and
  support.

**Multi-label Classification**:

- [`AveragePrecision`](mmpretrain.evaluation.AveragePrecision), or AP (mAP).
- [`MultiLabelMetric`](mmpretrain.evaluation.MultiLabelMetric), including precision, recall, f1-score and
  support.

To use these metrics during validation and testing, we need to modify the `val_evaluator` and `test_evaluator`
fields in the config file.

Here is several examples:

1. Calculate top-1 and top-5 accuracy during both validation and test.

   ```python
   val_evaluator = dict(type='Accuracy', topk=(1, 5))
   test_evaluator = val_evaluator
   ```

2. Calculate top-1 accuracy, top-5 accuracy, precision and recall during both validation and test.

   ```python
   val_evaluator = [
     dict(type='Accuracy', topk=(1, 5)),
     dict(type='SingleLabelMetric', items=['precision', 'recall']),
   ]
   test_evaluator = val_evaluator
   ```

3. Calculate mAP (mean AveragePrecision), CP (Class-wise mean Precision), CR (Class-wise mean Recall), CF
   (Class-wise mean F1-score), OP (Overall mean Precision), OR (Overall mean Recall) and OF1 (Overall mean
   F1-score).

   ```python
   val_evaluator = [
     dict(type='AveragePrecision'),
     dict(type='MultiLabelMetric', average='macro'),  # class-wise mean
     dict(type='MultiLabelMetric', average='micro'),  # overall mean
   ]
   test_evaluator = val_evaluator
   ```

## Add new metrics

MMPretrain supports the implementation of customized evaluation metrics for users who pursue higher customization.

You need to create a new file under `mmpretrain/evaluation/metrics`, and implement the new metric in the file, for example, in `mmpretrain/evaluation/metrics/my_metric.py`. And create a customized evaluation metric class `MyMetric` which inherits [`BaseMetric in MMEngine`](mmengine.evaluator.BaseMetric).

The data format processing method `process` and the metric calculation method `compute_metrics` need to be overwritten respectively. Add it to the `METRICS` registry to implement any customized evaluation metric.

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

Then, import it in the `mmpretrain/evaluation/metrics/__init__.py` to add it into the `mmpretrain.evaluation` package.

```python
# In mmpretrain/evaluation/metrics/__init__.py
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
More details can be found in {external+mmengine:doc}`MMEngine Documentation: Evaluation <design/evaluation>`.
```
